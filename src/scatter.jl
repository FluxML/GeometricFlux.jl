# from https://github.com/chengchingwen/Transformers.jl/tree/master/src/fix

for typ ∈ atomictypes
    lt = llvmtypes[typ]
    ilt = llvmtypes[inttype(typ)]
    # Note: atomic_cas! succeeded (i.e. it stored "new") if and only if the result is "cmp"
    if typ <: Integer
        @eval atomic_cas!(x::Ptr{$typ}, cmp::$typ, new::$typ) =
            llvmcall($"""
                     %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                     %rs = cmpxchg $lt* %ptr, $lt %1, $lt %2 acq_rel acquire
                     %rv = extractvalue { $lt, i1 } %rs, 0
                     ret $lt %rv
                     """, $typ, Tuple{Ptr{$typ},$typ,$typ},
                     x, cmp, new)
    else
        @eval atomic_cas!(x::Ptr{$typ}, cmp::$typ, new::$typ) =
            llvmcall($"""
                     %iptr = inttoptr i$WORD_SIZE %0 to $ilt*
                     %icmp = bitcast $lt %1 to $ilt
                     %inew = bitcast $lt %2 to $ilt
                     %irs = cmpxchg $ilt* %iptr, $ilt %icmp, $ilt %inew acq_rel acquire
                     %irv = extractvalue { $ilt, i1 } %irs, 0
                     %rv = bitcast $ilt %irv to $lt
                     ret $lt %rv
                     """, $typ, Tuple{Ptr{$typ},$typ,$typ},
                     x, cmp, new)
    end

    arithmetic_ops = [:add, :sub]
    for rmwop in [arithmetic_ops..., :xchg, :and, :nand, :or, :xor, :max, :min]
        rmw = string(rmwop)
        fn = Symbol("atomic_", rmw, "!")
        if (rmw == "max" || rmw == "min") && typ <: Unsigned
            # LLVM distinguishes signedness in the operation, not the integer type.
            rmw = "u" * rmw
        end
        if rmwop in arithmetic_ops && !(typ <: ArithmeticTypes) continue end
        if typ <: Integer
            @eval $fn(x::Ptr{$typ}, v::$typ) =
                llvmcall($"""
                         %ptr = inttoptr i$WORD_SIZE %0 to $lt*
                         %rv = atomicrmw $rmw $lt* %ptr, $lt %1 acq_rel
                         ret $lt %rv
                         """, $typ, Tuple{Ptr{$typ}, $typ}, x, v)
        else
            rmwop == :xchg || continue
            @eval $fn(x::Ptr{$typ}, v::$typ) =
                llvmcall($"""
                         %iptr = inttoptr i$WORD_SIZE %0 to $ilt*
                         %ival = bitcast $lt %1 to $ilt
                         %irv = atomicrmw $rmw $ilt* %iptr, $ilt %ival acq_rel
                         %rv = bitcast $ilt %irv to $lt
                         ret $lt %rv
                         """, $typ, Tuple{Ptr{$typ}, $typ}, x, v)
        end
    end
end

const opnames = Dict{Symbol, Symbol}(:+ => :add, :- => :sub, :* => :mul, :/ => :div)
for op in [:+, :-, :max, :min, :*, :/]
    opname = get(opnames, op, op)
    for typ ∈ atomictypes
        typ == Bool && continue
        !(op == :/ || op == :*) && typ <: Integer && continue
        @eval function $(Symbol("atomic_", opname, "!"))(var::Ptr{T}, val::T) where T<: $typ
            IT = inttype(T)
            old = unsafe_load(var)
            while true
                new = $op(old, val)
                cmp = old
                old = atomic_cas!(var, cmp, new)
                reinterpret(IT, old) == reinterpret(IT, cmp) && return new
                # Temporary solution before we have gc transition support in codegen.
                ccall(:jl_gc_safepoint, Cvoid, ())
            end
        end
    end
end

for op = [:add, :sub, :max, :min, :mul, :div]
    @eval function $(Symbol("scatter_", op, "!"))(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple},
                                                  s::Int=size(ys,1)) where T
        Threads.@threads for i = 1:s
            @inbounds for ind = CartesianIndices(xs)
                $(Symbol("atomic_", op, "!"))(
                    pointer(ys, Base._to_linear_index(ys, i, xs[ind]...)),
                    us[i, ind]
                )
            end
        end
        ys
    end
end

function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple},
                       s::Int=size(ys,1)) where T
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs, s)
    scatter_add!(ys_, us, xs, s)
    ys .+= map((x,y) -> ifelse(iszero(y), x, x/y), ys_, Ns)
    return ys
end

@adjoint function scatter_add!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_add!(ys_, us, xs)
    ys_, Δ -> (Δ, gather(zero(Δ)+Δ, xs), nothing)
end

@adjoint function scatter_sub!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_sub!(ys_, us, xs)
    ys_, Δ -> (Δ, -gather(zero(Δ)+Δ, xs), nothing)
end

@adjoint function scatter_mul!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    ys_ = copy(ys)
    scatter_mul!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_mul!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = gather(ys, xs) .* gather(zero(Δ)+Δ, xs)
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] *= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_div!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    ys_ = copy(ys)
    scatter_div!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_div!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = - gather(ys, xs) .* gather(zero(Δ)+Δ, xs) ./ us.^2
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] /= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

function gather_indices(X::Array{T}) where T
    Y = DefaultDict{T,Vector{CartesianIndex}}(CartesianIndex[])
    @inbounds for (ind, val) = pairs(X)
        push!(Y[val], ind)
    end
    Y
end

@adjoint function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    max = copy(ys)
    scatter_max!(max, us, xs)
    max, function (Δ)
       Δy = (ys .== max) .* Δ
       Δu = (us .== gather(max, xs)) .* gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    min = copy(ys)
    scatter_min!(min, us, xs)
    min, function (Δ)
       Δy = (ys .== min) .* Δ
       Δu = (us .== gather(min, xs)) .* gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_mean!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_mean!(ys_, us, xs)
    ys_, function (Δ)
        Δu = gather(zero(Δ)+Δ, xs)
        counts = zero.(xs)
        @inbounds for i = 1:size(ys, 2)
            counts += sum(xs.==i) * (xs.==i)
        end
        @inbounds for ind = CartesianIndices(counts)
            Δu[:, ind] ./= counts[ind]
        end
        (Δ, Δu, nothing)
    end
end

function scatter!(op::Symbol, ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    if op == :add
        return scatter_add!(ys, us, xs)
    elseif op == :sub
        return scatter_sub!(ys, us, xs)
    elseif op == :mul
        return scatter_mul!(ys, us, xs)
    elseif op == :div
        return scatter_div!(ys, us, xs)
    elseif op == :max
        return scatter_max!(ys, us, xs)
    elseif op == :min
        return scatter_min!(ys, us, xs)
    elseif op == :mean
        return scatter_mean!(ys, us, xs)
    end
end
