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
    @eval function $(Symbol("scatter_", op, "!"))(ys::Matrix{T}, us::Array{T}, xs::Array{Int}) where T
        l = length(xs)
        s = size(ys, 1)

        Threads.@threads for num = 1:l*s
            li = (num -1) ÷ s + 1
            i = (num - 1) % s + 1
            @inbounds ind = Tuple(CartesianIndices(xs)[li])
            @inbounds $(Symbol("atomic_", op, "!"))(
                pointer(ys,
                        Base._to_linear_index(ys,
                                              i,
                                              xs[li]
                                              )
                        ),
                us[i, ind...]
            )
        end

        return ys
    end

    @eval function $(Symbol("scatter_", op, "!"))(ys::Array{T}, us::Array{T}, xs::Array{<:Tuple}) where T
        l = length(xs)
        s = size(ys, 1)

        Threads.@threads for num = 1:l*s
            li = (num -1) ÷ s + 1
            i = (num - 1) % s + 1
            @inbounds ind = Tuple(CartesianIndices(xs)[li])
            @inbounds $(Symbol("atomic_", op, "!"))(
                pointer(ys,
                        Base._to_linear_index(ys,
                                              i,
                                              xs[li]...
                                              )
                        ),
                us[i, ind...]
            )
        end

        return ys
    end
end

function scatter_mean!(ys::Matrix{T}, us::Array{T}, xs::Array{Int}) where T
    l = length(xs)
    s = size(ys, 1)
    N = size(ys, 2)
    Ns = [sum(xs .== i) for i = 1:N]
    ys_ = fill!(similar(ys), 0)

    Threads.@threads for num = 1:l*s
        li = (num -1) ÷ s + 1
        i = (num - 1) % s + 1
        @inbounds ind = Tuple(CartesianIndices(xs)[li])
        @inbounds atomic_add!(
            pointer(ys_, Base._to_linear_index(ys_, i, xs[li])),
            us[i, ind...]
        )
    end

    Threads.@threads for i = 1:N
        for j = 1:s
            @inbounds atomic_div!(
                pointer(ys_, Base._to_linear_index(ys_, j, i)),
                T(Ns[i])
            )
            @inbounds atomic_add!(
                pointer(ys, Base._to_linear_index(ys, j, i)),
                ys_[j,i]
            )
        end
    end

    return ys
end

# function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:Tuple}) where T
#     l = length(xs)
#     s = size(ys, 1)
#     N = length(Set(xs))
#     Ns = [sum(xs .== i) for i = 1:N]
#     ys_ = fill!(similar(ys), 0)
#
#     Threads.@threads for num = 1:l*s
#         li = (num -1) ÷ s + 1
#         i = (num - 1) % s + 1
#         @inbounds ind = Tuple(CartesianIndices(xs)[li])
#         @inbounds atomic_add!(
#             pointer(ys_, Base._to_linear_index(ys_, i, xs[li]...)),
#             us[i, ind...]
#         )
#     end
#
#     Threads.@threads for i = 1:N
#         for j = 1:s
#             @inbounds atomic_div!(
#                 pointer(ys_, Base._to_linear_index(ys_, j, i)),
#                 T(Ns[i])
#             )
#             @inbounds atomic_add!(
#                 pointer(ys, Base._to_linear_index(ys, j, i)),
#                 ys_[j,i]
#             )
#         end
#     end
#
#     return ys
# end

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
