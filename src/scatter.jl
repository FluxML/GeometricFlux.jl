const name2op = Dict(:add => :+, :sub => :-, :mul => :*, :div => :/)

for op = [:add, :sub, :mul, :div]
    @eval function $(Symbol("scatter_", op, "!"))(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
        @simd for k = 1:length(xs)
            k = CartesianIndices(xs)[k]
            @inbounds ys[:, xs[k]...] .= $(name2op[op]).(ys[:, xs[k]...], us[:, k])
        end
        ys
    end
end

function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        @inbounds ys[:, xs[k]...] .= max.(ys[:, xs[k]...], us[:, k])
    end
    ys
end

function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        @inbounds ys[:, xs[k]...] .= min.(ys[:, xs[k]...], us[:, k])
    end
    ys
end

function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where T
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs)
    scatter_add!(ys_, us, xs)
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
