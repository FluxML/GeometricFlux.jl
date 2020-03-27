## Scatter operations

const ops = [:add, :sub, :mul, :div, :max, :min, :mean]
const name2op = Dict(:add => :+, :sub => :-, :mul => :*, :div => :/)

for op = [:add, :sub, :mul, :div]
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
        @simd for k = 1:length(xs)
            k = CartesianIndices(xs)[k]
            @inbounds ys[:, xs[k]...] .= $(name2op[op]).(view(ys, :, xs[k]...), view(us, :, k))
        end
        ys
    end
end

function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        @inbounds ys[:, xs[k]...] .= max.(view(ys, :, xs[k]...), view(us, :, k))
    end
    ys
end

function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        @inbounds ys[:, xs[k]...] .= min.(view(ys, :, xs[k]...), view(us, :, k))
    end
    ys
end

function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs)
    scatter_add!(ys_, us, xs)
    ys .+= save_div.(ys_, Ns)
    return ys
end



## Derivatives of scatter operations

@adjoint function scatter_add!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_add!(ys_, us, xs)
    ys_, Δ -> (Δ, gather(Δ, xs), nothing)
end

@adjoint function scatter_sub!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_sub!(ys_, us, xs)
    ys_, Δ -> (Δ, -gather(Δ, xs), nothing)
end

@adjoint function scatter_mul!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    ys_ = copy(ys)
    scatter_mul!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_mul!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = gather(ys, xs) .* gather(Δ, xs)
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] *= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_div!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    ys_ = copy(ys)
    scatter_div!(ys_, us, xs)
    ys_, function (Δ)
        Δy = Δ .+ zero(ys)
        scatter_div!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = - gather(ys, xs) .* gather(Δ, xs) ./ us.^2
        @inbounds for ind = CartesianIndices(xs)
            inds = filter(x -> x != ind, rev_xs[xs[ind]])
            for i = 1:size(us, 1)
                Δu[i, ind] /= prod(j -> us[i, j], inds)
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    max = copy(ys)
    scatter_max!(max, us, xs)
    max, function (Δ)
       Δy = (ys .== max) .* Δ
       Δu = (us .== gather(max, xs)) .* gather(Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    min = copy(ys)
    scatter_min!(min, us, xs)
    min, function (Δ)
       Δy = (ys .== min) .* Δ
       Δu = (us .== gather(min, xs)) .* gather(Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_mean!(ys::AbstractArray, us::AbstractArray, xs::AbstractArray)
    ys_ = copy(ys)
    scatter_mean!(ys_, us, xs)
    ys_, function (Δ)
        Δu = gather(Δ, xs)
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



## Bool

function scatter_add!(ys::Array{Bool}, us::Array{Bool}, xs::Array{<:IntOrTuple})
    scatter_add!(Int8.(ys), Int8.(us), xs)
end



## API

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

# Support different types of array
for op = ops
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{S}, xs::Array{<:IntOrTuple}) where {T<:Real,S<:Real}
        PT = promote_type(T, S)
        $fn(PT.(ys), PT.(us), xs)
    end
end
