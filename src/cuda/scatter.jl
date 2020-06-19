const MAX_THREADS = 1024

# Integer
for op = [:add, :sub, :max, :min, :and, :or, :xor]
    fn = Symbol("scatter_$(op)!")
    atm_op = Symbol("atomic_$(op)!")
    @eval function $fn(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = CartesianIndices(xs)[li]
                j = Base._to_linear_index(ys, i, xs[li])
                CUDAnative.$atm_op(pointer(ys, j), us[i, ind])
            end

            return
        end

        thread_x = min(MAX_THREADS, size(ys, 1))
        thread_y = min(MAX_THREADS ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $fn(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = CartesianIndices(xs)[li]
                j = Base._to_linear_index(ys, i, xs[li]...)
                CUDAnative.$atm_op(pointer(ys, j), us[i, ind])
            end

            return
        end

        thread_x = min(MAX_THREADS, size(ys, 1))
        thread_y = min(MAX_THREADS ÷ thread_x, length(xs))
        threads = (thread_x, thread_y)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        @cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end


# Floating point
for op = [:add, :sub, :mul, :div, :max, :min]
    fn = Symbol("scatter_$(op)!")
    atm_op = Symbol("atomic_$(op)!")
    @eval function $fn(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

            @inbounds if i <= size(ys, 1) && j <= length(xs)
                ind = CartesianIndices(xs)[j]
                k = Base._to_linear_index(ys, i, xs[j])
                CUDAnative.$atm_op(pointer(ys, k), us[i, ind])
            end

            return
        end

        thread_i = min(MAX_THREADS, size(ys, 1))
        thread_j = min(MAX_THREADS ÷ thread_i, length(xs))
        threads = (thread_i, thread_j)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $fn(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

            @inbounds if i <= size(ys, 1) && j <= length(xs)
                ind = CartesianIndices(xs)[j]
                k = Base._to_linear_index(ys, i, xs[j]...)
                CUDAnative.$atm_op(pointer(ys, k), us[i, ind])
            end

            return
        end

        thread_i = min(MAX_THREADS, size(ys, 1))
        thread_j = min(MAX_THREADS ÷ thread_i, length(xs))
        threads = (thread_i, thread_j)
        blocks = ceil.(Int, (size(ys, 1), length(xs)) ./ threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end


function scatter_mean!(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:AbstractFloat}
    yt = CuArrays.zero(ys)
    ot = CuArrays.zero(ys)
    os = CuArrays.one.(us)
    scatter_add!(ot, os, xs)
    scatter_add!(yt, us, xs)
    ys .+= save_div.(yt, ot)
    return ys
end

@adjoint function scatter_mul!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    ys_ = copy(ys)
    scatter_mul!(ys_, us, xs)
    ys_, function (Δ)
        Δy = zero(ys) .+ Δ
        scatter_mul!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = gather(ys, xs) .* gather(zero(Δ)+Δ, xs)
        @inbounds for ind = CartesianIndices(xs)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_xs[xs[ind...]])
            for i = 1:size(us, 1)
                multiplier = one(T)
                for j = inds
                    multiplier *= us[i, j...]
                end
                Δu[i, ind...] *= multiplier
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_div!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    ys_ = copy(ys)
    scatter_div!(ys_, us, xs)
    ys_, function (Δ)
        Δy = zero(ys) .+ Δ
        scatter_div!(Δy, us, xs)
        rev_xs = gather_indices(xs)
        Δu = - gather(ys, xs)
        Δu .*= gather(zero(Δ)+Δ, xs)
        Δu ./= us.^2
        @inbounds for ind = CartesianIndices(xs)
            ind = Tuple(ind)
            inds = filter(x -> x != ind, rev_xs[xs[ind...]])
            for i = 1:size(us, 1)
                denom = one(T)
                for j = inds
                    denom *= us[i, j...]
                end
                Δu[i, ind...] /= denom
            end
        end
        (Δy, Δu, nothing)
    end
end

@adjoint function scatter_max!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    max = copy(ys)
    scatter_max!(max, us, xs)
    max, function (Δ)
       Δy = numerical_cmp(ys, max) .* Δ
       Δu = gather(max, xs)
       Δu = numerical_cmp(us, Δu)
       Δu .*= gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end

@adjoint function scatter_min!(ys::CuArray{T}, us::CuArray{T}, xs::CuArray) where {T<:AbstractFloat}
    min = copy(ys)
    scatter_min!(min, us, xs)
    min, function (Δ)
       Δy = numerical_cmp(ys, min) .* Δ
       Δu = gather(min, xs)
       Δu = numerical_cmp(us, Δu)
       Δu .*= gather(zero(Δ)+Δ, xs)
       (Δy, Δu, nothing)
    end
end
