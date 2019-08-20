using CUDAnative
using CuArrays

const MAX_THREADS = 256

for op = [:add, :sub, :max, :min, :and, :or, :xor]
    @eval function $(Symbol("scatter_", op, "!"))(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = Tuple(CartesianIndices(xs)[li])
                CUDAnative.$(Symbol("atomic_", op, "!"))(
                    pointer(ys,
                            Base._to_linear_index(ys, i, xs[li])),
                    us[i, ind...]
                )
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

    @eval function $(Symbol("scatter_", op, "!"))(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:Integer}
        function kernel!(ys, us, xs)
            li = threadIdx().y + (blockIdx().y - 1) * blockDim().y
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if li <= length(xs) && i <= size(ys, 1)
                ind = Tuple(CartesianIndices(xs)[li])
                CUDAnative.$(Symbol("atomic_", op, "!"))(
                    pointer(ys,
                            Base._to_linear_index(ys, i, xs[li]...)),
                    us[i, ind...]
                )
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


op2func = Dict{Symbol, Function}(:add => +, :sub => -, :mul => *,:div => /, :max => max, :min => min)

for op = [:add, :sub, :mul, :div, :max, :min]
    @eval function $(Symbol("scatter_", op, "!"))(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            xi = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if xi <= size(ys, 1)
                for i = 1:length(xs)
                    ind = Tuple(CartesianIndices(xs)[i])
                    y = $(op2func[op])(ys[xi, xs[i]], us[xi, ind...])
                    CUDAnative.sync_threads()
                    ys[xi, xs[i]] = y
                    CUDAnative.sync_threads()
                end
            end

            return
        end
        threads = MAX_THREADS
        blocks = ceil(Int, size(ys, 1) / threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end

    @eval function $(Symbol("scatter_", op, "!"))(ys::CuArray{T}, us::CuArray{T}, xs::CuArray{<:Tuple}) where {T<:AbstractFloat}
        function kernel!(ys::CuDeviceArray{T}, us::CuDeviceArray{T}, xs)
            xi = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            @inbounds if xi <= size(ys, 1)
                for i = 1:length(xs)
                    ind = Tuple(CartesianIndices(xs)[i])
                    y = $(op2func[op])(ys[xi, xs[i]...], us[xi, ind...])
                    CUDAnative.sync_threads()
                    ys[xi, xs[i]...] = y
                    CUDAnative.sync_threads()
                end
            end

            return
        end
        threads = MAX_THREADS
        blocks = ceil(Int, size(ys, 1) / threads)
        CuArrays.@cuda blocks=blocks threads=threads kernel!(ys, us, xs)
        return ys
    end
end


function scatter_mean!(ys::CuMatrix{T}, us::CuArray{T}, xs::CuArray{Int}) where {T<:AbstractFloat}
    pdiv(x, y) = ifelse(iszero(y), x, x/y)
    yt = fill!(similar(ys), 0)
    ot = fill!(similar(ys), 0)
    os = fill!(similar(us), 1)
    scatter_add!(ot, os, xs)
    scatter_add!(yt, us, xs)
    m = pdiv.(yt, ot)
    return ys .+= m
end
