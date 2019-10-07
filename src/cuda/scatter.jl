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
    yt = CuArrays.zero(ys)
    ot = CuArrays.zero(ys)
    os = CuArrays.one.(us)
    scatter_add!(ot, os, xs)
    scatter_add!(yt, us, xs)
    ys .+= pdiv.(yt, ot)
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
                Δu[i, ind...] *= mapreduce(j -> us[i, j...], *, inds; init=one(T))
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
                Δu[i, ind...] /= mapreduce(j -> us[i, j...], *, inds; init=one(T))
            end
        end
        (Δy, Δu, nothing)
    end
end

function gather_indices(X::CuArray{T}) where T
    Y = gather_indices(Array(X))
    cuY = Dict{T,CuVector}(k => cu(Tuple.(v)) for (k, v) in Y)
    cuY
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

function numerical_cmp(X::CuArray{T}, Y::CuArray) where T
    Z = map((x,y) -> sign(x - y)^2, X, Y)
    Z = map(x -> (one(T) - x)^2, Z)
    Z
end
