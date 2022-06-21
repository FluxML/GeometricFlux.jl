import Base: *, inv

struct R{N} end

Base.ndims(::R{N}) where {N} = N

Base.identity(::R{N}, ::Type{T}=Float32) where {N,T<:Number} = zeros(T, N)


struct H{N} end

Base.ndims(::H{N}) where {N} = N

Base.identity(::H{N}, ::Type{T}=Float32) where {N,T<:Number} = ones(T, N)

(*)(h1, h2) = h1 * h2

inv(h) = 1. / h

Base.log(h) = log.(h)

Base.exp(c) = exp.(c)

"""
The logarithmic distance ||log(inv(h1).h2)||

"""
dist(h1, h2) = log(inv(h1) * h2)
