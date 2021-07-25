## Top-k pooling
import Base: +, -, *, /, broadcasted

function topk_index(y::AbstractVector, k::Integer)
    v = nlargest(k, y)
    return collect(1:length(y))[y .>= v[end]]
end

topk_index(y::Adjoint, k::Integer) = topk_index(y', k)



## Get feature with defaults

get_feature(::Nothing, i) = nothing
get_feature(A::Fill{T,2,Axes}, i::Integer) where {T,Axes} = view(A, :, 1)
get_feature(A::AbstractMatrix, i::Integer) = view(A, :, i)

"""
    bypass_graph(nf_func, ef_func, gf_func)

Bypassing graph in FeaturedGraph and let other layer process (node, edge and global)features only.
"""
function bypass_graph(nf_func=identity, ef_func=identity, gf_func=identity)
    return function (fg::FeaturedGraph)
        FeaturedGraph(graph(fg),
                      nf=nf_func(node_feature(fg)),
                      ef=ef_func(edge_feature(fg)),
                      gf=gf_func(global_feature(fg)))
    end
end

function add_self_loop!(adj::AbstractVector{T}, n::Int=length(adj)) where {T<:AbstractVector}
    for i = 1:n
        i in adj[i] || push!(adj[i], i)
    end
    adj
end

# Used for untrainable parameters, like epsilon in GINConv when set to false.
# NOTE: Only works for scalars untrainable params.
mutable struct Untrainable{T <: Number}
    value::T
end

+(u::Untrainable, z::Number) = u.value+z
@adjoint +(u::Untrainable, z::Number) = u+z, _->(nothing, 1.0)
+(z::Number,u::Untrainable) = u.value+z
@adjoint +(z::Number, u::Untrainable) = z+u, _->(1.0, nothing)
+(u1::Untrainable,u2::Untrainable) = u1.value + u2.value
@adjoint +(u1::Untrainable, u2::Untrainable) = u1+u2, _->(nothing,nothing)

-(u::Untrainable, z::Number) = u.value-z
@adjoint -(u::Untrainable, z::Number) = u-z, _->(nothing, -1.0)
-(z::Number,u::Untrainable) = z-u.value
@adjoint -(z::Number, u::Untrainable) = z-u, _->(1.0, nothing)
-(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint -(u1::Untrainable, u2::Untrainable) = u1-u2, _->(nothing,nothing)

*(u::Untrainable, z::Number) = u.value*z
@adjoint *(u::Untrainable, z::Number) = u*z, _->(nothing, u.value)
*(z::Number,u::Untrainable) = z*u.value
@adjoint *(z::Number, u::Untrainable) = z*u, _->(u.value, nothing)
*(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint *(u1::Untrainable, u2::Untrainable) = u1*u2, _->(nothing,nothing)

/(u::Untrainable, z::Number) = u.value/z
@adjoint /(u::Untrainable, z::Number) = u/z, _->(nothing, -u.value/(z^2))
/(z::Number,u::Untrainable) = z/u.value
@adjoint /(z::Number, u::Untrainable) = z/u, _->(1/u.value, nothing)
/(u1::Untrainable,u2::Untrainable) = u1.value - u2.value
@adjoint /(u1::Untrainable, u2::Untrainable) = u1/u2, _->(nothing,nothing)

broadcasted(::typeof(+), a::Untrainable, b::AbstractArray) = 
    broadcasted(+, a.value, b)
@adjoint broadcasted(::typeof(+), a::Untrainable, b::AbstractArray) = 
    broadcasted(+, a, b), _->(nothing, nothing, ones(size(b)))

broadcasted(::typeof(+), a::AbstractArray, b::Untrainable) = 
    broadcasted(+, a, b.value)
@adjoint broadcasted(::typeof(+), a::AbstractArray, b::Untrainable) = 
    broadcasted(+, a, b), _->(nothing, ones(size(b)), nothing)

broadcasted(::typeof(-), a::Untrainable, b::AbstractArray) = 
    broadcasted(-, a.value, b)
@adjoint broadcasted(::typeof(-), a::Untrainable, b::AbstractArray) = 
    broadcasted(-, a, b), _->(nothing, nothing, -ones(size(b)))

broadcasted(::typeof(-), a::AbstractArray, b::Untrainable) = 
    broadcasted(-, a, b.value)
@adjoint broadcasted(::typeof(-), a::AbstractArray, b::Untrainable) = 
    broadcasted(-, a, b), _->(nothing, ones(size(a)), nothing)

*(a::Untrainable, b::AbstractArray) = a.value * b
@adjoint *(a::Untrainable, b::AbstractArray) = 
    a * b, _->(nothing, nothing, a.value * ones(size(b)))
*(a::AbstractArray, b::Untrainable) = a * b.value
@adjoint broadcasted(::typeof(*), a::AbstractArray, b::Untrainable) = 
    a * b, _->(nothing, b.value * ones(size(a)), nothing)

broadcasted(::typeof(/), a::Untrainable, b::AbstractArray) = 
    broadcasted(/, a.value, b)
@adjoint broadcasted(::typeof(/), a::Untrainable, b::AbstractArray) = 
    broadcasted(/, a, b), _->(nothing, nothing, a.value ./ (b.^2))
    
/(a::AbstractArray, b::Untrainable) = /(a, b.value)
@adjoint /(a::AbstractArray, b::Untrainable) = 
    a/b, _->(nothing, (1/b.value) * ones(size(a)), nothing)

