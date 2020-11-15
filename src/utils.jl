## Top-k pooling

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
        FeaturedGraph(graph(fg), nf_func(node_feature(fg)), ef_func(edge_feature(fg)),
                      gf_func(global_feature(fg)))
    end
end

function add_self_loop!(adj::AbstractVector{T}, n::Int=length(adj)) where {T<:AbstractVector}
    for i = 1:n
        i in adj[i] || push!(adj[i], i)
    end
    adj
end