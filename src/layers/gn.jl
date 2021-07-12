_view(::Nothing, i) = nothing
_view(A::Fill{T,2,Axes}, i) where {T,Axes} = view(A, :, 1)
_view(A::AbstractMatrix, idx) = view(A, :, idx)

aggregate(aggr::typeof(+), X) = vec(sum(X, dims=2))
aggregate(aggr::typeof(-), X) = -vec(sum(X, dims=2))
aggregate(aggr::typeof(*), X) = vec(prod(X, dims=2))
aggregate(aggr::typeof(/), X) = 1 ./ vec(prod(X, dims=2))
aggregate(aggr::typeof(max), X) = vec(maximum(X, dims=2))
aggregate(aggr::typeof(min), X) = vec(minimum(X, dims=2))
aggregate(aggr::typeof(mean), X) = vec(aggr(X, dims=2))

abstract type GraphNet end

@inline update_edge(gn::GraphNet, e, vi, vj, u) = e
@inline update_vertex(gn::GraphNet, ē, vi, u) = vi
@inline update_global(gn::GraphNet, ē, v̄, u) = u

@inline function update_batch_edge(gn::GraphNet, adj, E, V, u)
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    mapreduce(i -> apply_batch_message(gn, i, adj[i], edge_idx, E, V, u), hcat, 1:n)
end

@inline apply_batch_message(gn::GraphNet, i, js, edge_idx, E, V, u) =
    mapreduce(j -> update_edge(gn, _view(E, edge_idx[(i,j)]), _view(V, i), _view(V, j), u), hcat, js)

@inline update_batch_vertex(gn::GraphNet, Ē, V, u) =
    mapreduce(i -> update_vertex(gn, _view(Ē, i), _view(V, i), u), hcat, 1:size(V,2))

@inline function aggregate_neighbors(gn::GraphNet, aggr, E, accu_edge)
    @assert !iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(E, accu_edge)
    NNlib.scatter(aggr, E, cluster)
end

@inline function aggregate_neighbors(gn::GraphNet, aggr::Nothing, E, accu_edge)
    @nospecialize E accu_edge num_V num_E
end

@inline aggregate_edges(gn::GraphNet, aggr, E) = aggregate(aggr, E)

@inline function aggregate_edges(gn::GraphNet, aggr::Nothing, E)
    @nospecialize E
end

@inline aggregate_vertices(gn::GraphNet, aggr, V) = aggregate(aggr, V)

@inline function aggregate_vertices(gn::GraphNet, aggr::Nothing, V)
    @nospecialize V
end

function propagate(gn::GraphNet, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)
    E, V, u = propagate(gn, adjacency_list(fg), fg.ef, fg.nf, fg.gf, naggr, eaggr, vaggr)
    FeaturedGraph(graph(fg), nf=V, ef=E, gf=u)
end

function propagate(gn::GraphNet, adj::AbstractVector{S}, E::R, V::Q, u::P,
                   naggr=nothing, eaggr=nothing, vaggr=nothing) where {S<:AbstractVector,R,Q,P}
    E = update_batch_edge(gn, adj, E, V, u)
    Ē = aggregate_neighbors(gn, naggr, E, accumulated_edges(adj))
    V = update_batch_vertex(gn, Ē, V, u)
    ē = aggregate_edges(gn, eaggr, E)
    v̄ = aggregate_vertices(gn, vaggr, V)
    u = update_global(gn, ē, v̄, u)
    E, V, u
end
