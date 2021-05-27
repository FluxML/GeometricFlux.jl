@inline function update_batch_edge(mp::T, adj, E::Fill{S,2,Axes}, X::CuMatrix, u) where {T<:MessagePassing,S,Axes}
    E = fill(E.value, E.axes)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::Fill{S,2,Axes}, u) where {T<:MessagePassing,S,Axes}
    X = fill(X.value, X.axes)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::CuMatrix, u) where {T<:MessagePassing}
    E = convert(typeof(X), E)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::AbstractMatrix, u) where {T<:MessagePassing}
    X = convert(typeof(E), X)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::CuMatrix, u) where {T<:MessagePassing}
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    mapreduce(i -> apply_batch_message(mp, i, adj[i], edge_idx, E, X, u), hcat, 1:n)
end

@inline apply_batch_message(mp::T, i, js, edge_idx, E::CuMatrix, X::CuMatrix, u) where {T<:MessagePassing} =
    mapreduce(j -> message(mp, _view(X, i), _view(X, j), _view(E, edge_idx[(i,j)])), hcat, js)

@inline function update_batch_vertex(mp::T, M::AbstractMatrix, X::CuMatrix, u) where {T<:MessagePassing}
    M = convert(typeof(X), M)
    update_batch_vertex(mp, M, X, u)
end

@inline function update_batch_vertex(mp::T, M::CuMatrix, X::AbstractMatrix, u) where {T<:MessagePassing}
    X = convert(typeof(M), X)
    update_batch_vertex(mp, M, X, u)
end

@inline update_batch_vertex(mp::T, M::CuMatrix, X::CuMatrix, u) where {T<:MessagePassing} =
    mapreduce(i -> update(mp, _view(M, i), _view(X, i)), hcat, 1:size(X,2))
