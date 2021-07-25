@inline function update_batch_edge(mp::MessagePassing, adj, E::Fill{S,2,Axes}, X::CuMatrix, u) where {S,Axes}
    E = fill(E.value, E.axes)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::MessagePassing, adj, E::CuMatrix, X::Fill{S,2,Axes}, u) where {S,Axes}
    X = fill(X.value, X.axes)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::MessagePassing, adj, E::AbstractMatrix, X::CuMatrix, u)
    E = convert(typeof(X), E)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::MessagePassing, adj, E::CuMatrix, X::AbstractMatrix, u)
    X = convert(typeof(E), X)
    update_batch_edge(mp, adj, E, X, u)
end

@inline function update_batch_edge(mp::MessagePassing, adj, E::CuMatrix, X::CuMatrix, u)
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    mapreduce(i -> apply_batch_message(mp, i, adj[i], edge_idx, E, X, u), hcat, 1:n)
end

@inline apply_batch_message(mp::MessagePassing, i, js, edge_idx, E::CuMatrix, X::CuMatrix, u) =
    mapreduce(j -> message(mp, _view(X, i), _view(X, j), _view(E, edge_idx[(i,j)])), hcat, js)

@inline function update_batch_vertex(mp::MessagePassing, M::AbstractMatrix, X::CuMatrix, u)
    M = convert(typeof(X), M)
    update_batch_vertex(mp, M, X, u)
end

@inline function update_batch_vertex(mp::MessagePassing, M::CuMatrix, X::AbstractMatrix, u)
    X = convert(typeof(M), X)
    update_batch_vertex(mp, M, X, u)
end

@inline update_batch_vertex(mp::MessagePassing, M::CuMatrix, X::CuMatrix, u) =
    mapreduce(i -> update(mp, _view(M, i), _view(X, i)), hcat, 1:size(X,2))
