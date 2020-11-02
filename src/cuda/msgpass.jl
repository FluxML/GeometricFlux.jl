@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::CuMatrix) where {T<:MessagePassing}
    E = convert(typeof(X), Matrix(E))
    update_batch_edge(mp, adj, E, X)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    X = convert(typeof(E), X)
    update_batch_edge(mp, adj, E, X)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::CuMatrix) where {T<:MessagePassing}
    n = size(adj, 1)
    edge_idx = edge_index_table(adj)
    hcat([apply_batch_message(mp, i, adj[i], edge_idx, E, X) for i in 1:n]...)
end

@inline function apply_batch_message(mp::T, i, js, edge_idx, E::CuMatrix, X::CuMatrix) where {T<:MessagePassing}
    hcat([message(mp, get_feature(X, i), get_feature(X, j), get_feature(E, edge_idx[(i,j)])) for j = js]...)
end

@inline function update_batch_vertex(mp::T, M::AbstractMatrix, X::CuMatrix) where {T<:MessagePassing}
    M = convert(typeof(X), M)
    update_batch_vertex(mp, M, X)
end

@inline function update_batch_vertex(mp::T, M::CuMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    X = convert(typeof(M), X)
    update_batch_vertex(mp, M, X)
end

@inline function update_batch_vertex(mp::T, M::CuMatrix, X::CuMatrix) where {T<:MessagePassing}
    X_ = [update(mp, get_feature(M, i), get_feature(X, i)) for i = 1:size(X,2)]
    hcat(X_...)
end
