@inline function update_batch_edge(mp::T, adj, E::AbstractMatrix, X::CuMatrix) where {T<:MessagePassing}
    E = convert(typeof(X), E)
    update_batch_edge(mp, adj, E, X)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::AbstractMatrix) where {T<:MessagePassing}
    X = convert(typeof(E), X)
    update_batch_edge(mp, adj, E, X)
end

@inline function update_batch_edge(mp::T, adj, E::CuMatrix, X::CuMatrix) where {T<:MessagePassing}
    edge_idx = edge_index_table(adj)
    edges = edge_list(adj)
    ijk = map(x -> (x[1], x[2], edge_idx[(x[1],x[2])]), edges)
    E_ = map(x -> message(mp, get_feature(X, x[1]), get_feature(X, x[2]), get_feature(E, x[3])), ijk)
    hcat(E_...)
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
    X_ = map(i -> update(mp, get_feature(M, i), get_feature(X, i)), 1:size(X,2))
    hcat(X_...)
end
