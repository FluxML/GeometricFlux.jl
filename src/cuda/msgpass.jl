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
    E_ = Vector[]
    for (i, js) = enumerate(adj)
        for j = js
            k = edge_idx[(i,j)]
            m = message(mp, get_feature(X, i), get_feature(X, j), get_feature(E, k))
            push!(E_, m)
        end
    end
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
    X_ = Vector[]
    for i = 1:size(X,2)
        x = update(mp, get_feature(M, i), get_feature(X, i))
        push!(X_, x)
    end
    hcat(X_...)
end
