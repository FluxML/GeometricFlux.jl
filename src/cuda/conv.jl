(g::GCNConv)(L̃::AbstractMatrix, X::CuArray) = g(cu(L̃), X)

(g::GCNConv)(L̃::CuArray, X::CuArray) = g.σ.(g.weight * X * L̃ .+ g.bias)

(c::ChebConv)(L̃::AbstractMatrix, X::CuArray) = c(cu(L̃), X)

function (c::ChebConv)(L̃::CuArray, X::CuArray)
    @assert size(X, 1) == c.in_channel "Input feature size must match input channel size."
    @assert size(X, 2) == size(L̃, 1) "Input vertex number must match Laplacian matrix size."

    Z_prev = X
    Z = X * L̃
    Y = view(c.weight,:,:,1) * Z_prev
    Y += view(c.weight,:,:,2) * Z
    for k = 3:c.k
        Z, Z_prev = 2*Z*L̃ - Z_prev, Z
        Y += view(c.weight,:,:,k) * Z
    end
    return Y .+ c.bias
end


# Avoid ambiguity
update_batch_edge(g::GATConv, adj, E::Fill{S,2,Axes}, X::CuMatrix, u) where {S,Axes} = update_batch_edge(g, adj, X)

update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix, u) = update_batch_vertex(g, M)
