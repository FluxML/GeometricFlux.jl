# Avoid ambiguity
function update_batch_vertex(g::GATConv, M::AbstractMatrix, X::CuMatrix, u)
    M = convert(typeof(X), M)
    update_batch_vertex(g, M, X, u)
end

function update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix, u)
    g.concat || (M = mean(M, dims=2))
    return M .+ g.bias
end
