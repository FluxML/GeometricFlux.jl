# Avoid ambiguity
function update_batch_vertex(g::GATConv, M::AbstractMatrix, X::CuMatrix)
    M = convert(typeof(X), M)
    update_batch_vertex(g, M, X)
end

function update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix)
    g.concat || (M = mean(M, dims=2))
    return M .+ g.bias
end
