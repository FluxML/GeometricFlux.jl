# Avoid ambiguity
function update_batch_vertex(g::GATConv, M::AbstractMatrix, X::CuMatrix, u)
    M = convert(typeof(X), M)
    update_batch_vertex(g, M, X, u)
end

function update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix, u)
    M = M .+ g.bias
    if !g.concat
        N = size(M, 2)
        M = reshape(mean(reshape(M, :, g.heads, N), dims=2), :, N)
    end
    return M
end
