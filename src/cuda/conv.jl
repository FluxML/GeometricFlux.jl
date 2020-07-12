# Avoid ambiguity
function update_batch_vertex(g::GATConv, M::AbstractMatrix, X::CuMatrix)
    M = convert(typeof(X), M)
    update_batch_vertex(g, M, X)
end

function update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix)
    if !g.concat
        N = size(M, 2)
        M = reshape(mean(reshape(M, :, g.heads, N), dims=2), :, N)
    end
    return M .+ g.bias
end
