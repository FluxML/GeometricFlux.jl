function GCNConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform)
    GCNConv(param(init(ch[2], ch[1])), param(laplacian_matrix(adj+I)), σ)
end


function ChebConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform)
    return ChebConv(, zeros(size(X, 1), size(X, 2), k))
end


function GraphConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, aggr::Symbol;
                   init = glorot_uniform)
    return GraphConv(, init(ch[2], ch[1]), aggr)
end


function GATConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer};
                 heads=1, concat=True, negative_slope=0.2, dropout=0)
    return GATConv()
end
