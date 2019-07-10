using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, nv

function GCNConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(N, ch[2])) : zeros(T, N, ch[2])
    adj = adjacency_matrix(g)
    GCNConv(param(init(ch[1], ch[2])), b, normalized_laplacian(adj+I, T), σ)
end


function ChebConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(N, ch[2])) : zeros(T, N, ch[2])
    adj = adjacency_matrix(g)
    L̃ = T(2. / eigmax(Matrix(adj))) * normalized_laplacian(adj, T) - I
    ChebConv(param(init(k, ch[1], ch[2])), b, L̃, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, aggr=+;
                   init = glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(N, ch[2])) : zeros(T, N, ch[2])
    GraphConv(adjlist(g), param(init(ch[1], ch[2])), b, aggr)
end


function GATConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat=true, negative_slope=0.2, init=glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(N, ch[2])) : zeros(T, N, ch[2])
    GATConv(adjlist(g), param(init(ch[1], ch[2])), b, param(init(2 * ch[2])), negative_slope)
end


function EdgeConv(g::AbstractSimpleWeightedGraph, nn; aggr::Symbol=:max)
    aggr in keys(aggr_func) || throw(DomainError(aggr, "not supported aggregation function."))
    EdgeConv(adjlist(g), nn, aggr_func[aggr])
end
