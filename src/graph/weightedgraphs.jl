using SimpleWeightedGraphs: AbstractSimpleWeightedGraph, nv

function GCNConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(ch[2], N)) : zeros(T, ch[2], N)
    adj = adjacency_matrix(g)
    GCNConv(param(init(ch[2], ch[1])), b, normalized_laplacian(adj+I, T), σ)
end


function ChebConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(ch[2], N)) : zeros(T, ch[2], N)
    adj = adjacency_matrix(g)
    L̃ = T(2. / eigmax(Matrix(adj))) * normalized_laplacian(adj, T) - I
    ChebConv(param(init(ch[2], ch[1], k)), b, L̃, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(ch[2], N)) : zeros(T, ch[2], N)
    GraphConv(adjlist(g), param(init(ch[2], ch[1])), param(init(ch[2], ch[1])), b, aggr)
end


function GATConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat=true, negative_slope=0.2, init=glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? param(init(ch[2], N)) : zeros(T, ch[2], N)
    GATConv(adjlist(g), param(init(ch[2], ch[1])), b, param(init(2 * ch[2])), negative_slope)
end


function GatedGraphConv(g::AbstractSimpleWeightedGraph, out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform)
    N = nv(g)
    w = param(init(out_ch, out_ch, num_layers))
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(adjlist(g), w, gru, out_ch, num_layers, aggr)
end


function EdgeConv(g::AbstractSimpleWeightedGraph, nn; aggr::Symbol=:max)
    EdgeConv(adjlist(g), nn, aggr)
end
