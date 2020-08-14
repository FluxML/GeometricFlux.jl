using SimpleWeightedGraphs: AbstractSimpleWeightedGraph

## Convolution layers accepting AbstractSimpleWeightedGraph

function GCNConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    w = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(g)
    GCNConv(w, b, σ, fg)
end


function ChebConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    w = T.(init(ch[2], ch[1], k))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(g)
    ChebConv(w, b, fg, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    w1 = T.(init(ch[2], ch[1]))
    w2 = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    GraphConv(FeaturedGraph(g), w1, w2, b, aggr)
end


function GATConv(g::AbstractSimpleWeightedGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat::Bool=true, negative_slope=0.2, init=glorot_uniform,
                 T::DataType=Float32, bias::Bool=true)
    w = T.(init(ch[2]*heads, ch[1]))
    b = bias ? T.(init(ch[2]*heads)) : zeros(T, ch[2]*heads)
    a = T.(init(2*ch[2], heads, 1))
    GATConv(FeaturedGraph(g), w, b, a, negative_slope, ch, heads, concat)
end


function GatedGraphConv(g::AbstractSimpleWeightedGraph, out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform, T::DataType=Float32)
    w = T.(init(out_ch, out_ch, num_layers))
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(FeaturedGraph(g), w, gru, out_ch, num_layers, aggr)
end


EdgeConv(g::AbstractSimpleWeightedGraph, nn; aggr::Symbol=:max) = EdgeConv(FeaturedGraph(g), nn, aggr)
