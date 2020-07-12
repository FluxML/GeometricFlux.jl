using LightGraphs: AbstractSimpleGraph, nv, adjacency_matrix, inneighbors, outneighbors,
                   all_neighbors

function adjacency_list(g::AbstractSimpleGraph)
    N = nv(g)
    Vector{Int}[outneighbors(g, i) for i = 1:N]
end

## Convolution layers accepting AbstractSimpleGraph

function GCNConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    w = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(g)
    GCNConv(w, b, σ, fg)
end


function ChebConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    fg = FeaturedGraph(g)
    ChebConv(T.(init(ch[2], ch[1], k)), b, fg, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    w1 = T.(init(ch[2], ch[1]))
    w2 = T.(init(ch[2], ch[1]))
    b = bias ? T.(init(ch[2])) : zeros(T, ch[2])
    GraphConv(FeaturedGraph(g), w1, w2, b, aggr)
end


function GATConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat::Bool=true, negative_slope=0.2, init=glorot_uniform,
                 T::DataType=Float32, bias::Bool=true)
    w = T.(init(ch[2]*heads, ch[1]))
    b = concat ? (bias ? T.(init(ch[2]*heads)) : zeros(T, ch[2]*heads)) :
        (bias ? T.(init(ch[2])) : zeros(T, ch[2]))
    a = T.(init(2*ch[2], heads, 1))
    GATConv(FeaturedGraph(g), w, b, a, negative_slope, ch, heads, concat)
end


function GatedGraphConv(g::AbstractSimpleGraph, out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform, T::DataType=Float32)
    w = T.(init(out_ch, out_ch, num_layers))
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(FeaturedGraph(g), w, gru, out_ch, num_layers, aggr)
end


EdgeConv(g::AbstractSimpleGraph, nn; aggr::Symbol=:max) = EdgeConv(FeaturedGraph(g), nn, aggr)
