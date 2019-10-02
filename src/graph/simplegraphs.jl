using LightGraphs: AbstractSimpleGraph, nv, adjacency_matrix

function GCNConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2], N) : zeros(T, ch[2], N)
    adj = adjacency_matrix(g)
    GCNConv(init(ch[2], ch[1]), b, normalized_laplacian(adj+I, T), σ)
end


function ChebConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2], N) : zeros(T, ch[2], N)
    adj = adjacency_matrix(g)
    L̃ = T(2. / eigmax(Matrix(adj))) * normalized_laplacian(adj, T) - I
    ChebConv(init(ch[2], ch[1], k), b, L̃, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2], N) : zeros(T, ch[2], N)
    GraphConv(adjlist(g), init(ch[2], ch[1]), init(ch[2], ch[1]), b, aggr)
end


function GATConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat=true, negative_slope=0.2, init=glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2], N) : zeros(T, ch[2], N)
    GATConv(adjlist(g), init(ch[2], ch[1]), b, init(2 * ch[2]), negative_slope)
end


function GatedGraphConv(g::AbstractSimpleGraph, out_ch::Integer, num_layers::Integer;
                        aggr=:add, init=glorot_uniform)
    N = nv(g)
    w = init(out_ch, out_ch, num_layers)
    gru = GRUCell(out_ch, out_ch)
    GatedGraphConv(adjlist(g), w, gru, out_ch, num_layers, aggr)
end


function EdgeConv(g::AbstractSimpleGraph, nn; aggr::Symbol=:max)
    EdgeConv(adjlist(g), nn, aggr)
end
