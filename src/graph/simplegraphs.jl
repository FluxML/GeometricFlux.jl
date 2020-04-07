using LightGraphs: AbstractSimpleGraph, nv, adjacency_matrix


## Linear algebra API for AbstractSimpleGraph

function degrees(sg::AbstractSimpleGraph, T::DataType=eltype(sg); dir::Symbol=:out)
    degrees(adjacency_matrix(sg, T; dir=dir), T; dir=dir)
end

function degree_matrix(sg::AbstractSimpleGraph, T::DataType=eltype(sg); dir::Symbol=:out)
    degree_matrix(adjacency_matrix(sg, T; dir=dir), T; dir=dir)
end

function inv_sqrt_degree_matrix(sg::AbstractSimpleGraph, T::DataType=eltype(sg); dir::Symbol=:out)
    inv_sqrt_degree_matrix(adjacency_matrix(sg, T; dir=dir), T; dir=dir)
end

function laplacian_matrix(sg::AbstractSimpleGraph, T::DataType=eltype(sg); dir::Symbol=:out)
    laplacian_matrix(adjacency_matrix(sg, T; dir=dir), T; dir=dir)
end

function normalized_laplacian(sg::AbstractSimpleGraph, T::DataType=eltype(sg); selfloop::Bool=false)
    adj = adjacency_matrix(sg, T)
    selfloop && (adj += I)
    normalized_laplacian(adj, T)
end


## Convolution layers accepting AbstractSimpleGraph

function GCNConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    fg = FeaturedGraph(Ref(g), Ref(nothing))
    GCNConv(init(ch[2], ch[1]), b, σ, fg)
end


function ChebConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    adj = adjacency_matrix(g)
    L̃ = T(2. / eigmax(Matrix(adj))) * normalized_laplacian(adj, T) - I
    ChebConv(init(ch[2], ch[1], k), b, L̃, k, ch[1], ch[2])
end


function GraphConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}, aggr=:add;
                   init = glorot_uniform, bias::Bool=true)
    N = nv(g)
    b = bias ? init(ch[2]) : zeros(T, ch[2])
    GraphConv(adjlist(g), init(ch[2], ch[1]), init(ch[2], ch[1]), b, aggr)
end


function GATConv(g::AbstractSimpleGraph, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat::Bool=true, negative_slope=0.2, init=glorot_uniform,
                 bias::Bool=true)
    N = nv(g)
    w = init(ch[2]*heads, ch[1])
    b = bias ? init(ch[2]*heads) : zeros(T, ch[2]*heads)
    a = init(2*ch[2], heads, 1)
    GATConv(adjlist(g), w, b, a, negative_slope, ch, heads, concat)
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
