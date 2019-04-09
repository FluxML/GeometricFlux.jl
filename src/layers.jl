GraphConv(ch::Pair{<:Integer,<:Integer}, aggr::Symbol, bias::Bool)

MessagePassing(aggr::Symbol)

GCNConv(ch::Pair{<:Integer,<:Integer})

ChebConv(k::Integer, ch::Pair{<:Integer,<:Integer})

GatedGraphConv(out_ch::Integer, len::Integer, aggr::Symbol, bias::Bool)

EdgeConv(nn, aggr::Symbol)

ODE()

Meta(edge_model, node_model, global_model)