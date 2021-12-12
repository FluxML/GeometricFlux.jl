# Avoid ambiguity
update_batch_edge(g::GATConv, adj, E::Fill{S,2,Axes}, X::CuMatrix, u) where {S,Axes} = update_batch_edge(g, adj, X)

update_batch_vertex(g::GATConv, M::CuMatrix, X::CuMatrix, u) = update_batch_vertex(g, M)
