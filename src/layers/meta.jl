abstract type Meta end

adjlist(m::T) where {T<:Meta} = m.adjlist
update_edge(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_vertex(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_global(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_edges(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_vertices(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")

# function propagate(meta::T; kwargs...) where {T<:Meta}
#     row, col = edge_index
#     for k ∈ edges()
#         e' = update_edge(meta, x[row], x[col], edge_attr, u, batch[row])
#     end
#     for i ∈ vertices()
#         E' = get_neighbors(i)
#         aggregate_neighbors(meta, E', naggr)
#         update_vertex(meta, i, x, edge_index, edge_attr, u, batch)
#     end
#     get_vertices()
#     get_edges()
#     aggregate_edges(meta, eaggr)
#     aggregate_vertices(meta, vaggr)
#     update_global(meta, x, edge_index, edge_attr, u, batch)
# end
