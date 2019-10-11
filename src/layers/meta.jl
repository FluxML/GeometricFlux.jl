abstract type Meta end

adjlist(m::T) where {T<:Meta} = m.adjlist
update_edge(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_vertex(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_global(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_edges(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_vertices(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")

get_vertices(x) = haskey(x, :X) && (return x[:X])
get_edges(x) = haskey(x, :E) && (return x[:E])

function get_neighbors(d, i::Integer, ne)
    result = Dict{Symbol,AbstractArray}()
    if haskey(d, :X)
        result[:x_i] = view(d[:X], :, i)
        result[:x_j] = view(d[:X], :, ne)
    end
    if haskey(d, :E)
        result[:e_ij] = view(d[:E], :, i, ne)
    end
    result
end

function propagate(meta::T; kwargs...) where {T<:Meta}
    arg_names = keys(kwargs)
    gi = GraphInfo(adjlist(meta))

    newE = update_edge(meta; gi=gi, kwargs...)  # x[row], x[col], edge_attr, u, batch[row]

    if :naggr in arg_names
        Ē = aggregate_neighbors(meta, kwargs[:naggr]; E=newE, cluster=generate_cluster(newE, gi))
    end

    newV = update_vertex(meta; Ē=Ē, kwargs...)  # x, edge_index, edge_attr, u, batch

    if :eaggr in arg_names
        Ē = aggregate_edges(meta, kwargs[:eaggr]; kwargs...)
    end
    if :vaggr in arg_names
        V̄ = aggregate_vertices(meta, kwargs[:vaggr]; kwargs...)
    end

    new_u = update_global(meta; Ē=Ē, V̄=V̄, u=kwargs[:u])

    (newE, newV, new_u)
end
