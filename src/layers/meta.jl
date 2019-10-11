abstract type Meta end

adjlist(m::T) where {T<:Meta} = m.adjlist
update_edge(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_vertex(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)
update_global(m::T; kwargs...) where {T<:Meta} = identity(; kwargs...)

aggregate_neighbors(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_edges(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")
aggregate_vertices(m::T, aggr::Symbol; kwargs...) where {T<:Meta} = error("not implement")

all_vertices_data(x) = ifelse(haskey(x, :X), (X=x[:X],), NamedTuple())
all_edges_data(x) = ifelse(haskey(x, :E), (E=x[:E],), NamedTuple())

function adjacent_vertices_data(x, i::Integer, ne)
    if haskey(x, :X)
        return (x_i=view(x[:X],:,i), x_j=view(x[:X],:,ne))
    else
        return NamedTuple()
    end
end

function incident_edges_data(x, i::Integer, ne)
    if haskey(x, :E)
        return (e_ij=view(x[:E],:,i,ne), )
    else
        return NamedTuple()
    end
end

function propagate(meta::T; kwargs...) where {T<:Meta}
    gi = GraphInfo(adjlist(meta))

    newE = update_edge(meta; gi=gi, kwargs...)  # x[row], x[col], edge_attr, u

    if haskey(kwargs, :naggr)
        Ē = aggregate_neighbors(meta, kwargs[:naggr]; E=newE, cluster=generate_cluster(newE, gi))
    end

    newV = update_vertex(meta; Ē=Ē, kwargs...)  # x, edge_index, edge_attr, u

    if haskey(kwargs, :eaggr)
        Ē = aggregate_edges(meta, kwargs[:eaggr]; kwargs...)
    end
    if haskey(kwargs, :vaggr)
        V̄ = aggregate_vertices(meta, kwargs[:vaggr]; kwargs...)
    end

    new_u = update_global(meta; Ē=Ē, V̄=V̄, u=kwargs[:u])

    (newE, newV, new_u)
end
