abstract type Meta end

adjlist(m::T) where {T<:Meta} = m.adjlist
update_edge(m::T; kwargs...) where {T<:Meta} = ifelse(haskey(kwargs, :E), kwargs[:E], nothing)
update_vertex(m::T; kwargs...) where {T<:Meta} = ifelse(haskey(kwargs, :X), kwargs[:X], nothing)
update_global(m::T; kwargs...) where {T<:Meta} = ifelse(haskey(kwargs, :u), kwargs[:u], nothing)

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

function propagate(meta::T; adjl=adjlist(meta), kwargs...) where {T<:Meta}
    gi = GraphInfo(adjl)

    newE = update_edge(meta; gi=gi, kwargs...)

    if haskey(kwargs, :naggr)
        Ē = aggregate_neighbors(meta, kwargs[:naggr]; E=newE, cluster=generate_cluster(newE, gi))
        kwargs = (kwargs..., Ē=Ē)
    end

    newV = update_vertex(meta; kwargs...)

    if haskey(kwargs, :eaggr)
        Ē = aggregate_edges(meta, kwargs[:eaggr]; kwargs...)
        kwargs = (kwargs..., Ē=Ē)
    end
    if haskey(kwargs, :vaggr)
        V̄ = aggregate_vertices(meta, kwargs[:vaggr]; kwargs...)
        kwargs = (kwargs..., V̄=V̄)
    end

    new_u = update_global(meta; kwargs...)

    (newE, newV, new_u)
end

function generate_cluster(M::AbstractMatrix, gi::GraphInfo)
    cluster = similar(M, Int, gi.E)
    @inbounds for i = 1:gi.V
        j = gi.edge_idx[i]
        k = gi.edge_idx[i+1]
        cluster[j+1:k] .= i
    end
    cluster
end
