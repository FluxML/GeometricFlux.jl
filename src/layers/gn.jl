const AGGR2FUN = Dict(:add => sum, :sub => sum, :mul => prod, :div => prod,
                      :max => maximum, :min => minimum, :mean => mean)

abstract type GraphNet end

@inline update_edge(gn::T, e, vi, vj, u) where {T<:GraphNet} = e
@inline update_vertex(gn::T, ē, vi, u) where {T<:GraphNet} = vi
@inline update_global(gn::T, ē, v̄, u) where {T<:GraphNet} = u

@inline function update_batch_edge(gn::T, E, V, u, adj) where {T<:GraphNet}
    edge_idx = edge_index_table(adj)
    E_ = Vector[]
    for (i, js) = enumerate(adj)
        for j = js
            k = edge_idx[(i,j)]
            e = update_edge(gn, E[:,k], V[:,i], V[:,j], u)
            push!(E_, e)
        end
    end
    hcat(E_...)
end

@inline function update_batch_vertex(gn::T, Ē, V, u) where {T<:GraphNet}
    V_ = Vector[]
    for i = 1:size(V,2)
        v = update_vertex(gn, Ē[:,i], V[:,i], u)
        push!(V_, v)
    end
    hcat(V_...)
end

@inline function aggregate_neighbors(gn::T, aggr::Symbol, E, accu_edge, num_V, num_E) where {T<:GraphNet}
    @assert iszero(E) "edge feature must not be zero."
    @assert iszero(accu_edge) "accumulated edge must not be zero."
    cluster = generate_cluster(E, accu_edge, num_V, num_E)
    pool(aggr, cluster, E)
end

@inline function aggregate_neighbors(gn::T, aggr::Nothing, E, accu_edge, num_V, num_E) where {T<:GraphNet}
    @nospecialize E accu_edge num_V num_E
end

@inline function aggregate_edges(gn::T, aggr::Symbol, E) where {T<:GraphNet}
    @assert iszero(E) "edge feature must not be zero."
    u = vec(AGGR2FUN[aggr](E, dims=(2,3)))
    aggr == :sub && (u = -u)
    aggr == :div && (u = 1 ./ u)
    u
end

@inline function aggregate_edges(gn::T, aggr::Nothing, E) where {T<:GraphNet}
    @nospecialize E
end

@inline function aggregate_vertices(gn::T, aggr::Symbol, V) where {T<:GraphNet}
    @assert iszero(V) "node feature must not be zero."
    u = vec(AGGR2FUN[aggr](V, dims=2))
    aggr == :sub && (u = -u)
    aggr == :div && (u = 1 ./ u)
    u
end

@inline function aggregate_vertices(gn::T, aggr::Nothing, V) where {T<:GraphNet}
    @nospecialize V
end

function propagate(gn::T, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing) where {T<:GraphNet}
    _propagate(fg, naggr, eaggr, vaggr)
end

function _propagate(fg::FeaturedGraph, naggr, eaggr, vaggr)
    num_V = nv(fg)
    accu_edge = accumulated_edges(adj, num_V)
    num_E = accu_edge[end]
    E = edge_feature(fg)
    V = node_feature(fg)
    u = global_feature(fg)

    E = update_batch_edge(gn, E, V, u, adj)

    Ē = aggregate_neighbors(gn, naggr, E, accu_edge, num_V, num_E)

    V = update_batch_vertex(gn, Ē, V, u)

    ē = aggregate_edges(gn, eaggr, E)

    v̄ = aggregate_vertices(gn, vaggr, V)

    u = update_global(gn, ē, v̄, u)

    E, V, u
end

Zygote.@nograd function generate_cluster(M::AbstractArray{T,N}, accu_edge, V, E) where {T,N}
    cluster = similar(M, Int, E)
    @inbounds for i = 1:V
        j = accu_edge[i]
        k = accu_edge[i+1]
        cluster[j+1:k] .= i
    end
    cluster
end
