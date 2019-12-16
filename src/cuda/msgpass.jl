function _apply_msg!(m, Y::CuArray, V, edge_idx, adj; kwargs...)
    @inbounds for i = 2:V
        j = edge_idx[i]
        k = edge_idx[i+1]
        M = message(m; adjacent_vertices_data(kwargs, i, adj[i])...,
                       incident_edges_data(kwargs, i, adj[i])...)
        assign!(Y, M; last_dim=j+1:k)
    end
    Y
end
