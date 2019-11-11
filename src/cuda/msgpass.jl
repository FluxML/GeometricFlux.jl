function _apply_msg!(m, Y::CuArray, V, edge_idx, adj; kwargs...)
    @inbounds for i = 2:V
        j = edge_idx[i]
        k = edge_idx[i+1]
        Y[:, j+1:k] = message(m; adjacent_vertices_data(kwargs, i, adj[i])...,
                                 incident_edges_data(kwargs, i, adj[i])...)
    end
    Y
end
