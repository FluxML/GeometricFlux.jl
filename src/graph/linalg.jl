using LightGraphs: AbstractGraph

## Linear algebra API for AbstractGraph

function degrees(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = adjacency_matrix(g, T; dir=dir)
    degrees(adj, T; dir=dir)
end

function degree_matrix(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = adjacency_matrix(g, T; dir=dir)
    degree_matrix(adj, T; dir=dir)
end

function inv_sqrt_degree_matrix(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = adjacency_matrix(g, T; dir=dir)
    inv_sqrt_degree_matrix(adj, T; dir=dir)
end

function laplacian_matrix(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = adjacency_matrix(g, T; dir=dir)
    laplacian_matrix(adj, T; dir=dir)
end

function normalized_laplacian(g::AbstractGraph, T::DataType=eltype(g); selfloop::Bool=false)
    adj = adjacency_matrix(g, T)
    selfloop && (adj += I)
    normalized_laplacian(adj, T)
end

function scaled_laplacian(g::AbstractGraph, T::DataType=eltype(g))
    adj = adjacency_matrix(g, T)
    scaled_laplacian(adj, T)
end
