## Linear algebra API for adjacency matrix

"""
    degrees(g[, T; dir=:out])

Degree of each vertex. Return a vector which contains the degree of each vertex in graph `g`.

# Arguments
- `g`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples
```jldoctest
julia> m = [0 1 1; 1 0 0; 1 0 0];

julia> GeometricFlux.degrees(m)
3-element Array{Int64,1}:
 2
 1
 1

```
"""
function degrees(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    if issymmetric(adj)
        d = vec(sum(adj, dims=1))
    else
        if dir == :out
            d = vec(sum(adj, dims=1))
        elseif dir == :in
            d = vec(sum(adj, dims=2))
        elseif dir == :both
            d = vec(sum(adj, dims=1)) + vec(sum(adj, dims=2))
        else
            throw(DomainError(dir, "invalid argument, only accept :in, :out and :both"))
        end
    end
    d
end

function degree_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    d = degrees(adj, T, dir=dir)
    return SparseMatrixCSC(T.(diagm(0=>d)))
end

function inv_sqrt_degree_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    d = degrees(adj, T, dir=dir).^(-0.5)
    return SparseMatrixCSC(T.(diagm(0=>d)))
end

function laplacian_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    degree_matrix(adj, T, dir=dir) - SparseMatrixCSC(T.(adj))
end

function normalized_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj); selfloop::Bool=false)
    selfloop && (adj += I)
    inv_sqrtD = inv_sqrt_degree_matrix(adj, T, dir=:both)
    I - inv_sqrtD * SparseMatrixCSC(T.(adj)) * inv_sqrtD
end

function neighbors(adj::AbstractMatrix, T::DataType=eltype(adj))
    n = size(adj,1)
    @assert n == size(adj,2) "adjacency matrix is not a square matrix."
    A = (adj .!= zero(T))
    if !issymmetric(adj)
        A = A .| A'
    end
    indecies = collect(1:n)
    ne = Vector{Int}[indecies[view(A, :, i)] for i = 1:n]
    return ne
end
