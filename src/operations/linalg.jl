using LinearAlgebra

## Linear algebra API for adjacency matrix

Zygote.@nograd issymmetric

function adjacency_matrix(adj::AbstractMatrix, T::DataType=eltype(adj))
    m, n = size(adj)
    (m == n) || throw(DimensionMismatch("adjacency matrix is not a square matrix: ($m, $n)"))
    T.(adj)
end

"""
    degrees(g[, T; dir=:out])

Degree of each vertex. Return a vector which contains the degree of each vertex in graph `g`.

# Arguments
- `g`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples
```jldoctest
julia> using GeometricFlux

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

"""
    degree_matrix(g[, T; dir=:out])

Degree matrix of graph `g`. Return a matrix which contains degrees of each vertex in its diagonal.
The values other than diagonal are zeros.

# Arguments
- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples
```jldoctest
julia> using GeometricFlux

julia> m = [0 1 1; 1 0 0; 1 0 0];

julia> GeometricFlux.degree_matrix(m)
3×3 SparseArrays.SparseMatrixCSC{Int64,Int64} with 3 stored entries:
  [1, 1]  =  2
  [2, 2]  =  1
  [3, 3]  =  1
```
"""
function degree_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    d = degrees(adj, T, dir=dir)
    return SparseMatrixCSC(T.(diagm(0=>d)))
end

"""
    inv_sqrt_degree_matrix(g[, T; dir=:out])

Inverse squared degree matrix of graph `g`. Return a matrix which contains inverse squared degrees of each vertex in its diagonal.
The values other than diagonal are zeros.

# Arguments
- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
function inv_sqrt_degree_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    d  = inv.(sqrt.(degrees(adj, T, dir=dir)))
    return Diagonal(d)
end

"""
    laplacian_matrix(g[, T; dir=:out])

Laplacian matrix of graph `g`.

# Arguments
- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
function laplacian_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    degree_matrix(adj, T, dir=dir) - SparseMatrixCSC(T.(adj))
end

"""
    normalized_laplacian(g[, T; selfloop=false])

Normalized Laplacian matrix of graph `g`.

# Arguments
- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
"""
function normalized_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj); selfloop::Bool=false)
    selfloop && (adj += I)
    _normalized_laplacian(adj, T)
end

# nograd can only used without keyword arguments
Zygote.@nograd function _normalized_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj))
    inv_sqrtD = inv_sqrt_degree_matrix(adj, T, dir=:both)
    T.(I - inv_sqrtD * adj * inv_sqrtD)
end

@doc raw"""
    scaled_laplacian(adj::AbstractMatrix[, T::DataType])

Scaled Laplacien matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments
- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
"""
function scaled_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj))
    @assert adj == transpose(adj) "scaled_laplacian only works with symmetric matrices"
    E, U = symeigen(adj)
    T(2. / maximum(E)) * normalized_laplacian(adj, T) - I
end

"""
From https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/master/src/symeigen.jl
Only works with symmetric matrices
References:
    * Seeger, M., Hetzel, A., Dai, Z., Meissner, E., & Lawrence, N. D. (2018). Auto-Differentiating Linear Algebra.
"""
function symeigen(A::AbstractMatrix)
    E, U = LinearAlgebra.eigen(A)
    E, Matrix(U)
end
@adjoint function symeigen(A)
    E, U = symeigen(A)
    (E, U), adjy -> (symeigen_back(E, U, adjy...),)
end
function symeigen_back(E::AbstractVector{T}, U, dE, dU; η=1e-40) where T
    all(x->x isa Nothing, (dU, dE)) && return nothing
    η = T(η)
    if dU === nothing
        D = LinearAlgebra.Diagonal(dE)
    else
        F = E .- E'
        F .= F./(F.^2 .+ η)
        dUU = dU' * U .* F
        D = (dUU + dUU')/2
        if dE !== nothing
            D = D + LinearAlgebra.Diagonal(dE)
        end
    end
    U * D * U'
end
