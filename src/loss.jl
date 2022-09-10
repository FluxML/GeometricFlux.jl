@doc raw"""
    laplacian_eig_loss(p, L, λ)

Laplacian eigenvector loss for position embeddings `p` and graph Laplacian `L`, defined as

```math
\frac{1}{k} trace(p^T \mathcal{L} p) + \frac{\lambda}{k} \lVert p^Tp - I \rVert_{F}^2
```

where ``\mathcal{L}`` is graph Laplacian (same as `L`) and ``\lVert \cdot \rVert_{F}^2`` is the Frobenius norm.

# Arguments

- `p::AbstractMatrix`: The position embeddings with dimensions of ``(k, N)``.
- `L::AbstractMatrix`: The Laplacian with dimensions of ``(N, N)`` from a graph which has
    ``N`` nodes.
- `λ::Real`: Regularization term and it must be positive.
"""
function laplacian_eig_loss(p::AbstractMatrix, L::AbstractMatrix, λ::Real)
    @assert λ > 0 "λ must be positive."
    k = size(p, 1)
    p = (p .- mean(p, dims=2)) ./ sqrt.(sum(abs2, p, dims=2))
    return (tr(p * L * p') + λ * sum(abs2, p*p' - I)) / k
end
