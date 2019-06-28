struct GCNConv{T,F}
    weight::AbstractMatrix{T}
    norm::AbstractMatrix{T}
    σ::F
end

function GCNConv(adj::AbstractMatrix{T}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = randn) where {T}
    GCNConv(param(init(ch[1], ch[2])), param(normalized_laplacian(adj+I, Float64)), σ)
end

(c::GCNConv)(X::AbstractMatrix) = c.σ(c.norm * X * c.weight)
