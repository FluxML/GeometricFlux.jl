
(g::GCNConv)(L̃::AbstractMatrix, X::Transpose{T,R}) where {T<:Real,R<:AbstractMatrix} = g(L̃, R(X))

(c::ChebConv)(L̃::AbstractMatrix, X::Transpose{T,R}) where {T<:Real,R<:AbstractMatrix} = c(L̃, R(X))
