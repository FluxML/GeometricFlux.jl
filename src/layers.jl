struct GCNConv{T,F}
    weight::AbstractMatrix{T}
    norm::AbstractMatrix{T}
    σ::F
end

function GCNConv(adj::AbstractMatrix{T}, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = randn) where {T}
    GCNConv(param(init(ch[1], ch[2])), normalized_laplacian(adj+I, Float64), σ)
end

(c::GCNConv)(X::AbstractMatrix) = c.σ(c.norm * X * c.weight)



struct ChebConv{T}
    weight::AbstractArray{T,3}
    L̃::AbstractMatrix{T}
    k::Integer
    in_channel::Integer
    out_channel::Integer
end

function ChebConv(adj::AbstractMatrix{T}, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = randn) where {T}
    L̃ = 2. / eigmax(adj) * normalized_laplacian(adj) - I
    ChebConv(param(init(k, ch[1], ch[2])), L̃, k, ch[1], ch[2])
end

function (c::ChebConv)(X::AbstractMatrix)
    fin = c.in_channel
    @assert size(X, 2) == fin "Input feature size must match input channel size."
    n = size(c.L̃, 1)
    @assert size(X, 1) == n "Input vertex number must match Laplacian matrix size."
    fout = c.out_channel

    T = eltype(X)
    Y = []
    Z = Array{T}(undef, n, c.k, fin)
    for j = 1:fout
        Z[:,1,:] = X
        Z[:,2,:] = c.L̃ * X
        for k = 3:c.k
            Z[:,k,:] = 2*c.L̃* view(Z, :, k-1, :) - view(Z, :, k-2, :)
        end

        y = view(Z, :, :, 1) * view(c.weight, :, 1, j)
        for i = 2:fin
            y += view(Z, :, :, i) * view(c.weight, :, i, j)
        end
        push!(Y, y)  # can be optimized
    end
    return hcat(Y...)
end
