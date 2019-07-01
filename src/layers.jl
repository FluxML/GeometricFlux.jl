# const aggr_func = Dict(:+ => scatter_add!, :max => scatter_max!)

abstract type MessagePassing end
message(m::T, xi, xj, eij) where {T<:MessagePassing} = error("not implement")
update(m::T, xi, mi) where {T<:MessagePassing} = error("not implement")

function propagate(mp::T, neighbors, X, E; aggr=+) where {T<:MessagePassing}
    # aggr in keys(aggr_func) || throw(DomainError(aggr, "not supported aggregation function."))
    # scatter_func = aggr_func[aggr]
    Y = Vector{AbstractArray}()
    for i = 1:length(neighbors)
        xi = view(X', :, i)
        ne = view(neighbors, i)
        xjs = map(j -> view(X', :, j), ne)
        eijs = map(j -> view(E, :, i, j), ne)
        m = map((xj, eij) -> message(mp, xi, xj, eij), xjs, eijs)[]
        mi = sum(m, dims=2)  # BUG: aggr(m...)
        push!(Y, update(mp, xi, mi))
    end
    return hcat(Y...)'
end



struct GCNConv{T,F}
    weight::AbstractMatrix{T}
    norm::AbstractMatrix{T}
    σ::F
end

function GCNConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, σ = identity;
                 init = glorot_uniform, T::DataType=Float32)
    GCNConv(param(init(ch[1], ch[2])), normalized_laplacian(adj+I, T), σ)
end

(c::GCNConv)(X::AbstractMatrix) = c.σ(c.norm * X * c.weight)



struct ChebConv{T}
    weight::AbstractArray{T,3}
    L̃::AbstractMatrix{T}
    k::Integer
    in_channel::Integer
    out_channel::Integer
end

function ChebConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, k::Integer;
                  init = glorot_uniform, T::DataType=Float32)
    L̃ = T(2. / eigmax(adj)) * normalized_laplacian(adj, T) - I
    ChebConv(param(init(k, ch[1], ch[2])), L̃, k, ch[1], ch[2])
end

function (c::ChebConv)(X::AbstractMatrix)
    fin = c.in_channel
    @assert size(X, 2) == fin "Input feature size must match input channel size."
    N = size(c.L̃, 1)
    @assert size(X, 1) == N "Input vertex number must match Laplacian matrix size."
    fout = c.out_channel

    T = eltype(X)
    Y = Vector{TrackedArray}()
    Z = Array{T}(undef, N, c.k, fin)
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



struct GraphConv{V,T,F}
    edgelist::V
    weight::AbstractMatrix{T}
    aggr::F
end

function GraphConv(el::AbstractVector{<:AbstractVector{<:Integer}},
                   ch::Pair{<:Integer,<:Integer}, aggr=+;
                   init = glorot_uniform)
    GraphConv(el, param(init(ch[1], ch[2])), aggr)
end

function GraphConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}, aggr=+;
                   init = glorot_uniform)
    GraphConv(neighbors(adj), param(init(ch[1], ch[2])), aggr)
end

function (g::GraphConv)(X::AbstractMatrix)
    N = size(X, 1)
    X_ = copy(X)'
    for i = 1:N
        ne = g.edgelist[i]
        X_[:,i] += sum(view(X', :, ne), dims=2)
    end
    X_' * g.weight
end



struct GATConv{V,T}
    edgelist::V
    weight::AbstractMatrix{T}
    a::AbstractArray
    negative_slope::Real
end

function GATConv(adj::AbstractMatrix, ch::Pair{<:Integer,<:Integer}; heads=1,
                 concat=true, negative_slope=0.2, init=glorot_uniform)
    GATConv(neighbors(adj), param(init(ch[1], ch[2])), param(init(2 * ch[2])), negative_slope)
end

function (g::GATConv)(X::AbstractMatrix)
    N = size(X, 1)
    X_ = (X * g.weight)'
    Y = Vector{TrackedArray}()
    for i = 1:N
        ne = g.edgelist[i]
        i_ne = vcat([i], ne)
        α = [leakyrelu(g.a' * vcat(X_[:,i], X_[:,j]), g.negative_slope) for j in i_ne]
        α = asoftmax(α)
        y = sum(α[i] * X_[:,i_ne[i]] for i = 1:length(α))
        push!(Y, y)
    end
    return hcat(Y...)'
end

function asoftmax(xs)
    xs = [exp.(x) for x in xs]
    s = sum(xs)
    return [x ./ s for x in xs]
end
