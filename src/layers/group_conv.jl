"""
    EEquivGraphConv(in_dim, int_dim, out_dim; init=glorot_uniform)
    EEquivGraphConv(in_dim, nn_edge, nn_x, nn_h)

E(n)-equivariant graph neural network layer as defined in the paper "[E(n) Equivariant Neural Networks](https://arxiv.org/abs/2102.09844)" by Satorras, Hoogeboom, and Welling (2021).

# Arguments

Either one of two sets of arguments:

Set 1:

- `in_dim`: node feature dimension. Data is assumed to be of the form [feature; coordinate], so `in_dim` must strictly be less than the dimension of the input vectors.
- `int_dim`: intermediate dimension, can be arbitrary.
- `out_dim`: the output of the layer will have dimension `out_dim` + (dimension of input vector - `in_dim`).
- `init`: neural network initialization function, should be compatible with `Flux.Dense`.

Set 2:

- `in_dim`: as in Set 1.
- `nn_edge`: a differentiable function that must take vectors of dimension `in_dim * 2 + 2` (output designated `int_dim`)
- `nn_x`: a differentiable function that must take vectors of dimension `int_dim` to dimension `1`.
- `nn_h`: a differentiable function that must take vectors of dimension `in_dim + int_dim` to `out_dim`.

```jldoctest
julia> in_dim, int_dim, out_dim = 3,6,5
(3, 5, 5)

julia> egnn = EEquivGraphConv(in_dim, int_dim, out_dim)
EEquivGraphConv{Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}(Dense(8 => 5), Dense(5 => 1), Dense(8 => 5), 3, 5, 5)

julia> m_len = 2*in_dim + 2
8

julia> nn_edge = Flux.Dense(m_len, int_dim)
Dense(8 => 5)       # 45 parameters

julia> nn_x = Flux.Dense(int_dim, 1)
Dense(5 => 1)       # 6 parameters

julia> nn_h = Flux.Dense(in_dim + int_dim, out_dim)
Dense(8 => 5)       # 45 parameters

julia> egnn = EEquivGraphConv(in_dim, nn_edge, nn_x, nn_h)
EEquivGraphConv{Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}, Dense{typeof(identity), Matrix{Float32}, Vector{Float32}}}(Dense(8 => 5), Dense(5 => 1), Dense(8 => 5), 3, 5, 5)
```
"""

struct EEquivGraphConv{E,X,H} <: MessagePassing
   nn_edge::E
   nn_x::X
   nn_h::H
   in_dim::Int
   int_dim::Int
   out_dim::Int
end

@functor EEquivGraphConv

function EEquivGraphConv(in_dim::Int, int_dim::Int, out_dim::Int; init=glorot_uniform)
    m_len = 2in_dim + 2
    nn_edge = Flux.Dense(m_len, int_dim; init=init)
    nn_x = Flux.Dense(int_dim, 1; init=init)
    nn_h = Flux.Dense(in_dim + int_dim, out_dim; init=init)
    return EEquivGraphConv(nn_edge, nn_x, nn_h, in_dim, int_dim, out_dim)
end

function EEquivGraphConv(in_dim::Int, nn_edge, nn_x, nn_h)
    m_len = 2in_dim + 2
    int_dim = Flux.outputsize(nn_edge, (m_len, 2))[1]
    out_dim = Flux.outputsize(nn_h, (in_dim + int_dim, 2))[1]
    return EEquivGraphConv(nn_edge, nn_x, nn_h, in_dim, int_dim, out_dim)
end

function ϕ_edge(egnn::EEquivGraphConv, h_i, h_j, dist, a)
    N = size(h_i, 2)
    return egnn.nn_edge(vcat(h_i, h_j, dist, ones(N)' * a))
end

ϕ_x(egnn::EEquivGraphConv, m_ij) = egnn.nn_x(m_ij)

function message(egnn::EEquivGraphConv, v_i, v_j, e)
    in_dim = egnn.in_dim
    h_i = v_i[1:in_dim,:]
    h_j = v_j[1:in_dim,:]

    N = size(h_i, 2)

    x_i = v_i[in_dim+1:end,:]
    x_j = v_j[in_dim+1:end,:]

    if isnothing(e)
        a = 1
    else
        a = e[1]
    end

    dist = sum(abs2.(x_i - x_j); dims=1)
    edge_msg = ϕ_edge(egnn, h_i, h_j, dist, a)
    output_vec = vcat(edge_msg, (x_i - x_j) .* ϕ_x(egnn, edge_msg)[1], ones(N)')
    return reshape(output_vec, :, N)
end

function update(e::EEquivGraphConv, m, h)
    N = size(m, 2)
    mi = m[1:e.int_dim,:]
    x_msg = m[e.int_dim+1:end-1,:]
    M = m[end,:]

    C = 1 ./ (M.-1)
    C = reshape(C, :, N)

    nn_node_out = e.nn_h(vcat(h[1:e.in_dim,:], mi))

    coord_dim = size(h,1) - e.in_dim

    z = zeros(e.out_dim + coord_dim, N)
    z[1:e.out_dim,:] = nn_node_out
    z[e.out_dim+1:end,:] = h[e.in_dim+1:end,:] + C .* x_msg
    return z
end

function(egnn::EEquivGraphConv)(fg::AbstractFeaturedGraph)
    X = node_feature(fg)
    GraphSignals.check_num_nodes(fg, X)
    _, V, _ = propagate(egnn, graph(fg), nothing, X, nothing, +, nothing, nothing)
    return ConcreteFeaturedGraph(fg, nf=V)
end
