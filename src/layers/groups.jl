"""
    EEquivGraphConv((in_dim, int_dim, out_dim); init)
"""

struct EEquivGraphConv <: MessagePassing
   nn_edge
   nn_x
   nn_h

   in_dim
   int_dim
   out_dim
end

@functor EEquivGraphConv

function EEquivGraphConv(dims::NTuple{3,Int}; init=glorot_uniform)
    in_dim, int_dim, out_dim = dims

    m_len = in_dim * 2 + 2

    nn_edge = Flux.Dense(m_len, int_dim; init=init)

    nn_x = Flux.Dense(int_dim, 1; init=init)
    nn_h = Flux.Dense(in_dim + int_dim, out_dim; init=init)

    return EEquivGraphConv(nn_edge, nn_x, nn_h, dims...)
end

function EEquivGraphConv(nn_edge, nn_x, nn_h; init=glorot_uniform)

    # Assume that these are strictly MLPs (no conv)
    nn_edge.init(init)
    nn_x.init(init)
    nn_h.init(init)

    in_dim = nn_edge.layers[1].W |> x->size(x)[2]
    int_dim = nn_edge.layers[end].W |> x->size(x)[1]
    out_dim = nn_h.layers[end].W |> x->size(x)[1]
    return EEquivGraphConv(nn_edge, nn_x, nn_h, in_dim, int_dim, out_dim)
end

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

    input = vcat(h_i, h_j, sum(abs2.(x_i - x_j); dims=1), ones(N)' * a)
    edge_msg = egnn.nn_edge(input)
    output_vec = vcat(edge_msg, (x_i - x_j) .* egnn.nn_x(edge_msg)[1], ones(N)')
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
