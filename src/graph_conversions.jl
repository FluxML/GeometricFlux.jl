### CONVERT_TO_COO REPRESENTATION ########

function convert_to_coo(graph::COO_T; num_nodes=nothing)
    s, t = graph   
    num_nodes = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes 
    @assert length(s) == length(t)
    @assert min(minimum(s), minimum(t)) >= 1 
    @assert max(maximum(s), maximum(t)) <= num_nodes 

    num_edges = length(s)
    return graph, num_nodes, num_edges
end

function convert_to_coo(adj_mat::ADJMAT_T; dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = size(adj_mat, 1)
    @assert num_nodes == size(adj_mat, 2)
    @assert all(x -> (x == 1) || (x == 0), adj_mat)
    num_edges = round(Int, sum(adj_mat))
    s = zeros(Int, num_edges)
    t = zeros(Int, num_edges)
    e = 0
    for j in 1:num_nodes
        for i in 1:num_nodes
            if adj_mat[i, j] == 1
                e += 1
                s[e] = i
                t[e] = j
            end
        end
    end
    @assert e == num_edges
    if dir == :in
        s, t = t, s
    end
    return (s, t), num_nodes, num_edges
end

function convert_to_coo(adj_list::ADJLIST_T; dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    s = zeros(Int, num_edges)
    t = zeros(Int, num_edges)
    e = 0
    for i in 1:num_nodes
        for j in adj_list[i]
            e += 1
            s[e] = i
            t[e] = j 
        end
    end
    @assert e == num_edges
    if dir == :in
        s, t = t, s
    end
    (s, t), num_nodes, num_edges
end

########################################################################
