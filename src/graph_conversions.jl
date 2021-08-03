### CONVERT_TO_COO REPRESENTATION ########

function to_coo(eindex::COO_T; dir=:out, num_nodes=nothing)
    s, t = eindex   
    num_nodes = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes 
    @assert length(s) == length(t)
    @assert min(minimum(s), minimum(t)) >= 1 
    @assert max(maximum(s), maximum(t)) <= num_nodes 

    num_edges = length(s)
    return eindex, num_nodes, num_edges
end

function to_coo(adj_mat::ADJMAT_T; dir=:out, num_nodes=nothing)
    nz = findall(!=(0), adj_mat) # vec of cartesian indexes
    s, t = ntuple(i -> map(t->t[i], nz), 2)
    if dir == :in
        s, t = t, s
    end
    num_nodes = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes 
    num_edges = length(s)
    return (s, t), num_nodes, num_edges
end

function to_coo(adj_list::ADJLIST_T; dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    @assert num_nodes > 0
    s = similar(adj_list[1], eltype(adj_list[1]), num_edges)
    t = similar(adj_list[1], eltype(adj_list[1]), num_edges)
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

### CONVERT TO ADJACENCY MATRIX ################

### DENSE ####################

to_dense(adj_mat::AbstractSparseMatrix, x...; kws...) = to_dense(collect(adj_mat), x...; kws...)

function to_dense(adj_mat::ADJMAT_T, T::DataType=eltype(adj_mat); dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = size(adj_mat, 1)
    @assert num_nodes == size(adj_mat, 2)
    # @assert all(x -> (x == 1) || (x == 0), adj_mat)
    num_edges = round(Int, sum(adj_mat))
    if dir == :in
        adj_mat = adj_mat'
    end
    if T != eltype(adj_mat)
        adj_mat = T.(adj_mat)
    end
    return adj_mat, num_nodes, num_edges
end

function to_dense(adj_list::ADJLIST_T, T::DataType=Int; dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    @assert num_nodes > 0
    A = similar(adj_list[1], T, (num_nodes, num_nodes))
    if dir == :out
        for (i, neigs) in enumerate(adj_list)
            A[i, neigs] .= 1
        end
    else 
        for (i, neigs) in enumerate(adj_list)
            A[neigs, i] .= 1
        end
    end
    A, num_nodes, num_edges
end

function to_dense(eindex::COO_T, T::DataType=Int; dir=:out, num_nodes=nothing)
    # `dir` will be ignored since the input `eindex` is alwasys in source target format.
    # The output will always be a adjmat in :out format (e.g. A[i,j] denotes from i to j)
    s, t = eindex
    n = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes
    adj_mat = fill!(similar(s, T, (n, n)), 0)
    adj_mat[s .+ n .* (t .- 1)] .= 1 # exploiting linear indexing
    return adj_mat, n, length(s)
end

### SPARSE #############

function to_sparse(adj_mat::ADJMAT_T, T::DataType=eltype(adj_mat); dir=:out, num_nodes=nothing)
    @assert dir ∈ [:out, :in]
    num_nodes = size(adj_mat, 1)
    @assert num_nodes == size(adj_mat, 2)
    # @assert all(x -> (x == 1) || (x == 0), adj_mat)
    num_edges = round(Int, sum(adj_mat))
    if dir == :in
        adj_mat = adj_mat'
    end
    if T != eltype(adj_mat)
        adj_mat = T.(adj_mat)
    end
    return sparse(adj_mat), num_nodes, num_edges
end

function to_sparse(adj_list::ADJLIST_T, T::DataType=Int; dir=:out, num_nodes=nothing)
    eindex, num_nodes, num_edges = to_coo(adj_list; dir, num_nodes)
    to_sparse(eindex; dir, num_nodes)
end

function to_sparse(eindex::COO_T, T::DataType=Int; dir=:out, num_nodes=nothing)
    s, t = eindex    
    val = fill!(similar(s, T), 1)
    A = sparse(s, t, val)
    num_nodes = isnothing(num_nodes) ? max(maximum(s), maximum(t)) : num_nodes 
    num_edges = length(s)
    A, num_nodes, num_edges
end

@non_differentiable to_coo(x...)
@non_differentiable to_dense(x...)
@non_differentiable to_sparse(x...)
