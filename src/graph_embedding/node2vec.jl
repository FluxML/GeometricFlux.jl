const Alias = Tuple{SparseVector{Int}, SparseVector{Float64}}

"""
    node2vec(g; walks_per_node, len, p, q)

Computes node embeddings on graph `g`, as per [1]. Performs biased random walks on the graph,
then computes word embeddings by treating those random walks like sentences.

# Arguments

- `g::FeaturedGraph`: The graph to perform random walk on.
- `walks_per_node::Int`: Number of walks starting on each node,
total number of walks is `nv(g) * walks_per_node`
- `len::Int`: Length of random walks
- `p::Real`: Return parameter from [1]
- `q::Real`: In-out parameter from [1]

[1] https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf
"""
function node2vec(g::FeaturedGraph; walks_per_node::Int=100, len::Int=5, p::Real=0.5, q::Real=0.5)
    walks = simulate_walks(g; walks_per_node=walks_per_node, len=len, p=p, q=q)
    model = walks2vec(walks)
    return copy(model.vectors)
end

"""
Modified version of Node2Vec.learn_embeddings[1]. Uses
a Julia interface[2] to the original word2vec C code[3].

Treats each random walk like a sentence, and computed word
embeddings using node ID as words.

[1] https://github.com/ollin18/Node2Vec.jl
[2] https://github.com/JuliaText/Word2Vec.jl
[3] https://code.google.com/archive/p/word2vec/
"""
function walks2vec(walks::Vector{Vector{Int}};size::Int=100)

    str_walks=map(x -> string.(x),walks)

    if Sys.iswindows()
        rpath = pwd()
    else
        rpath = "/tmp"
    end
    the_walks = joinpath(rpath,"str_walk.txt")
    the_vecs = joinpath(rpath,"str_walk-vec.txt")

    writedlm(the_walks,str_walks)
    word2vec(the_walks,the_vecs,verbose=true,size=size)
    model=wordvectors(the_vecs)
    rm(the_walks)
    rm(the_vecs)
    model
end


"""
    Conducts a random walk over `g` in O(l) time,
weighted by alias sampling probabilities `alias_nodes`
and `alias_edges`
"""
function node2vec_walk(
    g::FeaturedGraph,
    alias_nodes::Dict{Int, Alias},
    alias_edges::Dict{Tuple{Int, Int}, Alias};
    start_node::Int,
    walk_length::Int)::Vector{Int}
    walk::Vector{Int} = [start_node]
    current::Int = start_node
    for _ in 2:walk_length
        curr = walk[end]
        cur_nbrs = sort(outneighbors(g, curr))
        if length(walk) == 1
            push!(walk, cur_nbrs[alias_sample(alias_nodes[curr]...)])
        else
            prev = walk[end-1]
            next = cur_nbrs[alias_sample(alias_edges[(prev, curr)]...)]
            push!(walk, next)
        end
    end
    return walk
end

"Returns J and q for a given edge"
function get_alias_edge(g::FeaturedGraph, src::Int, dst::Int, p::Float64, q::Float64)::Alias
    unnormalized_probs = spzeros(length(outneighbors(g, dst)))
    for (i, (dst_nbr, weight)) in enumerate(weighted_outneighbors(g, dst))
        if dst_nbr == src
            unnormalized_probs[i] = weight/p
        elseif has_edge(g, dst_nbr, src)
            unnormalized_probs[i] = weight
        else
            unnormalized_probs[i] = weight/q
        end
    end
    normalized_probs = unnormalized_probs ./ sum(unnormalized_probs)
    return alias_setup(normalized_probs)
end

"""
    Computes weighted probability transition aliases J and q for nodes and edges
using return parameter `p` and In-out parameter `q`

Implementation as specified in the node2vec paper by Grover and Leskovec (2016)
https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf
"""
function preprocess_modified_weights(g::FeaturedGraph, p::Float64, q::Float64)

    alias_nodes::Dict{Int, Alias} = Dict()
    alias_edges::Dict{Tuple{Int, Int}, Alias} = Dict()

    for node in vertices(g)
        probs = [1 / length(outneighbors(g, node)) for _ in outneighbors(g, node)]
        alias_nodes[node] =  alias_setup(probs)
    end
    for edge in edges(g)
        alias_edges[(edge.src, edge.dst)] = get_alias_edge(g, edge.src, edge.dst, p, q)
        if !is_directed(g)
            alias_edges[(edge.dst, edge.src)] = get_alias_edge(g, edge.dst, edge.src, p, q)
        end
    end
    return alias_nodes, alias_edges
end


"Given a graph, compute `walks_per_node` * nv(g) random walks."
function simulate_walks(g::FeaturedGraph; walks_per_node::Int, len::Int, p::Float64, q::Float64)::Vector{Vector{Int}}
    alias_nodes, alias_edges = preprocess_modified_weights(g, p, q)
    walks::Vector{Vector{Int}} = []
    for _ in 1:walks_per_node
        for node in shuffle(vertices(g))
            walk::Vector{Int} = node2vec_walk(g, alias_nodes, alias_edges; start_node=node, walk_length=len)
            push!(walks, walk)
        end
    end
    return walks
end
