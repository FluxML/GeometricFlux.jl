const Alias = Tuple{SparseVector{Int}, SparseVector{Float64}}

"""
    node2vec(g; walks_per_node, len, p, q, dims)

Returns an embedding matrix with size of (`nv(g)`, `dims`). It computes node embeddings
on graph `g`. It performs biased random walks on the graph,
then computes word embeddings by treating those random walks as sentences.

# Arguments

- `g::FeaturedGraph`: The graph to perform random walk on.
- `walks_per_node::Int`: Number of walks starting on each node, total number of walks is `nv(g) * walks_per_node`
- `len::Int`: Length of random walks
- `p::Real`: Return parameter. It controls the likelihood of immediately revisiting a node in the walk
- `q::Real`: In-out parameter. It allows the search to differentiate between inward and outward nodes.
- `dims::Int`: Number of vector dimensions
"""
function node2vec(g::FeaturedGraph; walks_per_node::Int=100, len::Int=5, p::Real=0.5, q::Real=0.5, dims::Int=128)
    walks = simulate_walks(g; walks_per_node=walks_per_node, len=len, p=p, q=q)
    model = walks2vec(walks,dims=dims)
    vecs = []
    println(typeof(model))
    for i in 1:nv(g)
        push!(vecs, get_vector(model, string(i)))
    end
    matrix = cat(vecs..., dims=2)
    return matrix
end

"""
Modified version of Node2Vec.learn_embeddings from Node2Vec.jl. Uses
Word2Vec.jl to the original word2vec C code [google_word2vec](@cite).

Treats each random walk like a sentence, and computed word
embeddings using node ID as words.
"""
function walks2vec(walks::Vector{Vector{Int}}; dims::Int=100)
    str_walks=map(x -> string.(x),walks)

    if Sys.iswindows()
        rpath = pwd()
    else
        rpath = "/tmp"
    end
    the_walks = joinpath(rpath,"str_walk.txt")
    the_vecs = joinpath(rpath,"str_walk-vec.txt")

    writedlm(the_walks,str_walks)
    word2vec(the_walks,the_vecs,verbose=true,size=dims)
    model=wordvectors(the_vecs)
    rm(the_walks)
    rm(the_vecs)
    model
end


"""
Conducts a random walk over `g` in O(l) time.
It is weighted by alias sampling probabilities `alias_nodes` and `alias_edges`.
"""
function node2vec_walk(
    g::FeaturedGraph,
    alias_nodes::Dict{Int, Alias},
    alias_edges::Dict{Tuple{Int, Int}, Alias};
    start_node::Int,
    walk_length::Int)::Vector{Int}
    walk::Vector{Int} = [start_node]
    for _ in 2:walk_length
        curr = walk[end]
        cur_nbrs = sort(neighbors(g, curr; dir=:out))
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

"""
Returns J and q for a given edge
"""
function get_alias_edge(g::FeaturedGraph, src::Integer, dst::Integer, p::Real, q::Real)::Alias
    unnormalized_probs = spzeros(length(neighbors(g, dst; dir=:out)))
    neighbor_weight_pairs = zip(weighted_outneighbors(g, dst)...)
    for (i, (dst_nbr, weight)) in enumerate(neighbor_weight_pairs)
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

# Returns (neighbors::Vector{Int}, weights::Vector{Float64})
function weighted_outneighbors(fg::FeaturedGraph, i::Integer)
    nbrs = neighbors(fg, i; dir=:out)
    nbrs, sparse(graph(fg))[i, nbrs]
end

"""
Computes weighted probability transition aliases J and q for nodes and edges
using return parameter `p` and in-out parameter `q`.
"""
function preprocess_modified_weights(g::FeaturedGraph, p::Real, q::Real)

    alias_nodes = Dict{Int, Alias}()
    alias_edges = Dict{Tuple{Int, Int}, Alias}()

    for node in 1:nv(g)
        nbrs = neighbors(g, node, dir=:out)
        probs = fill(1, length(nbrs)) ./ length(nbrs)
        alias_nodes[node] =  alias_setup(probs)
    end
    for (_, edge) in edges(g)
        src, dst = edge
        alias_edges[(src, dst)] = get_alias_edge(g, src, dst, p, q)
        if !is_directed(g)
            alias_edges[(dst, src)] = get_alias_edge(g, dst, src, p, q)
        end
    end
    return alias_nodes, alias_edges
end


"""
Given a graph, compute `walks_per_node` * nv(g) random walks.
"""
function simulate_walks(g::FeaturedGraph; walks_per_node::Int, len::Int, p::Real, q::Real)::Vector{Vector{Int}}
    alias_nodes, alias_edges = preprocess_modified_weights(g, p, q)
    walks = Vector{Int}[]
    for _ in 1:walks_per_node
        for node in shuffle(1:nv(g))
            walk::Vector{Int} = node2vec_walk(g, alias_nodes, alias_edges; start_node=node, walk_length=len)
            push!(walks, walk)
        end
    end
    return walks
end
