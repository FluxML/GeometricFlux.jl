#===================================
Define FeaturedGraph type as a subtype of LightGraphs' AbstractGraph.

All LightGraphs functions rely on a standard API to function. 
As long as your graph structure is a subtype of AbstractGraph and 
implements the following API functions with the given return values, 
all functions within the LightGraphs package should just work:
    edges
    Base.eltype
    edgetype (example: edgetype(g::CustomGraph) = LightGraphs.SimpleEdge{eltype(g)}))
    has_edge
    has_vertex
    inneighbors
    ne
    nv
    outneighbors
    vertices
    is_directed(::Type{CustomGraph})::Bool (example: is_directed(::Type{<:CustomGraph}) = false)
    is_directed(g::CustomGraph)::Bool
    zero
=============================================#

abstract type AbstractFeaturedGraph <: AbstractGraph{Int} end

"""
    NullGraph()

Null object for `FeaturedGraph`.
"""
struct NullGraph <: AbstractFeaturedGraph end


struct FeaturedGraph <: AbstractFeaturedGraph
    edge_index
    num_nodes::Int
    num_edges::Int
    # ndata::Dict{String, Any} # https://github.com/FluxML/Zygote.jl/issues/717        
    # edata::Dict{String, Any}
    # gdata::Dict{String, Any}
    nf
    ef
    gf
end


function FeaturedGraph(u::AbstractVector{Int}, v::AbstractVector{Int}; 
                        num_nodes = max(maximum(u), maximum(v)), 
                        # ndata = Dict{String, Any}(), 
                        # edata = Dict{String, Any}(),
                        # gdata = Dict{String, Any}(),
                        nf = nothing, 
                        ef = nothing, 
                        gf = nothing)

    @assert length(u) == length(v)
    @assert min(minimum(u), minimum(v)) >= 1 
    @assert max(maximum(u), maximum(v)) <= num_nodes 
    
    num_edges = length(u)
    
    ## I would like to have dict data store, but currently this 
    ## doesn't play well with zygote due to 
    ## https://github.com/FluxML/Zygote.jl/issues/717    
    # ndata["x"] = nf
    # edata["e"] = ef
    # gdata["g"] = gf
    

    FeaturedGraph((u, v), num_nodes, num_edges, 
                   nf, ef, gf)
end

# Construct from adjacency matrix # TODO deprecate?
function FeaturedGraph(adj_mat::AbstractMatrix; dir=:out, kws...)
    @assert dir == :out  # TODO
    num_nodes = size(adj_mat, 1)
    @assert num_nodes == size(adj_mat, 2)
    @assert all(x -> (x == 1) || (x == 0), adj_mat)
    num_edges = round(Int, sum(adj_mat))
    u = zeros(Int, num_edges)
    v = zeros(Int, num_edges)
    e = 0
    for j in 1:num_nodes
        for i in 1:num_nodes
            if adj_mat[i, j] == 1
                e += 1
                u[e] = i
                v[e] = j
            end
        end
    end
    @assert e == num_edges
    FeaturedGraph(u, v; num_nodes, kws...)
end


# Construct from adjacency list # TODO deprecate?
function FeaturedGraph(adj_list::AbstractVector{<:AbstractVector}; dir=:out, kws...)
    @assert dir == :out  # TODO
    num_nodes = length(adj_list)
    num_edges = sum(length.(adj_list))
    u = zeros(Int, num_edges)
    v = zeros(Int, num_edges)
    e = 0
    for i in 1:num_nodes
        for j in adj_list[i]
            e += 1
            u[e] = i
            v[e] = j 
        end
    end
    @assert e == num_edges
    FeaturedGraph(u, v; num_nodes, kws...)
end


# from other featured_graph 
function FeaturedGraph(fg::FeaturedGraph; 
                # ndata=copy(fg.ndata), edata=copy(fg.edata), gdata=copy(fg.gdata), # copy keeps the refs to old data 
                nf=node_feature(fg), ef=edge_feature(fg), gf=global_feature(fg))
    
    FeaturedGraph(fg.edge_index[1], fg.edge_index[2]; 
                #   ndata, edata, gdata, 
                  nf, ef, gf)
end

@functor FeaturedGraph

edge_index(fg::FeaturedGraph) = fg.edge_index

LightGraphs.edges(fg::FeaturedGraph) = zip(fg.edge_index[1], fg.edge_index[2])

LightGraphs.edgetype(fg::FeaturedGraph) = Tuple{eltype(fg.edge_index[1]), eltype(fg.edge_index[2])}

function LightGraphs.has_edge(fg::FeaturedGraph, i::Integer, j::Integer)
    u, v = fg.edge_index
    return any((u .== i) .& (v .== j))
end

LightGraphs.nv(fg::FeaturedGraph) = fg.num_nodes
LightGraphs.ne(fg::FeaturedGraph) = fg.num_edges
LightGraphs.has_vertex(fg::FeaturedGraph, i::Int) = i in 1:fg.num_nodes
LightGraphs.vertices(fg::FeaturedGraph) = 1:fg.num_nodes

function LightGraphs.outneighbors(fg::FeaturedGraph, i::Integer)
    u, v = fg.edge_index
    return v[u .== i]
end

function LightGraphs.inneighbors(fg::FeaturedGraph, i::Integer)
    u, v = fg.edge_index
    return u[v .== i]
end

LightGraphs.is_directed(::FeaturedGraph) = true
LightGraphs.is_directed(::Type{FeaturedGraph}) = true

function adjacency_list(fg::FeaturedGraph; dir=:out)
    # TODO probably this has to be called with `dir=:in` by gnn layers
    # TODO dir=:both
    fneighs = dir == :out ? outneighbors : inneighbors
    return [fneighs(fg, i) for i in 1:fg.num_nodes]
end

# TODO return sparse matrix
function LightGraphs.adjacency_matrix(fg::FeaturedGraph, T::DataType=Int; dir=:out)
    # TODO dir=:both
    u, v = fg.edge_index
    n = fg.num_nodes
    adj_mat = zeros(T, n, n)
    adj_mat[u .+ n .* (v .- 1)] .= 1 # exploiting linear indexing
    return dir == :out ? adj_mat : adj_mat'
end

Zygote.@nograd adjacency_matrix, adjacency_list


# function ChainRulesCore.rrule(::typeof(copy), x)
#     copy_pullback(ȳ) = (NoTangent(), ȳ)
#     return copy(x), copy_pullback
# end

# node_feature(fg::FeaturedGraph) = fg.ndata["x"]
# edge_feature(fg::FeaturedGraph) = fg.edata["e"]
# global_feature(fg::FeaturedGraph) = fg.gdata["g"]

node_feature(fg::FeaturedGraph) = fg.nf
edge_feature(fg::FeaturedGraph) = fg.ef
global_feature(fg::FeaturedGraph) = fg.gf

## TO DEPRECATE EVERYTHING BELOW ??? ##############################

# has_graph(fg::FeaturedGraph) = true
# has_graph(fg::NullGraph) = false
# graph(fg::FeaturedGraph) = adjacency_list(fg) # DEPRECATE

# function Base.getproperty(fg::FeaturedGraph, sym::Symbol)
#     if sym === :nf
#         return fg.ndata["x"]
#     elseif sym === :ef
#         return fg.edata["e"]
#     elseif sym === :gf
#         return fg.gdata["g"]
#     else # fallback to getfield
#         return getfield(fg, sym)
#     end
# end

## Already in GraphSignals ##############
LightGraphs.ne(adj_list::AbstractVector{<:AbstractVector}) = sum(length.(adj_list))
LightGraphs.nv(adj_list::AbstractVector{<:AbstractVector}) = length(adj_list)
LightGraphs.ne(adj_mat::AbstractMatrix) = round(Int, sum(adj_mat))
LightGraphs.nv(adj_mat::AbstractMatrix) = size(adj_mat, 1)

adjacency_list(adj::AbstractVector{<:AbstractVector}) = adj

function LightGraphs.is_directed(g::AbstractVector{T}) where {T<:AbstractVector}
    edges = Set{Tuple{Int64,Int64}}()
    for (i, js) in enumerate(g)
        for j in Set(js)
            if i != j
                e = (i,j)
                if e in edges
                    pop!(edges, e)
                else
                    push!(edges, (j,i))
                end
            end
        end
    end
    !isempty(edges)
end

LightGraphs.is_directed(g::AbstractMatrix) = !issymmetric(Matrix(g))

# function LightGraphs.laplacian_matrix(fg::FeaturedGraph, T::DataType=Int; dir::Symbol=:out)
#     A = adjacency_matrix(fg, T; dir=dir)
#     D = Diagonal(vec(sum(A; dims=2)))
#     return D - A
# end

## from GraphLaplacians

"""
    normalized_laplacian(g[, T]; selfloop=false, dir=:out)

Normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function normalized_laplacian(fg::FeaturedGraph, T::DataType=Int; selfloop::Bool=false, dir::Symbol=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    selfloop && (A += I)
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    return I - inv_sqrtD * A * inv_sqrtD
end

@doc raw"""
    scaled_laplacian(g[, T]; dir=:out)

Scaled Laplacian matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from LightGraphs) or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: the edge directionality considered (:out, :in, :both).
"""
function scaled_laplacian(fg::FeaturedGraph, T::DataType=Int; dir=:out)
    A = adjacency_matrix(fg, T; dir=dir)
    @assert issymmetric(A) "scaled_laplacian only works with symmetric matrices"
    E = eigen(Symmetric(A)).values
    degs = vec(sum(A; dims=2))
    inv_sqrtD = Diagonal(inv.(sqrt.(degs)))
    Lnorm = I - inv_sqrtD * A * inv_sqrtD
    return  2 / maximum(E) * Lnorm - I
end


function add_self_loop!(adj::AbstractVector{<:AbstractVector})
    for i = 1:length(adj)
        i in adj[i] || push!(adj[i], i)
    end
    adj
end

# # TODO Do we need a separate package just for laplacians?
# GraphLaplacians.scaled_laplacian(fg::FeaturedGraph, T::DataType) =
#     scaled_laplacian(adjacency_matrix(fg, T))
# GraphLaplacians.normalized_laplacian(fg::FeaturedGraph, T::DataType; kws...) =
#     normalized_laplacian(adjacency_matrix(fg, T); kws...)


@non_differentiable normalized_laplacian(x...)
@non_differentiable scaled_laplacian(x...)
@non_differentiable add_self_loop!(x...)
