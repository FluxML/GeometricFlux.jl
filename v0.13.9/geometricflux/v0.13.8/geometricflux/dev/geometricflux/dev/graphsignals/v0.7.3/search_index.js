var documenterSearchIndex = {"docs":
[{"location":"manual/featuredgraph/#FeaturedGraph","page":"FeaturedGraph","title":"FeaturedGraph","text":"","category":"section"},{"location":"manual/featuredgraph/#Construct-a-FeaturedGraph-and-graph-representations","page":"FeaturedGraph","title":"Construct a FeaturedGraph and graph representations","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"A FeaturedGraph is aimed to represent a composition of graph representation and graph signals. A graph representation is required to construct a FeaturedGraph object. Graph representation can be accepted in several forms: adjacency matrix, adjacency list or graph representation provided from JuliaGraphs.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> adj = [0 1 1;\n              1 0 1;\n              1 1 0]\n3×3 Matrix{Int64}:\n 0  1  1\n 1  0  1\n 1  1  0\n\njulia> FeaturedGraph(adj)\nFeaturedGraph(\n\tUndirected graph with (#V=3, #E=3) in adjacency matrix,\n)","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Currently, SimpleGraph and SimpleDiGraph from LightGraphs.jl, SimpleWeightedGraph and SimpleWeightedDiGraph from SimpleWeightedGraphs.jl, as well as MetaGraph and MetaDiGraph from MetaGraphs.jl are supported.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"If a graph representation is not given, a FeaturedGraph object will be regarded as a NullGraph. A NullGraph object is just used as a special case of FeaturedGraph to represent a null object.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> FeaturedGraph()\nNullGraph()","category":"page"},{"location":"manual/featuredgraph/#FeaturedGraph-constructors","page":"FeaturedGraph","title":"FeaturedGraph constructors","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"NullGraph()","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"FeaturedGraph","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.FeaturedGraph","page":"FeaturedGraph","title":"GraphSignals.FeaturedGraph","text":"FeaturedGraph(g, [mt]; directed=:auto, nf, ef, gf, pf=nothing,\n    T, N, E, with_batch=false)\n\nA type representing a graph structure and storing also arrays  that contain features associated to nodes, edges, and the whole graph. \n\nA FeaturedGraph can be constructed out of different objects g representing the connections inside the graph. When constructed from another featured graph fg, the internal graph representation is preserved and shared.\n\nArguments\n\ng: Data representing the graph topology. Possible type are \nAn adjacency matrix.\nAn adjacency list.\nA Graphs' graph, i.e. SimpleGraph, SimpleDiGraph from Graphs, or SimpleWeightedGraph,   SimpleWeightedDiGraph from SimpleWeightedGraphs.\nAn AbstractFeaturedGraph object.\nmt::Symbol: Matrix type for g in matrix form. if graph is in matrix form, mt is   recorded as one of :adjm, :normedadjm, :laplacian, :normalized or :scaled.\ndirected: It specify that direction of a graph. It can be :auto, :directed and   :undirected. Default value is :auto, which infers direction automatically.\nnf: Node features.\nef: Edge features.\ngf: Global features.\npf: Positional features. If nothing is given, positional encoding is turned off. If an   array is given, positional encoding is assigned as given array. If :auto is given,   positional encoding is generated automatically for node features and with_batch is considered.\nT: It specifies the element type of graph. Default value is the element type of g.\nN: Number of nodes for g.\nE: Number of edges for g.\nwith_batch::Bool: Consider last dimension of all features as batch dimension.\n\nUsage\n\nusing GraphSignals, CUDA\n\n# Construct from adjacency list representation\ng = [[2,3], [1,4,5], [1], [2,5], [2,4]]\nfg = FeaturedGraph(g)\n\n# Number of nodes and edges\nnv(fg)  # 5\nne(fg)  # 10\n\n# From a Graphs' graph\nfg = FeaturedGraph(erdos_renyi(100, 20))\n\n# Copy featured graph while also adding node features\nfg = FeaturedGraph(fg, nf=rand(100, 5))\n\n# Send to gpu\nfg = fg |> cu\n\nSee also graph, node_feature, edge_feature, and global_feature.\n\n\n\n\n\n","category":"type"},{"location":"manual/featuredgraph/#Graph-Signals","page":"FeaturedGraph","title":"Graph Signals","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Graph signals is a collection of any signals defined on a graph. Graph signals can be the signals related to vertex, edges or graph itself. If a vertex signal is given, it is recorded as a node feature in FeaturedGraph. A node feature is stored as the form of generic array, of which type is AbstractArray. A node feature can be indexed by the node index, which is the same index for given graph.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Node features can be optionally given in construction of a FeaturedGraph.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> fg = FeaturedGraph(adj, nf=rand(5, 3))\nFeaturedGraph(\n\tUndirected graph with (#V=3, #E=3) in adjacency matrix,\n\tNode feature:\tℝ^5 <Matrix{Float64}>,\n)\n\njulia> has_node_feature(fg)\ntrue\n\njulia> node_feature(fg)\n5×3 Matrix{Float64}:\n 0.534928  0.719566  0.952673\n 0.395465  0.268515  0.335446\n 0.79428   0.18623   0.454377\n 0.530675  0.402474  0.00920068\n 0.642556  0.719674  0.772497","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Users check node/edge/graph features are available by has_node_feature, has_edge_feature and has_global_feature, respectively, and fetch these features by node_feature, edge_feature and global_feature.","category":"page"},{"location":"manual/featuredgraph/#Getter-methods","page":"FeaturedGraph","title":"Getter methods","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"graph","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.graph","page":"FeaturedGraph","title":"GraphSignals.graph","text":"graph(fg)\n\nGet referenced graph in fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"node_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.node_feature","page":"FeaturedGraph","title":"GraphSignals.node_feature","text":"node_feature(fg)\n\nGet node feature attached to fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"edge_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.edge_feature","page":"FeaturedGraph","title":"GraphSignals.edge_feature","text":"edge_feature(fg)\n\nGet edge feature attached to fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"global_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.global_feature","page":"FeaturedGraph","title":"GraphSignals.global_feature","text":"global_feature(fg)\n\nGet global feature attached to fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/#Check-methods","page":"FeaturedGraph","title":"Check methods","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"has_graph","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.has_graph","page":"FeaturedGraph","title":"GraphSignals.has_graph","text":"has_graph(fg)\n\nCheck if graph is available or not for fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"has_node_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.has_node_feature","page":"FeaturedGraph","title":"GraphSignals.has_node_feature","text":"has_node_feature(::AbstractFeaturedGraph)\n\nCheck if node_feature is available or not for fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"has_edge_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.has_edge_feature","page":"FeaturedGraph","title":"GraphSignals.has_edge_feature","text":"has_edge_feature(::AbstractFeaturedGraph)\n\nCheck if edge_feature is available or not for fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"has_global_feature","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.has_global_feature","page":"FeaturedGraph","title":"GraphSignals.has_global_feature","text":"has_global_feature(::AbstractFeaturedGraph)\n\nCheck if global_feature is available or not for fg.\n\nArguments\n\nfg::AbstractFeaturedGraph: A concrete object of AbstractFeaturedGraph type.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/#Graph-properties","page":"FeaturedGraph","title":"Graph properties","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"FeaturedGraph is itself a graph, so we can query some graph properties from a FeaturedGraph.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> nv(fg)\n3\n\njulia> ne(fg)\n3\n\njulia> is_directed(fg)\nfalse","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Users can query number of vertex and number of edge by nv and ne, respectively. is_directed checks if the underlying graph is a directed graph or not.","category":"page"},{"location":"manual/featuredgraph/#Graph-related-APIs","page":"FeaturedGraph","title":"Graph-related APIs","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"nv","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"ne","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"is_directed","category":"page"},{"location":"manual/featuredgraph/#Pass-FeaturedGraph-to-CUDA","page":"FeaturedGraph","title":"Pass FeaturedGraph to CUDA","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Passing a FeaturedGraph to CUDA is easy. Just pipe a FeaturedGraph object to gpu provided by Flux.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> using Flux\n\njulia> fg = fg |> gpu\nFeaturedGraph(\n\tUndirected graph with (#V=3, #E=3) in adjacency matrix,\n\tNode feature:\tℝ^5 <CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}>,\n)","category":"page"},{"location":"manual/featuredgraph/#Linear-algebra-for-FeaturedGraph","page":"FeaturedGraph","title":"Linear algebra for FeaturedGraph","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"FeaturedGraph supports the calculation of graph Laplacian matrix in inplace manner.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"julia> fg = FeaturedGraph(adj, nf=rand(5, 3))\nFeaturedGraph(\n\tUndirected graph with (#V=3, #E=3) in adjacency matrix,\n\tNode feature:\tℝ^5 <Matrix{Float64}>,\n)\n\njulia> laplacian_matrix!(fg)\nFeaturedGraph(\n\tUndirected graph with (#V=3, #E=3) in Laplacian matrix,\n\tNode feature:\tℝ^5 <Matrix{Float64}>,\n)\n\njulia> laplacian_matrix(fg)\n3×3 SparseArrays.SparseMatrixCSC{Int64, Int64} with 9 stored entries:\n -2   1   1\n  1  -2   1\n  1   1  -2","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"laplacian_matrix! mutates the adjacency matrix into a Laplacian matrix in a FeaturedGraph object and the Laplacian matrix can be fetched by laplacian_matrix. The Laplacian matrix is cached in a FeaturedGraph object and can be passed to a graph neural network model for training or inference. This way reduces the calculation overhead for Laplacian matrix during the training process.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"FeaturedGraph supports not only Laplacian matrix, but also normalized Laplacian matrix and scaled Laplacian matrix calculation.","category":"page"},{"location":"manual/featuredgraph/#Inplaced-linear-algebraic-APIs","page":"FeaturedGraph","title":"Inplaced linear algebraic APIs","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"laplacian_matrix!","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"normalized_laplacian!","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"scaled_laplacian!","category":"page"},{"location":"manual/featuredgraph/#Linear-algebraic-APIs","page":"FeaturedGraph","title":"Linear algebraic APIs","text":"","category":"section"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"Non-inplaced APIs returns a vector or a matrix directly.","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"adjacency_matrix","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"degrees","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"degree_matrix","category":"page"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"laplacian_matrix","category":"page"},{"location":"manual/featuredgraph/#Graphs.LinAlg.laplacian_matrix","page":"FeaturedGraph","title":"Graphs.LinAlg.laplacian_matrix","text":"laplacian_matrix(g, [T=eltype(g)]; dir=:out)\n\nLaplacian matrix of graph g, defined as\n\nD - A\n\nwhere D is degree matrix and A is adjacency matrix from g.\n\nArguments\n\ng: Should be a adjacency matrix, FeaturedGraph, SimpleGraph, SimpleDiGraph (from Graphs)   or SimpleWeightedGraph, SimpleWeightedDiGraph (from SimpleWeightedGraphs).\nT: The element type of result degree vector. The default type is the element type of g.\ndir::Symbol: The way to calculate degree of a graph g regards its directions.   Should be :in, :out, or :both.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"normalized_laplacian","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.normalized_laplacian","page":"FeaturedGraph","title":"GraphSignals.normalized_laplacian","text":"normalized_laplacian(g, [T=float(eltype(g))]; dir=:both, selfloop=false)\n\nNormalized Laplacian matrix of graph g, defined as\n\nI - D^-frac12 tildeA D^-frac12\n\nwhere D is degree matrix and tildeA is adjacency matrix w/o self loop from g.\n\nArguments\n\ng: Should be a adjacency matrix, FeaturedGraph, SimpleGraph, SimpleDiGraph (from Graphs)   or SimpleWeightedGraph, SimpleWeightedDiGraph (from SimpleWeightedGraphs).\nT: The element type of result degree vector. The default type is the element type of g.\ndir::Symbol: The way to calculate degree of a graph g regards its directions.   Should be :in, :out, or :both.\nselfloop::Bool: Adding self loop to tildeA or not.\n\n\n\n\n\n","category":"function"},{"location":"manual/featuredgraph/","page":"FeaturedGraph","title":"FeaturedGraph","text":"scaled_laplacian","category":"page"},{"location":"manual/featuredgraph/#GraphSignals.scaled_laplacian","page":"FeaturedGraph","title":"GraphSignals.scaled_laplacian","text":"scaled_laplacian(g, [T=float(eltype(g))])\n\nScaled Laplacien matrix of graph g, defined as\n\nhatL = frac2lambda_max tildeL - I\n\nwhere tildeL is the normalized Laplacian matrix.\n\nArguments\n\ng: Should be a adjacency matrix, FeaturedGraph, SimpleGraph, SimpleDiGraph (from Graphs)   or SimpleWeightedGraph, SimpleWeightedDiGraph (from SimpleWeightedGraphs).\nT: The element type of result degree vector. The default type is the element type of g.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GraphSignals","category":"page"},{"location":"#GraphSignals","page":"Home","title":"GraphSignals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"GraphSignals is aim to provide a general data structure, which is composed of a graph and graph signals, in order to support a graph neural network library, specifically, GeometricFlux.jl. The concept of graph is used ubiquitously in several fields, including computer science, social science, biological science and neural science. GraphSignals provides graph signals attached to a graph as a whole for training or inference a graph neural network. Some characteristics of this package are listed:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Graph signals can be node features, edge features or global features, which are all general arrays.\nGraph Laplacian and related matrices are supported to calculated from a general data structure.\nSupport graph representations from JuliaGraphs.","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"FeaturedGraph supports various graph representations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports graph in adjacency matrix.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> adjm = [0 1 1 1;\n               1 0 1 0;\n               1 1 0 1;\n               1 0 1 0];\n\njulia> fg = FeaturedGraph(adjm)\nFeaturedGraph(\n\tUndirected graph with (#V=4, #E=5) in adjacency matrix,\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"It also supports graph in adjacency list.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> adjl = [\n               [2, 3, 4],\n               [1, 3],\n               [1, 2, 4],\n               [1, 3]\n               ];\n\njulia> fg = FeaturedGraph(adjl)\nFeaturedGraph(\n\tUndirected graph with (#V=4, #E=5) in adjacency matrix,\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports SimpleGraph from LightGraphs and convert adjacency matrix into a Laplacian matrix as well.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using LightGraphs\n\njulia> N = 4\n4\n\njulia> ug = SimpleGraph(N)\n{4, 0} undirected simple Int64 graph\n\njulia> add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4);\n\njulia> add_edge!(ug, 2, 3); add_edge!(ug, 3, 4);\n\njulia> fg = FeaturedGraph(ug)\nFeaturedGraph(\n\tUndirected graph with (#V=4, #E=5) in adjacency matrix,\n)\n\njulia> laplacian_matrix!(fg)\nFeaturedGraph(\n\tUndirected graph with (#V=4, #E=5) in Laplacian matrix,\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Features can be attached to it.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> N = 4\n4\n\njulia> E = 5\n5\n\njulia> nf = rand(3, N);\n\njulia> ef = rand(5, E);\n\njulia> gf = rand(7);\n\njulia> fg = FeaturedGraph(ug, nf=nf, ef=ef, gf=gf)\nFeaturedGraph(\n\tUndirected graph with (#V=4, #E=5) in adjacency matrix,\n\tNode feature:\tℝ^3 <Matrix{Float64}>,\n\tEdge feature:\tℝ^5 <Matrix{Float64}>,\n\tGlobal feature:\tℝ^7 <Vector{Float64}>,\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"If there are mismatched node features attached to it, a DimensionMismatch is throw out and hint user.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> nf = rand(3, 7);\n\njulia> fg = FeaturedGraph(ug, nf=nf)\nERROR: DimensionMismatch(\"number of nodes must match between graph (4) and node features (7)\")","category":"page"},{"location":"manual/sparsegraph/#Sparse-graph-Strucutre","page":"Sparse graph strucutre","title":"Sparse graph Strucutre","text":"","category":"section"},{"location":"manual/sparsegraph/#The-need-of-graph-structure","page":"Sparse graph strucutre","title":"The need of graph structure","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"Graph convolution can be classified into spectral-based graph convolution and spatial-based graph convolution. Spectral-based graph convolution relys on the algebaric operations, including +, -, *, which are applied to features with graph structure. Spatial-based graph convolution relys on the indexing operations, since spatial-based graph convolution always indexes the neighbors of vertex. A graph structure can be use under two view point of a part of algebaric operations or an indexing structure.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"Message-passing neural network requires to access neighbor information for each vertex. Messages are passed from a vertex's neighbors to itself. A efficient indexing data structure is required to access incident edges or neighbor vertices from a specific vertex.","category":"page"},{"location":"manual/sparsegraph/#SparseGraph","page":"Sparse graph strucutre","title":"SparseGraph","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"SparseGraph is implemented with sparse matrix. It is built on top of built-in sparse matrix, SparseMatrixCSC. SparseMatrixCSC can be used as a regular matrix and performs algebaric operations with matrix or vectors.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"To benefit message-passing scheme, making a graph structure as an indexing structure is important. A well-designed indexing structure is made to leverage the sparse format of SparseMatrixCSC, which is in CSC format. CSC format stores sparse matrix in a highly compressed manner. Comparing to traditional COO format, CSC format compresses the column indices into column pointers. All values are stored in single vector. If we want to index the sparse matrix A, the row indices can be fetched by rowvals[colptr[j]:(colptr[j+1]-1)] and the non-zero values can be indexed by nzvals[colptr[j]:(colptr[j+1]-1)]. The edge indices are designed in the same manner edges[colptr[j]:(colptr[j+1]-1)]. This way matches the need of indexing neighbors of vertex. This makes neighbor indices or values close together. It takes O(1) to get negihbor indices, instead of searching neighbor in O(N). Thus, SparseGraph takes both advantages of both algebaric operations and indexing operations.","category":"page"},{"location":"manual/sparsegraph/#Create-SparseGraph","page":"Sparse graph strucutre","title":"Create SparseGraph","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"SparseGraph accepts adjacency matrix, adjacency list, and almost all graphs defined in JuliaGraphs.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> using GraphSignals, LightGraphs\n\njulia> ug = SimpleGraph(4)\n{4, 0} undirected simple Int64 graph\n\njulia> add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4);\n\njulia> add_edge!(ug, 2, 3); add_edge!(ug, 3, 4);\n\njulia> sg = SparseGraph(ug)\nSparseGraph(#V=4, #E=5)","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"The indexed adjacency list is a list of list strucutre. The inner list consists of a series of tuples containing a vertex index and a edge index, respectively.","category":"page"},{"location":"manual/sparsegraph/#Operate-SparseGraph-as-graph","page":"Sparse graph strucutre","title":"Operate SparseGraph as graph","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"It supports basic graph APIs for querying graph information, including number of vertices nv and number of edges ne.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> is_directed(sg)\nfalse\n\njulia> nv(sg)\n4\n\njulia> ne(sg)\n5\n\njulia> eltype(sg)\nInt64\n\njulia> has_vertex(sg, 3)\ntrue\n\njulia> has_edge(sg, 1, 2)\ntrue","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"We can compare two graph structure if they are equivalent or not.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> adjm = [0 1 1 1; 1 0 1 0; 1 1 0 1; 1 0 1 0]\n4×4 Matrix{Int64}:\n 0  1  1  1\n 1  0  1  0\n 1  1  0  1\n 1  0  1  0\n\njulia> sg2 = SparseGraph(adjm, false)\nSparseGraph(#V=4, #E=5)\n\njulia> sg == sg2\ntrue","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"We can also iterate over edges.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> for (i, e) in edges(sg)\n           println(\"edge index: \", i, \", edge: \", e)\n       end\nedge index: 1, edge: (2, 1)\nedge index: 2, edge: (3, 1)\nedge index: 3, edge: (3, 2)\nedge index: 4, edge: (4, 1)\nedge index: 5, edge: (4, 3)","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"Edge index is the index for each edge. It is used to index edge features.","category":"page"},{"location":"manual/sparsegraph/#Indexing-operations","page":"Sparse graph strucutre","title":"Indexing operations","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"To get neighbors of a specified vertex, neighbors is used by passing a SparseGraph object and a vertex index. A vector of neighbor vertex index is returned.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> neighbors(sg, 1)\n3-element view(::Vector{Int64}, 1:3) with eltype Int64:\n 2\n 3\n 4","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"To get incident edges of a specified vertex, incident_edges can be used and it will return edge indices.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> incident_edges(sg, 1)\n3-element view(::Vector{Int64}, 1:3) with eltype Int64:\n 1\n 2\n 4","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"An edge index can be fetched by querying an edge, for example, edge (1, 2) and edge (2, 1) refers to the same edge with index 1.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> edge_index(sg, 1, 2)\n1\n\njulia> edge_index(sg, 2, 1)\n1","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"One can have the opportunity to index the underlying sparse matrix.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> sg[1, 2]\n1\n\njulia> sg[2, 1]\n1","category":"page"},{"location":"manual/sparsegraph/#Aggregate-over-neighbors","page":"Sparse graph strucutre","title":"Aggregate over neighbors","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"In message-passing scheme, it is always to aggregate node features or edge feature from neighbors. For convention, edge_scatter and neighbor_scatter are used to apply aggregate operations over edge features or neighbor vertex features. The actual aggregation is supported by scatter operations.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> nf = rand(10, 4);\n\njulia> neighbor_scatter(+, nf, sg)\n10×4 Matrix{Float64}:\n 1.54937   1.03974   1.72926  1.03974\n 1.38554   0.775991  1.34106  0.775991\n 1.13192   0.424888  1.34657  0.424888\n 2.23452   1.63226   2.436    1.63226\n 0.815662  0.718865  1.25237  0.718865\n 2.35763   1.42174   2.26442  1.42174\n 1.94051   1.44812   1.71694  1.44812\n 1.83641   1.89104   1.80857  1.89104\n 2.43027   1.92217   2.37003  1.92217\n 1.58177   1.16149   1.87467  1.16149","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"For example, neighbor_scatter aggregates node features nf via neighbors in graph sg by + operation.","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"julia> ef = rand(9, 5);\n\njulia> edge_scatter(+, ef, sg)\n9×4 Matrix{Float64}:\n 2.22577  0.967172  1.92781   1.92628\n 1.4842   1.20605   2.30014   0.849819\n 2.20728  1.01527   0.899094  1.35062\n 1.09119  0.589925  1.62597   1.51175\n 1.42288  1.63764   1.23445   0.693258\n 1.57561  0.926591  1.72599   0.690108\n 1.68402  0.544808  1.58687   1.70676\n 1.10908  1.0898    1.05256   0.508157\n 2.33764  1.26419   1.87927   1.11151","category":"page"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"Or, edge_scatter aggregates edge features ef via incident edges in graph sg by + operation.","category":"page"},{"location":"manual/sparsegraph/#SparseGraph-APIs","page":"Sparse graph strucutre","title":"SparseGraph APIs","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"SparseGraph\nGraphSignals.neighbors\nincident_edges\nneighbor_scatter\nedge_scatter","category":"page"},{"location":"manual/sparsegraph/#GraphSignals.SparseGraph","page":"Sparse graph strucutre","title":"GraphSignals.SparseGraph","text":"SparseGraph(A, directed, [T])\n\nA sparse graph structure represents by sparse matrix. A directed graph is represented by a sparse matrix, of which column index as source node index and row index as sink node index.\n\nArguments\n\nA: Adjacency matrix.\ndirected: If this is a directed graph or not.\nT: Element type for SparseGraph.\n\n\n\n\n\n","category":"type"},{"location":"manual/sparsegraph/#Graphs.neighbors","page":"Sparse graph strucutre","title":"Graphs.neighbors","text":"neighbors(sg, i)\n\nReturn the neighbors of vertex i in sparse graph sg.\n\nArguments\n\nsg::SparseGraph: sparse graph to query.\ni: vertex index.\n\n\n\n\n\n","category":"function"},{"location":"manual/sparsegraph/#GraphSignals.incident_edges","page":"Sparse graph strucutre","title":"GraphSignals.incident_edges","text":"incident_edges(sg, i)\n\nReturn the edges incident to vertex i in sparse graph sg.\n\nArguments\n\nsg::SparseGraph: sparse graph to query.\ni: vertex index.\n\n\n\n\n\n","category":"function"},{"location":"manual/sparsegraph/#Internals","page":"Sparse graph strucutre","title":"Internals","text":"","category":"section"},{"location":"manual/sparsegraph/","page":"Sparse graph strucutre","title":"Sparse graph strucutre","text":"In the design of SparseGraph, it resolve the problem of indexing edge features. For a graph, edge is represented in (i, j) and edge features are considered as a matrix ef with edge number in its column. The problem is to unifiedly fetch corresponding edge feature ef[:, k] for edge (i, j) over directed and undirected graph. To resolve this issue, edge index is set to be the unique index for each edge. Further, aggregate_index is designed to generate indices for aggregating from neighbor nodes or incident edges. Conclusively, it provides the core operations needed in message-passing scheme.","category":"page"}]
}
