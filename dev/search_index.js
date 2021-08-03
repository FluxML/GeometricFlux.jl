var documenterSearchIndex = {"docs":
[{"location":"basics/passgraph/#Graph-passing-1","page":"Graph passing","title":"Graph passing","text":"","category":"section"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"Graph is an input data structure for graph neural network. Passing a graph into a GNN layer can have different behaviors. If the graph remains fixed across samples, that is, all samples utilize the same graph structure, a static graph is used. Graphs can be carried within FeaturedGraph to provide variable graphs to GNN layer. Users have the flexibility to pick an adequate approach for their own needs.","category":"page"},{"location":"basics/passgraph/#Static-graph-1","page":"Graph passing","title":"Static graph","text":"","category":"section"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"A static graph is used to reduce redundant computation during passing through layers. A static graph can be set in graph convolutional layers such that this graph is shared for computations across those layers. An adjacency matrix adj_mat is given to represent a graph and is provided to a graph convolutional layer as follows:","category":"page"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"GCNConv(adj_mat, feat=>h1, relu)","category":"page"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"Simple(Di)Graph, SimpleWeighted(Di)Graph or Meta(Di)Graph provided by the packages LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are valid arguments for passing as a static graph to this layer. An adjacency list is also accepted in the type of Vector{Vector} is also accepted.","category":"page"},{"location":"basics/passgraph/#Variable-graph-1","page":"Graph passing","title":"Variable graph","text":"","category":"section"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"Variable graphs are supported through FeaturedGraph, which contains both the graph information and the features. Each FeaturedGraph can contain a different graph structure and its features. Data of FeaturedGraph are fed directly to graph convolutional layer or graph neural network to let each feature be learned on different graph structures. A adjacency matrix adj_mat is given to construct a FeaturedGraph as follows:","category":"page"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"FeaturedGraph(adj_mat, features)","category":"page"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"Simple(Di)Graph, SimpleWeighted(Di)Graph or Meta(Di)Graph provided by the packages LightGraphs, SimpleWeightedGraphs and MetaGraphs, respectively, are acceptable for constructing a FeaturedGraph. An adjacency list is also accepted, too.","category":"page"},{"location":"basics/passgraph/#Cached-graph-in-layers-1","page":"Graph passing","title":"Cached graph in layers","text":"","category":"section"},{"location":"basics/passgraph/#","page":"Graph passing","title":"Graph passing","text":"While a variable graph is given by FeaturedGraph, a GNN layer doesn't need a static graph anymore. A cache mechanism is designed to cache static graph to reduce computation. A cached graph is retrieved from layer and computation is then performed. For each time, it will assign current computed graph back to layer.","category":"page"},{"location":"start/#Get-started-1","page":"Get started","title":"Get started","text":"","category":"section"},{"location":"manual/models/#Models-1","page":"Models","title":"Models","text":"","category":"section"},{"location":"manual/models/#Autoencoders-1","page":"Models","title":"Autoencoders","text":"","category":"section"},{"location":"manual/models/#Graph-Autoencoder-1","page":"Models","title":"Graph Autoencoder","text":"","category":"section"},{"location":"manual/models/#","page":"Models","title":"Models","text":"Z = enc(X A) \nhatA = sigma (ZZ^T)","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"where A denotes the adjacency matrix.","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"GAE","category":"page"},{"location":"manual/models/#GeometricFlux.GAE","page":"Models","title":"GeometricFlux.GAE","text":"GAE(enc[, σ])\n\nGraph autoencoder.\n\nArguments\n\nenc: encoder. It can be any graph convolutional layer.\n\nEncoder is specified by user and decoder will be InnerProductDecoder layer.\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#","page":"Models","title":"Models","text":"Reference: Variational Graph Auto-Encoders","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"","category":"page"},{"location":"manual/models/#Variational-Graph-Autoencoder-1","page":"Models","title":"Variational Graph Autoencoder","text":"","category":"section"},{"location":"manual/models/#","page":"Models","title":"Models","text":"H = enc(X A) \nZ_mu Z_logσ = GCN_mu(H A) GCN_sigma(H A) \nhatA = sigma (ZZ^T)","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"where A denotes the adjacency matrix, X denotes node features.","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"VGAE","category":"page"},{"location":"manual/models/#GeometricFlux.VGAE","page":"Models","title":"GeometricFlux.VGAE","text":"VGAE(enc[, σ])\n\nVariational graph autoencoder.\n\nArguments\n\nenc: encoder. It can be any graph convolutional layer.\n\nEncoder is specified by user and decoder will be InnerProductDecoder layer.\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#","page":"Models","title":"Models","text":"Reference: Variational Graph Auto-Encoders","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"","category":"page"},{"location":"manual/models/#Special-Layers-1","page":"Models","title":"Special Layers","text":"","category":"section"},{"location":"manual/models/#Inner-product-Decoder-1","page":"Models","title":"Inner-product Decoder","text":"","category":"section"},{"location":"manual/models/#","page":"Models","title":"Models","text":"hatA = sigma (ZZ^T)","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"where Z denotes the input matrix from encoder.","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"InnerProductDecoder","category":"page"},{"location":"manual/models/#GeometricFlux.InnerProductDecoder","page":"Models","title":"GeometricFlux.InnerProductDecoder","text":"InnerProductDecoder(σ)\n\nInner-product decoder layer.\n\nArguments\n\nσ: activation function.\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#","page":"Models","title":"Models","text":"Reference: Variational Graph Auto-Encoders","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"","category":"page"},{"location":"manual/models/#Variational-Encoder-1","page":"Models","title":"Variational Encoder","text":"","category":"section"},{"location":"manual/models/#","page":"Models","title":"Models","text":"H = enc(X A) \nZ_mu Z_logσ = GCN_mu(H A) GCN_sigma(H A)","category":"page"},{"location":"manual/models/#","page":"Models","title":"Models","text":"VariationalEncoder","category":"page"},{"location":"manual/models/#GeometricFlux.VariationalEncoder","page":"Models","title":"GeometricFlux.VariationalEncoder","text":"VariationalEncoder(nn, h_dim, z_dim)\n\nVariational encoder layer.\n\nArguments\n\nnn: neural network. It can be any graph convolutional layer.\nh_dim: dimension of hidden layer. This should fit the output dimension of nn.\nz_dim: dimension of latent variable layer. This will be parametrized into μ and logσ.\n\nEncoder can be any graph convolutional layer.\n\n\n\n\n\n","category":"type"},{"location":"manual/models/#","page":"Models","title":"Models","text":"Reference: Variational Graph Auto-Encoders","category":"page"},{"location":"abstractions/msgpass/#Message-passing-scheme-1","page":"Message passing scheme","title":"Message passing scheme","text":"","category":"section"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"Message passing scheme is a popular GNN scheme in many frameworks. It adapts the property of connectivity of neighbors and form a general approach for spatial graph convolutional neural network. It comprises two user-defined functions and one aggregate function. A message function is defined to process information from edge states and node states from neighbors and itself. Messages from each node are obtained and aggregated by aggregate function to provide node-level information for update function. Update function takes current node state and aggregated message and gives a new node state.","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"Message passing scheme is realized into a abstract type MessagePassing. Any subtype of MessagePassing is a message passing layer which utilize default message and update functions:","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"message(mp, x_i, x_j, e_ij) = x_j\nupdate(mp, m, x) = m","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"mp denotes a message passing layer. message accepts node state x_i for node i and its neighbor state x_j for node j, as well as corresponding edge state e_ij for edge (i,j). The default message function gives all the neighbor state x_j for neighbor of node i. update takes aggregated message m and current node state x, and then outputs m.","category":"page"},{"location":"abstractions/msgpass/#Message-function-1","page":"Message passing scheme","title":"Message function","text":"","category":"section"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"A message function accepts feature vector representing node state x_i, feature vectors for neighbor state x_j and corresponding edge state e_ij. A vector is expected to output from message for message. User can override message for customized message passing layer to provide desired behavior.","category":"page"},{"location":"abstractions/msgpass/#Aggregate-messages-1","page":"Message passing scheme","title":"Aggregate messages","text":"","category":"section"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"Messages from message function are aggregated by an aggregate function. An aggregated message is passed to update function for node-level computation. An aggregate function is given by the following:","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"propagate(mp, fg::FeaturedGraph, aggr::Symbol=:add)","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"propagate function calls the whole message passing layer. fg acts as an input for message passing layer and aggr represents assignment of aggregate function to propagate function. :add represents an aggregate function of addition of all messages.","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"The following aggr are available aggregate functions:","category":"page"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":":add: sum over all messages :sub: negative of sum over all messages :mul: multiplication over all messages :div: inverse of multiplication over all messages :max: the maximum of all messages :min: the minimum of all messages :mean: the average of all messages","category":"page"},{"location":"abstractions/msgpass/#Update-function-1","page":"Message passing scheme","title":"Update function","text":"","category":"section"},{"location":"abstractions/msgpass/#","page":"Message passing scheme","title":"Message passing scheme","text":"An update function takes aggregated message m and current node state x as arguments. An output vector is expected to be the new node state for next layer. User can override update for customized message passing layer to provide desired behavior.","category":"page"},{"location":"manual/pool/#Pooling-layers-1","page":"Pooling Layers","title":"Pooling layers","text":"","category":"section"},{"location":"manual/pool/#","page":"Pooling Layers","title":"Pooling Layers","text":"GlobalPool","category":"page"},{"location":"manual/pool/#GeometricFlux.GlobalPool","page":"Pooling Layers","title":"GeometricFlux.GlobalPool","text":"GlobalPool(aggr, dim...)\n\nGlobal pooling layer.\n\nIt pools all features with aggr operation.\n\nArguments\n\naggr: An aggregate function applied to pool all features.\n\n\n\n\n\n","category":"type"},{"location":"manual/pool/#","page":"Pooling Layers","title":"Pooling Layers","text":"LocalPool","category":"page"},{"location":"manual/pool/#GeometricFlux.LocalPool","page":"Pooling Layers","title":"GeometricFlux.LocalPool","text":"LocalPool(aggr, cluster)\n\nLocal pooling layer.\n\nIt pools features with aggr operation accroding to cluster. It is implemented with scatter operation.\n\nArguments\n\naggr: An aggregate function applied to pool all features.\ncluster: An index structure which indicates what features to aggregate with.\n\n\n\n\n\n","category":"type"},{"location":"manual/pool/#","page":"Pooling Layers","title":"Pooling Layers","text":"TopKPool","category":"page"},{"location":"manual/pool/#GeometricFlux.TopKPool","page":"Pooling Layers","title":"GeometricFlux.TopKPool","text":"TopKPool(adj, k, in_channel)\n\nTop-k pooling layer.\n\nArguments\n\nadj: Adjacency matrix  of a graph.\nk: Top-k nodes are selected to pool together.\nin_channel: The dimension of input channel.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#Convolution-Layers-1","page":"Convolutional Layers","title":"Convolution Layers","text":"","category":"section"},{"location":"manual/conv/#Graph-Convolutional-Layer-1","page":"Convolutional Layers","title":"Graph Convolutional Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"X = sigma(hatD^-12 hatA hatD^-12 X Theta)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where hatA = A + I, A denotes the adjacency matrix, and hatD = hatd_ij = sum_j=0 hata_ij is degree matrix.","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"GCNConv","category":"page"},{"location":"manual/conv/#GeometricFlux.GCNConv","page":"Convolutional Layers","title":"GeometricFlux.GCNConv","text":"GCNConv([fg,] in => out, σ=identity; bias=true, init=glorot_uniform)\n\nGraph convolutional layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nin: The dimension of input features.\nout: The dimension of output features.\nσ: Activation function.\nbias: Add learnable bias.\ninit: Weights' initializer.\n\nThe input to the layer is a node feature array X  of size (num_features, num_nodes).\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Semi-supervised Classification with Graph Convolutional Networks","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Chebyshev-Spectral-Graph-Convolutional-Layer-1","page":"Convolutional Layers","title":"Chebyshev Spectral Graph Convolutional Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"X = sum^K-1_k=0 Z^(k) Theta^(k)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where Z^(k) is the k-th term of Chebyshev polynomials, and can be calculated by the following recursive form:","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Z^(0) = X \nZ^(1) = hatL X \nZ^(k) = 2 hatL Z^(k-1) - Z^(k-2)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"and hatL = frac2lambda_max L - I.","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"ChebConv","category":"page"},{"location":"manual/conv/#GeometricFlux.ChebConv","page":"Convolutional Layers","title":"GeometricFlux.ChebConv","text":"ChebConv([fg,] in=>out, k; bias=true, init=glorot_uniform)\n\nChebyshev spectral graph convolutional layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nin: The dimension of input features.\nout: The dimension of output features.\nk: The order of Chebyshev polynomial.\nbias: Add learnable bias.\ninit: Weights' initializer.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Graph-Neural-Network-Layer-1","page":"Convolutional Layers","title":"Graph Neural Network Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"textbfx_i = sigma (Theta_1 textbfx_i + sum_j in mathcalN(i) Theta_2 textbfx_j)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"GraphConv","category":"page"},{"location":"manual/conv/#GeometricFlux.GraphConv","page":"Convolutional Layers","title":"GeometricFlux.GraphConv","text":"GraphConv([fg,] in => out, σ=identity, aggr=+; bias=true, init=glorot_uniform)\n\nGraph neural network layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nin: The dimension of input features.\nout: The dimension of output features.\nσ: Activation function.\naggr: An aggregate function applied to the result of message function. +, -,\n\n*, /, max, min and mean are available.\n\nbias: Add learnable bias.\ninit: Weights' initializer.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Graph-Attentional-Layer-1","page":"Convolutional Layers","title":"Graph Attentional Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"textbfx_i = alpha_ii Theta textbfx_i + sum_j in mathcalN(i) alpha_ij Theta textbfx_j","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where the attention coefficient alpha_ij can be calculated from","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"alpha_ij = fracexp(LeakyReLU(textbfa^T Theta textbfx_i  Theta textbfx_j))sum_k in mathcalN(i) cup i exp(LeakyReLU(textbfa^T Theta textbfx_i  Theta textbfx_k))","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"GATConv","category":"page"},{"location":"manual/conv/#GeometricFlux.GATConv","page":"Convolutional Layers","title":"GeometricFlux.GATConv","text":"GATConv([fg,] in => out;\n        heads=1,\n        concat=true,\n        init=glorot_uniform    \n        bias=true, \n        negative_slope=0.2)\n\nGraph attentional layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nin: The dimension of input features.\nout: The dimension of output features.\nbias::Bool: Keyword argument, whether to learn the additive bias.\nheads: Number attention heads \nconcat: Concatenate layer output or not. If not, layer output is averaged.\nnegative_slope::Real: Keyword argument, the parameter of LeakyReLU.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Graph Attention Networks","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Gated-Graph-Convolution-Layer-1","page":"Convolutional Layers","title":"Gated Graph Convolution Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"textbfh^(0)_i = textbfx_i  textbf0 \ntextbfh^(l)_i = GRU(textbfh^(l-1)_i sum_j in mathcalN(i) Theta textbfh^(l-1)_j)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where textbfh^(l)_i denotes the l-th hidden variables passing through GRU. The dimension of input textbfx_i needs to be less or equal to out.","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"GatedGraphConv","category":"page"},{"location":"manual/conv/#GeometricFlux.GatedGraphConv","page":"Convolutional Layers","title":"GeometricFlux.GatedGraphConv","text":"GatedGraphConv([fg,] out, num_layers; aggr=+, init=glorot_uniform)\n\nGated graph convolution layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nout: The dimension of output features.\nnum_layers: The number of gated recurrent unit.\naggr: An aggregate function applied to the result of message function. +, -,\n\n*, /, max, min and mean are available.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Gated Graph Sequence Neural Networks","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Edge-Convolutional-Layer-1","page":"Convolutional Layers","title":"Edge Convolutional Layer","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"textbfx_i = sum_j in mathcalN(i) f_Theta(textbfx_i  textbfx_j - textbfx_i)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where f_Theta denotes a neural network parametrized by Theta, i.e., a MLP.","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"EdgeConv","category":"page"},{"location":"manual/conv/#GeometricFlux.EdgeConv","page":"Convolutional Layers","title":"GeometricFlux.EdgeConv","text":"EdgeConv([fg,] nn; aggr=max)\n\nEdge convolutional layer.\n\nArguments\n\nfg: Optionally pass a FeaturedGraph. \nnn: A neural network (e.g. a Dense layer or a MLP). \naggr: An aggregate function applied to the result of message function. +, max and mean are available.\n\n\n\n\n\n","category":"type"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: Dynamic Graph CNN for Learning on Point Clouds","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"","category":"page"},{"location":"manual/conv/#Graph-Isomorphism-Network-1","page":"Convolutional Layers","title":"Graph Isomorphism Network","text":"","category":"section"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"textbfx_i = f_Thetaleft((1 + varepsilon) dot textbfx_i + sum_j in mathcalN(i) textbfx_j right)","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"where f_Theta denotes a neural network parametrized by Theta, i.e., a MLP.","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"GINConv","category":"page"},{"location":"manual/conv/#","page":"Convolutional Layers","title":"Convolutional Layers","text":"Reference: How Powerful are Graph Neural Networks?","category":"page"},{"location":"manual/linalg/#Linear-Algebra-1","page":"Linear Algebra","title":"Linear Algebra","text":"","category":"section"},{"location":"manual/linalg/#","page":"Linear Algebra","title":"Linear Algebra","text":"GraphSignals.degrees","category":"page"},{"location":"manual/linalg/#","page":"Linear Algebra","title":"Linear Algebra","text":"GraphSignals.degree_matrix","category":"page"},{"location":"manual/linalg/#","page":"Linear Algebra","title":"Linear Algebra","text":"GraphSignals.inv_sqrt_degree_matrix","category":"page"},{"location":"manual/linalg/#","page":"Linear Algebra","title":"Linear Algebra","text":"GraphSignals.laplacian_matrix","category":"page"},{"location":"manual/linalg/#","page":"Linear Algebra","title":"Linear Algebra","text":"GraphSignals.normalized_laplacian","category":"page"},{"location":"cooperate/#Cooperate-with-Flux-layers-1","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"","category":"section"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"GeometricFlux is designed to be compatible with Flux layers. Flux layers usually have array input and array output. In GeometricFlux, there are are two APIs which allow different input/output types for GNN layers. For example, GCNConv layer provides the following two APIs:","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"(g::GCNConv)(X::AbstractMatrix) -> AbstractMatrix\n(g::GCNConv)(fg::FeaturedGraph) -> FeaturedGraph","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"If your feed a GCNConv layer with a Matrix, it will return you a Matrix. If you feed a GCNConv layer with a FeaturedGraph, it will return you a FeaturedGraph. These APIs ensure the consistency between input and output types. What you feed is what you get. So, the API for array type is compatible directly with other Flux layers. However, the API for FeaturedGraph is not compatible directly.","category":"page"},{"location":"cooperate/#Fetching-features-from-FeaturedGraph-and-output-compatible-result-with-Flux-layers-1","page":"Cooperate with Flux layers","title":"Fetching features from FeaturedGraph and output compatible result with Flux layers","text":"","category":"section"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"With a layer outputs a FeaturedGraph, it is not compatible with Flux layers. Since Flux layers need single feature in array form as input, node features, edge features and global features can be selected by using FeaturedGraph APIs: node_feature, edge_feature or global_feature, respectively.","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"model = Flux.Chain(\n    GCNConv(1024=>256, relu),\n    node_feature,  # or edge_feature or global_feature\n    softmax\n)","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"In a multitask learning scenario, multiple outputs are required. A branching selection of features can be made as follows:","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"model = Flux.Chain(\n    GCNConv(1024=>256, relu),\n    x -> (node_feature(x), global_feature(x)),\n    (nf, gf) -> (softmax(nf), identity.(gf))\n)","category":"page"},{"location":"cooperate/#Branching-different-features-through-different-layers-1","page":"Cooperate with Flux layers","title":"Branching different features through different layers","text":"","category":"section"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"(Image: )","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"A bypass_graph function is designed for passing each feature through different layers from a FeaturedGraph. An example is given as follow:","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"Flux.Chain(\n    ...\n    bypass_graph(\n        nf_func=GCNConv(1024=>256, relu),\n        ef_func=Dense(1024, 256, relu),\n        gf_func=identity,\n    ),\n    ...\n)","category":"page"},{"location":"cooperate/#","page":"Cooperate with Flux layers","title":"Cooperate with Flux layers","text":"bypass_graph will pass node feature to a GCNConv layer and edge feature to a Dense layer. Meanwhile, a FeaturedGraph is decomposed and keep the graph in FeaturedGraph to the downstream layers. A new FeaturedGraph is constructed with processed node feature, edge feature and global feature. bypass_graph acts as a layer which accepts a FeaturedGraph and output a FeaturedGraph. Thus, it by pass the graph in a FeaturedGraph but pass different features to different layers.","category":"page"},{"location":"basics/layers/#Building-graph-neural-networks-1","page":"Building layers","title":"Building graph neural networks","text":"","category":"section"},{"location":"basics/layers/#","page":"Building layers","title":"Building layers","text":"Building GNN is as simple as building neural network in Flux. The syntax here is the same as Flux. Chain is used to stack layers into a GNN. A simple example is shown here:","category":"page"},{"location":"basics/layers/#","page":"Building layers","title":"Building layers","text":"model = Chain(GCNConv(adj_mat, feat=>h1),\n              GCNConv(adj_mat, h1=>h2, relu))","category":"page"},{"location":"basics/layers/#","page":"Building layers","title":"Building layers","text":"GCNConv is used for layer construction for neural network. The first argument adj_mat is the representation of a graph in form of adjacency matrix. The feature dimension in first layer is mapped from feat to h1. In second layer, h1 is then mapped to h2. Default activation function is given as identity if it is not specified by users.","category":"page"},{"location":"basics/layers/#Customize-layers-1","page":"Building layers","title":"Customize layers","text":"","category":"section"},{"location":"basics/layers/#","page":"Building layers","title":"Building layers","text":"Customizing your own GNN layers are the same as customizing layers in Flux. You may want to reference Flux documentation.","category":"page"},{"location":"#GeometricFlux:-The-Geometric-Deep-Learning-Library-in-Julia-1","page":"Home","title":"GeometricFlux: The Geometric Deep Learning Library in Julia","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"GeometricFlux is a framework for geometric deep learning/machine learning.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"It extends Flux machine learning library for geometric deep learning.\nIt supports of CUDA GPU with CUDA.jl\nIt integrates with JuliaGraphs ecosystems.\nIt supports generic graph neural network architectures (i.g. message passing scheme and graph network block)\nIt contains built-in GNN benchmark datasets (provided by GraphMLDatasets)","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"] add GeometricFlux","category":"page"},{"location":"#Quick-start-1","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The basic graph convolutional network (GCN) is constructed as follow.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),\n              GCNConv(adj_mat, hidden=>target_catg),\n              softmax)","category":"page"},{"location":"#Load-dataset-1","page":"Home","title":"Load dataset","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Load cora dataset from GeometricFlux.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using JLD2\nusing SparseArrays\n\n@load joinpath(pkgdir(GeometricFlux), \"data/cora_features.jld2\") features\n@load joinpath(pkgdir(GeometricFlux), \"data/cora_labels.jld2\") labels\n@load joinpath(pkgdir(GeometricFlux), \"data/cora_graph.jld2\") g","category":"page"},{"location":"#Training/testing-data-1","page":"Home","title":"Training/testing data","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Data is stored in sparse array, thus, we have to convert it into normal array.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"train_X = Matrix{Float32}(features)\ntrain_y = Matrix{Float32}(labels)\nadj_mat = Matrix{Float32}(adjacency_matrix(g))","category":"page"},{"location":"#Loss-function-1","page":"Home","title":"Loss function","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"loss(x, y) = logitcrossentropy(model(x), y)\naccuracy(x, y) = mean(onecold(model(x)) .== onecold(y))","category":"page"},{"location":"#Training-1","page":"Home","title":"Training","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"ps = Flux.params(model)\ntrain_data = [(train_X, train_y)]\nopt = ADAM()\nevalcb() = @show(accuracy(train_X, train_y))\n\n@epochs epochs Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))","category":"page"},{"location":"#Logs-1","page":"Home","title":"Logs","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"[ Info: Epoch 1\naccuracy(train_X, train_y) = 0.11669128508124077\n[ Info: Epoch 2\naccuracy(train_X, train_y) = 0.19608567208271788\n[ Info: Epoch 3\naccuracy(train_X, train_y) = 0.3098227474150665\n[ Info: Epoch 4\naccuracy(train_X, train_y) = 0.387370753323486\n[ Info: Epoch 5\naccuracy(train_X, train_y) = 0.44645494830132937\n[ Info: Epoch 6\naccuracy(train_X, train_y) = 0.46824224519940916\n[ Info: Epoch 7\naccuracy(train_X, train_y) = 0.48892171344165436\n[ Info: Epoch 8\naccuracy(train_X, train_y) = 0.5025849335302807\n[ Info: Epoch 9\naccuracy(train_X, train_y) = 0.5151403249630724\n[ Info: Epoch 10\naccuracy(train_X, train_y) = 0.5291728212703102\n[ Info: Epoch 11\naccuracy(train_X, train_y) = 0.543205317577548\n[ Info: Epoch 12\naccuracy(train_X, train_y) = 0.5550221565731167\n[ Info: Epoch 13\naccuracy(train_X, train_y) = 0.5638847858197932\n[ Info: Epoch 14\naccuracy(train_X, train_y) = 0.5657311669128509\n[ Info: Epoch 15\naccuracy(train_X, train_y) = 0.5749630723781388\n[ Info: Epoch 16\naccuracy(train_X, train_y) = 0.5834564254062038\n[ Info: Epoch 17\naccuracy(train_X, train_y) = 0.5919497784342689\n[ Info: Epoch 18\naccuracy(train_X, train_y) = 0.5978581979320532\n[ Info: Epoch 19\naccuracy(train_X, train_y) = 0.6019202363367799\n[ Info: Epoch 20\naccuracy(train_X, train_y) = 0.6067208271787297","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Check examples/gcn.jl for details.","category":"page"},{"location":"abstractions/gn/#Graph-network-block-1","page":"Graph network block","title":"Graph network block","text":"","category":"section"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"Graph network (GN) is a more generic model for graph neural network. It describes an update order: edge, node and then global. There are three corresponding update functions for edge, node and then global, respectively. Three update functions return their default values as follow:","category":"page"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"update_edge(gn, e, vi, vj, u) = e\nupdate_vertex(gn, ē, vi, u) = vi\nupdate_global(gn, ē, v̄, u) = u","category":"page"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"Information propagation between different levels are achieved by aggregate functions. Three aggregate functions aggregate_neighbors, aggregate_edges and aggregate_vertices are defined to aggregate states.","category":"page"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"GN block is realized into a abstract type GraphNet. User can make a subtype of GraphNet to customize GN block. Thus, a GN block is defined as a layer in GNN. MessagePassing is a subtype of GraphNet.","category":"page"},{"location":"abstractions/gn/#Update-functions-1","page":"Graph network block","title":"Update functions","text":"","category":"section"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"update_edge acts as the first update function to apply to edge states. It takes edge state e, node i state vi, node j state vj and global state u. It is expected to return a feature vector for new edge state. update_vertex updates nodes state by taking aggregated edge state ē, node i state vi and global state u. It is expected to return a feature vector for new node state. update_global updates global state with aggregated information from edge and node. It takes aggregated edge state ē, aggregated node state v̄ and global state u. It is expected to return a feature vector for new global state. User can define their own behavior by overriding update functions.","category":"page"},{"location":"abstractions/gn/#Aggregate-functions-1","page":"Graph network block","title":"Aggregate functions","text":"","category":"section"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"An aggregate function aggregate_neighbors aggregates edge states for edges incident to some node i into node-level information. Aggregate function aggregate_edges aggregates all edge states into global-level information. The last aggregate function aggregate_vertices aggregates all vertex states into global-level information. It is available for assigning aggregate function by assigning aggregate operations to propagate function.","category":"page"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"propagate(gn, fg::FeaturedGraph, naggr=nothing, eaggr=nothing, vaggr=nothing)","category":"page"},{"location":"abstractions/gn/#","page":"Graph network block","title":"Graph network block","text":"naggr, eaggr and vaggr are arguments for aggregate_neighbors, aggregate_edges and aggregate_vertices, respectively. Available aggregate functions are assigned by following symbols to them: :add, :sub, :mul, :div, :max, :min and :mean.","category":"page"}]
}
