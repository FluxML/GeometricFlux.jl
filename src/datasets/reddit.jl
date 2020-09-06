reddit_init() = register(DataDep(
    "Reddit",
    """
    The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.
    """,
    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip",
    "9a16353c28f8ddd07148fc5ac9b57b818d7911ea0fbe9052d66d49fc32b372bf";
    post_fetch_method=preprocess_reddit,
))

function preprocess_reddit(local_path)
    unzip(local_path)

    graph_file = datadep"Reddit/reddit_graph.npz"
    data_file = datadep"Reddit/reddit_data.npz"

    py"""
    import numpy as np
    import scipy.sparse as sp
    graph = np.load($graph_file, allow_pickle=True)
    data = np.load($data_file, allow_pickle=True)
    """

    graph = sparse(Vector(py"graph['row']") .+ 1,
                   Vector(py"graph['col']") .+ 1,
                   Vector{Int32}(py"graph['data']"))
    X = Matrix{Float32}(py"data['feature']")
    y = Vector{Int32}(py"data['label']")
    ids = Vector{Int32}(py"data['node_ids']")
    types = Vector{Int32}(py"data['node_types']")
    
    jld2file = replace(local_path, "reddit.zip"=>"reddit.all.jld2")
    @save jld2file graph X y ids types
end

struct Reddit <: Dataset
end

function dataset(::Reddit)
    file = datadep"Reddit/reddit.all.jld2"
    @load file graph X y ids types
    graph, X, y, ids, types
end