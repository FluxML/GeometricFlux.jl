cora_init() = register(DataDep(
    "Cora",
    """
    The full Cora citation network dataset from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    """,
    "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz",
    "62e054f93be00a3dedb15b7ac15a2a07168ceab68b40bf95f54d2289d024c6bc";
    post_fetch_method=preprocess_cora,
))

function preprocess_cora(local_path)
    py"""
    import numpy as np
    import scipy.sparse as sp
    data = np.load($local_path, allow_pickle=True)
    A = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=data['adj_shape'])
    X = sp.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), shape=data['attr_shape'])
    """
    
    graph = SparseMatrixCSC(Array(py"A.toarray()"))
    X = SparseMatrixCSC(Array(py"X.toarray()"))
    y = py"data['labels']"

    # trainfile = replace(local_path, "cora.npz"=>"cora.train.jld2")
    # testfile = replace(local_path, "cora.npz"=>"cora.test.jld2")
    # @save trainfile graph train_X train_y
    # @save testfile graph test_X test_y

    datafile = replace(cora_file, "cora.npz"=>"cora.all.jld2")
    @save datafile graph X y
end

struct Cora <: Dataset
end

function dataset(cora::Cora)
    file = datadep"Cora/cora.all.jld2"
    @load file graph X y
    graph, X, y
end

# function traindata(cora::Cora)
#     file = datadep"Cora/cora.train.jld2"
#     @load file graph train_X train_y
#     graph, train_X, train_y
# end

# function testdata(dataset::Symbol)
#     file = datadep"Cora/cora.test.jld2"
#     @load file graph test_X test_y
#     graph, test_X, test_y
# end
