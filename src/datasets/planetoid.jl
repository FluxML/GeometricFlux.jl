const PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
const DATASETS = ["citeseer", "cora", "pubmed"]
const EXTS = ["allx", "ally", "graph", "test.index", "tx", "ty", "x", "y"]
const DATAURLS = [joinpath(PLANETOID_URL, "ind.$(dataset).$(ext)") for dataset in DATASETS, ext in EXTS]

planetoid_init() = register(DataDep(
    "Planetoid",
    """
    The citation network datasets "Cora", "CiteSeer", "PubMed" from
    "Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861> paper.
    Nodes represent documents and edges represent citation links.
    """,
    reshape(DATAURLS, :),
    # "58984da6e25012ee40ecc927e9f0fa7c0245a18ef0f4cc759dd657f83ec60bf8";
    # post_fetch_method=preprocess,
))

function preprocess(local_path)
    dataset = "cora"
    # for dataset in DATASETS
    graph_file = datadep"Planetoid/ind.cora.graph"
    X_file = datadep"Planetoid/ind.cora.allx"
    y_file = datadep"Planetoid/ind.cora.ally"
    test_file = datadep"Planetoid/ind.cora.test.index"

    X = read_data(X_file)
    y = read_data(y_file)
    testindex = read_index(test_file)
    graph = read_graph(graph_file)

    @save "$(datasets).train.jld2" g train_X train_y
    @save "$(datasets).test.jld2" g test_X test_y
    # end
end

function read_data(filename)
    py"""
    import pickle
    from scipy.sparse import csr_matrix

    with open($filename,"rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()

    if type(data) is csr_matrix:
        data = data.toarray()
    """
    return SparseMatrixCSC(Array(py"data"))
end

read_index(filename) = map(x -> parse(Int64, x), readlines(filename))

function read_graph(filename)
    py"""
    import pickle

    with open($filename,"rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        data = u.load()
    """
    return Dict(py"data")
end


function trainfile(dataset::Symbol)
    if !(dataset in [:citeseer, :cora, :pubmed])
        error("`dataset` should be one of citeseer, cora, pubmed.")
    end
    @load datadep"Planetoid/$(datasets).train.jld2" g train_X train_y
    g, train_X, train_y
end

function testfile(dataset::Symbol)
    if !(dataset in [:citeseer, :cora, :pubmed])
        error("`dataset` should be one of citeseer, cora, pubmed.")
    end
    @load datadep"Planetoid/$(datasets).test.jld2" g test_X test_y
    g, test_X, test_y
end
