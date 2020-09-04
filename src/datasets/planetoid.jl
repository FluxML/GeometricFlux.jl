const PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
const PLANETOID_DATASETS = [:citeseer, :cora, :pubmed]
const EXTS = ["allx", "ally", "graph", "test.index", "tx", "ty", "x", "y"]
const DATAURLS = [joinpath(PLANETOID_URL, "ind.$(d).$(ext)") for d in PLANETOID_DATASETS, ext in EXTS]

planetoid_init() = register(DataDep(
    "Planetoid",
    """
    The citation network datasets "Cora", "CiteSeer", "PubMed" from
    "Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861> paper.
    Nodes represent documents and edges represent citation links.
    """,
    reshape(DATAURLS, :),
    "f52b3d47f5993912d7509b51e8090b6807228c4ba8c7d906f946868005c61c18";
    post_fetch_method=preprocess_planetoid,
))

function preprocess_planetoid(local_path)
    for dataset in PLANETOID_DATASETS
        graph_file = @datadep_str "Planetoid/ind.$(dataset).graph"
        trainX_file = @datadep_str "Planetoid/ind.$(dataset).x"
        trainy_file = @datadep_str "Planetoid/ind.$(dataset).y"
        testX_file = @datadep_str "Planetoid/ind.$(dataset).tx"
        testy_file = @datadep_str "Planetoid/ind.$(dataset).ty"

        train_X = read_data(trainX_file)
        train_y = read_data(trainy_file)
        test_X = read_data(testX_file)
        test_y = read_data(testy_file)
        graph = read_graph(graph_file)

        trainfile = replace(graph_file, "ind.$(dataset).graph"=>"$(dataset).train.jld2")
        testfile = replace(graph_file, "ind.$(dataset).graph"=>"$(dataset).test.jld2")
        @save trainfile graph train_X train_y
        @save testfile graph test_X test_y
    end
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

struct Planetoid <: Dataset
    dataset::Symbol

    function Planetoid(ds::Symbol)
        ds in PLANETOID_DATASETS || throw(error("`dataset` should be one of citeseer, cora, pubmed."))
        new(ds)
    end
end


function traindata(pla::Planetoid)
    file = @datadep_str "Planetoid/$(pla.dataset).train.jld2"
    @load file graph train_X train_y
    graph, train_X, train_y
end

function testdata(pla::Planetoid)
    file = @datadep_str "Planetoid/$(pla.dataset).test.jld2"
    @load file graph test_X test_y
    graph, test_X, test_y
end
