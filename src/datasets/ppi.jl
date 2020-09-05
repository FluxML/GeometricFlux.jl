ppi_init() = register(DataDep(
    "PPI",
    """
    The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).
    """,
    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip",
    "1f5b2b09ac0f897fa6aa1338c64ab75a5473674cbba89380120bede8cddb2a6a";
    post_fetch_method=preprocess_ppi,
))

function preprocess_ppi(local_path)
    unzip(local_path)

    for phase in ["train", "test", "valid"]
        graph_file = @datadep_str "PPI/$(phase)_graph.json"
        id_file = @datadep_str "PPI/$(phase)_graph_id.npy"
        X_file = @datadep_str "PPI/$(phase)_feats.npy"
        y_file = @datadep_str "PPI/$(phase)_labels.npy"

        py"""
        import numpy as np
        ids = np.load($id_file)
        X = np.load($X_file)
        y = np.load($y_file)
        """

        X = Matrix{Float32}(py"X")
        y = SparseMatrixCSC{Int32,Int64}(Array(py"y"))
        ids = Array(py"ids")
        graph = read_ppi_graph(graph_file)
        
        jld2file = replace(local_path, "ppi.zip"=>"ppi.$(phase).jld2")
        @save jld2file graph X y ids
    end
end

function read_ppi_graph(filename::String)
    d = JSON.Parser.parsefile(filename)
    g = SimpleDiGraph{Int32}(length(d["nodes"]))

    for pair in d["links"]
        add_edge!(g, pair["source"], pair["target"])
    end
    g
end

struct PPI <: Dataset
end

function traindata(::PPI)
    file = datadep"PPI/ppi.train.jld2"
    @load file graph X y ids
    graph, X, y, ids
end

function validdata(::PPI)
    file = datadep"PPI/ppi.valid.jld2"
    @load file graph X y ids
    graph, X, y, ids
end

function testdata(::PPI)
    file = datadep"PPI/ppi.test.jld2"
    @load file graph X y ids
    graph, X, y, ids
end