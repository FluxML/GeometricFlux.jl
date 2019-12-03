ppi_init() = register(DataDep(
    "Cora full datasets",
    """
    The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).
    """,
    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip",
    "";
    fetch_method=http_download,
    post_fetch_method=DataDeps.unpack,
))
