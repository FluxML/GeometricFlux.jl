entities_init() = register(DataDep(
    "Cora full datasets",
    """
    The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from
    the `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.
    """,
    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{}.tgz",
    "";
    fetch_method=http_download,
    post_fetch_method=DataDeps.unpack,
))
