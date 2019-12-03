cora_init() = register(DataDep(
    "Cora",
    """
    The full Cora citation network dataset from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
    Nodes represent documents and edges represent citation links.
    """,
    "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz",
    "";
    post_fetch_method=DataDeps.unpack,
))
