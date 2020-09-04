qm7b_init() = register(DataDep(
    "Cora full datasets",
    """
    The QM7b dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    7,211 molecules with 14 regression targets.
    """,
    "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat",
    "";
    fetch_method=http_download,
    post_fetch_method=DataDeps.unpack,
))
