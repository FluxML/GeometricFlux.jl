reddit_init() = register(DataDep(
    "Reddit",
    """
    The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.
    """,
    "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/reddit.zip",
    "";
    fetch_method=http_download,
    post_fetch_method=DataDeps.unpack,
))
