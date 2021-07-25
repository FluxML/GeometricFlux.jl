"""
    bypass_graph(nf_func, ef_func, gf_func)

Bypassing graph in FeaturedGraph and let other layer process (node, edge and global)features only.
"""
function bypass_graph(nf_func=identity, ef_func=identity, gf_func=identity)
    return function (fg::FeaturedGraph)
        FeaturedGraph(graph(fg),
                      nf=nf_func(node_feature(fg)),
                      ef=ef_func(edge_feature(fg)),
                      gf=gf_func(global_feature(fg)))
    end
end
