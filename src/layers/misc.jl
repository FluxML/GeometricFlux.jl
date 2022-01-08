"""
Bypassing graph in FeaturedGraph and let other layer process (node, edge and global) features only.
"""
struct Bypass{N,E,G}
    node_layer::N
    edge_layer::E
    global_layer::G
end

@functor Bypass

Bypass(; node_layer=identity, edge_layer=identity, global_layer=identity) =
    Bypass(node_layer, edge_layer, global_layer)

function (l::Bypass)(fg::FeaturedGraph)
    nf = l.node_layer(node_feature(fg))
    ef = l.edge_layer(edge_feature(fg))
    gf = l.global_layer(global_feature(fg))
    return FeaturedGraph(fg, nf=nf, ef=ef, gf=gf)
end

function (l::Bypass)(fsg::FeaturedSubgraph)
    nf = l.node_layer(node_feature(fsg))
    ef = l.edge_layer(edge_feature(fsg))
    gf = l.global_layer(global_feature(fsg))
    fg = parent(fsg)
    vidx = fsg.nodes
    nf = NNlib.scatter(+, nf, vidx; init=0, dstsize=(size(nf,1), nv(fg)))
    ef = NNlib.scatter(+, ef, edges(fsg); init=0, dstsize=(size(ef,1), ne(fg)))
    fg = FeaturedGraph(fg, nf=nf, ef=ef, gf=gf)
    return subgraph(fg, vidx)
end
