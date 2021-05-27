struct FeatureSelector
    enable_node_feature::Bool
    enable_edge_feature::Bool
    enable_global_feature::Bool
end

function FeatureSelector(feature::Symbol)
    if feature == :node
        return FeatureSelector(true, false, false)
    elseif feature == :edge
        return FeatureSelector(false, true, false)
    elseif feature == :global
        return FeatureSelector(false, false, true)
    else
        throw(ArgumentError("only accept :node, :edge and :global while got $feature"))
    end
end

function (fs::FeatureSelector)(fg::FeaturedGraph)
    if fs.enable_node_feature && has_node_feature(fg)
        return node_feature(fg)
    elseif fs.enable_edge_feature && has_edge_feature(fg)
        return edge_feature(fg)
    elseif fs.enable_global_feature && has_global_feature(fg)
        return global_feature(fg)
    end
end

(fs::FeatureSelector)(fg::NullGraph) = nothing


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
