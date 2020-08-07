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
        ArgumentError("only accept :node, :edge and :global while got $feature")
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
