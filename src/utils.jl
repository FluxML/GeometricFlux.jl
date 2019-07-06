function fadj(g::AbstractSimpleWeightedGraph)
    N = nv(g)
    el = Vector{Vector{Int}}(N)
    for i = 1:N
        el[i] = outneighbors(g, i)
    end
    return el
end
