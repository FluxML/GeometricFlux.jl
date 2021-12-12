using GeometricFlux
using GraphSignals
using Graphs
using SparseArrays
using Plots
using GraphPlot
using Clustering
using Cairo, Compose

clusters = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

int2col_str(x::Int) = x==1 ? "lightblue" : "red"


g = smallgraph(:karate)
fg = FeaturedGraph(g)
vectors = node2vec(fg; walks_per_node=10, len=80, p=1.0, q=1.0)
R = kmeans(vectors, 2)


learned_clusters = copy(assignments(R))
# ensure that the cluster containing node 1 is cluster 1
if assignments(R)[1] != 1
    learned_clusters = [i == 1 ? 2 : 1 for i in assignments(R)]
end

output_plot_name = "karateclub.pdf"
draw(
    PDF(output_plot_name, 16cm, 16cm),
    gplot(g,
        nodelabel=map(string, 1:34),
        nodefillc=[int2col_str(learned_clusters[i]) for i in 1:34],
        nodestrokec=["white" for _ in 1:34]
    )
)

incorrect = sum(learned_clusters .!= clusters)
println(incorrect, " incorrect cluster labelings")
println("Drawn graph to ", output_plot_name)
