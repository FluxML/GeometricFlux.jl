@testset "node2vec" begin
    clusters = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    g = smallgraph(:karate)
    fg = FeaturedGraph(g)
    vectors = node2vec(fg; walks_per_node=10, len=80, p=1.0, q=1.0)
    R = kmeans(vectors, 2)

    learned_clusters = copy(assignments(R))
    # ensure that the cluster containing node 1 is cluster 1
    if assignments(R)[1] != 1
        learned_clusters = [i == 1 ? 2 : 1 for i in assignments(R)]
    end

    incorrect = sum(learned_clusters .!= clusters)
    @test incorrect < 4
end
