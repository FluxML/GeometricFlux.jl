N = 4
E = 5
adj = [0 1 1 1;
       1 0 1 0;
       1 1 0 1;
       1 0 1 0]

nf = rand(3, N)
ef = rand(5, E)
gf = rand(7)

@testset "utils" begin
    @testset "topk_index" begin
        X = [8,7,6,5,4,3,2,1]
        @test topk_index(X, 4) == [1,2,3,4]
        @test topk_index(X', 4) == [1,2,3,4]
    end

    @testset "bypass_graph" begin
        fg = FeaturedGraph(adj, nf, ef ,gf)
        layer = bypass_graph(x -> x .+ 1.,
                             x -> x .+ 2.,
                             x -> x .+ 3.)
        fg_ = layer(fg)
        @test graph(fg_) == adj
        @test node_feature(fg_) == nf .+ 1.
        @test edge_feature(fg_) == ef .+ 2.
        @test global_feature(fg_) == gf .+ 3.
    end
end
