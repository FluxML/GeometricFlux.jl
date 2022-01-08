@testset "misc" begin
    @testset "Bypass" begin
        T = Float32
        N = 4
        E = 5
        adj = T[0 1 1 1;
                1 0 1 0;
                1 1 0 1;
                1 0 1 0]

        nf = rand(3, N)
        ef = rand(5, E)
        gf = rand(7)
        nodes = [1,2,4]

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=gf)
        fsg = subgraph(fg, nodes)

        layer = Bypass(node_layer=Dropout(0.5),
                       global_layer=x -> x .+ 3.)
        fsg_ = layer(fsg)
        @test node_feature(fsg_) == view(nf, :, nodes)
        @test edge_feature(fsg_) == view(ef, :, edges(fsg))
        @test global_feature(fsg_) == gf .+ 3.
    end
end
