@testset "gn" begin
    T = Float32
    in_channel = 10
    out_channel = 5
    V = 6
    E = 7

    nf = rand(T, in_channel, V)
    ef = rand(T, in_channel, E)
    gf = rand(T, in_channel)

    adj = T[0. 1. 0. 0. 0. 0.;
            1. 0. 0. 1. 1. 1.;
            0. 0. 0. 0. 0. 1.;
            0. 1. 0. 0. 1. 0.;
            0. 1. 0. 1. 0. 1.;
            0. 1. 1. 0. 1. 0.]

    struct NewGNLayer <: GraphNet end
    
    l = NewGNLayer()

    @testset "without aggregation" begin
        function (l::NewGNLayer)(fg::FeaturedGraph)
            GeometricFlux.propagate(l, fg, edge_feature(fg), node_feature(fg), global_feature(fg))
        end

        fg = FeaturedGraph(adj, nf=nf)
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (in_channel, V)
        @test size(ef_) == (0, 2E)
        @test size(gf_) == (0,)
    end

    @testset "with neighbor aggregation" begin
        function (l::NewGNLayer)(fg::FeaturedGraph)
            GeometricFlux.propagate(l, fg, edge_feature(fg), node_feature(fg), global_feature(fg), +)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=zeros(0))
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (in_channel, V)
        @test size(ef_) == (in_channel, 2E)
        @test size(gf_) == (0,)
    end

    GeometricFlux.update_edge(l::NewGNLayer, e, vi, vj, u) = rand(T, out_channel)
    @testset "update edge with neighbor aggregation" begin
        function (l::NewGNLayer)(fg::FeaturedGraph)
            GeometricFlux.propagate(l, fg, edge_feature(fg), node_feature(fg), global_feature(fg), +)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=zeros(0))
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (in_channel, V)
        @test size(ef_) == (out_channel, 2E)
        @test size(gf_) == (0,)
    end

    GeometricFlux.update_vertex(l::NewGNLayer, ē, vi, u) = rand(T, out_channel)
    @testset "update edge/vertex with all aggregation" begin
        function (l::NewGNLayer)(fg::FeaturedGraph)
            GeometricFlux.propagate(l, fg, edge_feature(fg), node_feature(fg), global_feature(fg), +, +, +)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=gf)
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (out_channel, V)
        @test size(ef_) == (out_channel, 2E)
        @test size(gf_) == (in_channel,)
    end
end
