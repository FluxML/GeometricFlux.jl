@testset "gn" begin
    T = Float32
    in_channel = 10
    out_channel = 5
    V = 6
    E = 7

    nf = repeat(T.(collect(1:V)'), outer=(in_channel, 1))
    ef = repeat(T.(collect(1:E)'), outer=(in_channel, 1))
    gf = rand(T, in_channel)

    adj = T[0. 1. 0. 0. 0. 0.;
            1. 0. 0. 1. 1. 1.;
            0. 0. 0. 0. 0. 1.;
            0. 1. 0. 0. 1. 0.;
            0. 1. 0. 1. 0. 1.;
            0. 1. 1. 0. 1. 0.]

    struct NewGNLayer <: GraphNet end

    @testset "without aggregation" begin
        function (l::NewGNLayer)(fg::AbstractFeaturedGraph)
            nf = node_feature(fg)
            ef = edge_feature(fg)
            GraphSignals.check_num_nodes(fg, nf)
            GraphSignals.check_num_edges(fg, ef)
            return GeometricFlux.propagate(l, graph(fg), ef, nf, global_feature(fg), nothing, nothing, nothing)
        end

        fg = FeaturedGraph(adj, nf=nf)
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test nf_ == nf
        @test size(ef_) == (0, 2E)
        @test size(gf_) == (0,)
    end

    @testset "with neighbor aggregation" begin
        function (l::NewGNLayer)(fg::AbstractFeaturedGraph)
            nf = node_feature(fg)
            ef = edge_feature(fg)
            GraphSignals.check_num_nodes(fg, nf)
            GraphSignals.check_num_edges(fg, ef)
            return GeometricFlux.propagate(l, graph(fg), ef, nf, global_feature(fg), +, nothing, nothing)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=zeros(0))
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (in_channel, V)
        @test size(ef_) == (0, 2E)
        @test size(gf_) == (0,)
    end

    GeometricFlux.update_edge(l::NewGNLayer, e, vi, vj, u) = similar(e, out_channel, size(e)[2:end]...)
    @testset "update edge with neighbor aggregation" begin
        function (l::NewGNLayer)(fg::AbstractFeaturedGraph)
            nf = node_feature(fg)
            ef = edge_feature(fg)
            GraphSignals.check_num_nodes(fg, nf)
            GraphSignals.check_num_edges(fg, ef)
            return GeometricFlux.propagate(l, graph(fg), ef, nf, global_feature(fg), +, nothing, nothing)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=zeros(0))
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (in_channel, V)
        @test size(ef_) == (out_channel, 2E)
        @test size(gf_) == (0,)
    end

    GeometricFlux.update_vertex(l::NewGNLayer, ē, vi, u) = similar(vi, out_channel, size(vi)[2:end]...)
    @testset "update edge/vertex with all aggregation" begin
        function (l::NewGNLayer)(fg::AbstractFeaturedGraph)
            nf = node_feature(fg)
            ef = edge_feature(fg)
            GraphSignals.check_num_nodes(fg, nf)
            GraphSignals.check_num_edges(fg, ef)
            return GeometricFlux.propagate(l, graph(fg), ef, nf, global_feature(fg), +, +, +)
        end

        fg = FeaturedGraph(adj, nf=nf, ef=ef, gf=gf)
        l = NewGNLayer()
        ef_, nf_, gf_ = l(fg)

        @test size(nf_) == (out_channel, V)
        @test size(ef_) == (out_channel, 2E)
        @test size(gf_) == (in_channel,)
    end
end
