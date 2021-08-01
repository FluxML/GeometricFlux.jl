@testset "gn" begin
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 7
    T = Float32

    adj =  [0 1 0 0 0 0
            1 0 0 1 1 1
            0 0 0 0 0 1
            0 1 0 0 1 0
            0 1 0 1 0 1
            0 1 1 0 1 0]

    struct NewGNLayer{G} <: GraphNet end
    NewGNLayer() = NewGNLayer{GRAPH_T}()

    V = rand(T, in_channel, num_V)
    E = rand(T, in_channel, 2num_E)
    u = rand(T, in_channel)

    @testset "without aggregation" begin
        (l::NewGNLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg)

        fg = FeaturedGraph(adj, nf=V, graph_type=GRAPH_T)
        l = NewGNLayer()
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test all(edge_feature(fg_) .== fill(nothing, 2*num_E))
        @test global_feature(fg_) === nothing
    end

    @testset "with neighbor aggregation" begin
        (l::NewGNLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=nothing, graph_type=GRAPH_T)
        l = NewGNLayer()
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
        @test global_feature(fg_) === nothing
    end

    @testset "update edge with neighbor aggregation" begin
        (l::NewGNLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +)
        GeometricFlux.update_edge(l::NewGNLayer{GRAPH_T}, e, vi, vj, u) = rand(T, out_channel)
    

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=nothing, graph_type=GRAPH_T)
        l = NewGNLayer()
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test global_feature(fg_) === nothing
    end

    @testset "update edge/vertex with all aggregation" begin
        (l::NewGNLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +, +, +)
        GeometricFlux.update_vertex(l::NewGNLayer{GRAPH_T}, ē, vi, u) = rand(T, out_channel)

        fg = FeaturedGraph(adj, nf=V, ef=E, gf=u, graph_type=GRAPH_T)
        l = NewGNLayer()
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (in_channel,)
    end
end
