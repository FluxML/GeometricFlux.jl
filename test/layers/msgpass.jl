@testset "MessagePassing" begin 
    in_channel = 10
    out_channel = 5
    num_V = 6
    num_E = 14
    T = Float32

    adj =  [0 1 0 0 0 0
            1 0 0 1 1 1
            0 0 0 0 0 1
            0 1 0 0 1 0
            0 1 0 1 0 1
            0 1 1 0 1 0]

    struct NewLayer <: MessagePassing end

    X = rand(T, in_channel, num_V)
    E = rand(T, in_channel, num_E)
    u = rand(T, in_channel)


    @testset "default aggregation (+)" begin
        l = NewLayer()
        (l::NewLayer)(fg) = GeometricFlux.propagate(l, fg)

        fg = FeaturedGraph(adj, nf=X)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test edge_feature(fg_)  === nothing
        @test global_feature(fg_) === nothing
    end

    @testset "neighbor aggregation (+)" begin
        l = NewLayer()
        (l::NewLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test edge_feature(fg_) ≈ E
        @test global_feature(fg_) ≈ u
    end

    GeometricFlux.message(l::NewLayer, xi, xj, e, u) = ones(T, out_channel, size(e,2))

    @testset "custom message and neighbor aggregation" begin
        l = NewLayer()
        (l::NewLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test edge_feature(fg_) ≈ edge_feature(fg)
        @test global_feature(fg_) ≈ global_feature(fg)
    end

    GeometricFlux.update_edge(l::NewLayer, m, e) = m

    @testset "update_edge" begin
        l = NewLayer()
        (l::NewLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, num_E)
        @test global_feature(fg_) ≈ global_feature(fg)
    end

    GeometricFlux.update(l::NewLayer, m̄, xi, u) = rand(T, 2*out_channel, size(xi, 2))

    @testset "update edge/vertex" begin
        l = NewLayer()
        (l::NewLayer)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u)
        fg_ = l(fg)

        @test all(adjacency_matrix(fg_) .== adj)
        @test size(node_feature(fg_)) == (2*out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, num_E)
        @test size(global_feature(fg_)) == (in_channel,)
    end

    struct NewLayerW <: MessagePassing
        weight
    end

    NewLayerW(in, out) = NewLayerW(randn(T, out, in))

    GeometricFlux.message(l::NewLayerW, x_i, x_j, e_ij) = l.weight * x_j
    GeometricFlux.update(l::NewLayerW, m, x) = l.weight * x + m

    @testset "message and update with weights" begin
        l = NewLayerW(in_channel, out_channel)
        (l::NewLayerW)(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=E, gf=u)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test edge_feature(fg_) === E
        @test global_feature(fg_) === u
    end
end
