@testset "msgpass" begin
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

    struct NewLayer{G} <: MessagePassing
        weight
    end
    NewLayer(m, n) = NewLayer{GRAPH_T}(randn(T, m,n))

    X = Array{T}(reshape(1:num_V*in_channel, in_channel, num_V))
    
    @testset "no message or update" begin
        (l::NewLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +)

        fg = FeaturedGraph(adj, nf=X, ef=Fill(zero(T), 0, 2num_E), graph_type=GRAPH_T)
        l = NewLayer(out_channel, in_channel)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (in_channel, num_V)
        @test size(edge_feature(fg_)) == (in_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    
    @testset "message function" begin
        (l::NewLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +)
        GeometricFlux.message(l::NewLayer{GRAPH_T}, x_i, x_j, e_ij) = l.weight * x_j
    
        fg = FeaturedGraph(adj, nf=X, ef=Fill(zero(T), 0, 2num_E), graph_type=GRAPH_T)
        l = NewLayer(out_channel, in_channel)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end

    @testset "message and update" begin
        (l::NewLayer{GRAPH_T})(fg) = GeometricFlux.propagate(l, fg, +)
        GeometricFlux.update(l::NewLayer{GRAPH_T}, m, x) = l.weight * x + m

        fg = FeaturedGraph(adj, nf=X, ef=Fill(zero(T), 0, 2num_E), graph_type=GRAPH_T)
        l = NewLayer(out_channel, in_channel)
        fg_ = l(fg)

        @test adjacency_matrix(fg_) == adj
        @test size(node_feature(fg_)) == (out_channel, num_V)
        @test size(edge_feature(fg_)) == (out_channel, 2*num_E)
        @test size(global_feature(fg_)) == (0,)
    end
end
