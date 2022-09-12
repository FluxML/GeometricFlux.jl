@testset "cuda/group_conv" begin
    T = Float32
    batch_size = 10
    in_dim = 3
    out_dim = 5
    pos_dim = 2
    in_edge_dim = 7

    N = 4
    E = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]

    @testset "EEquivGraphConv" begin
        @testset "layer without graph" begin
            l = EEquivGraphConv(in_dim=>out_dim, pos_dim, in_edge_dim) |> gpu

            nf = rand(T, in_dim, N)
            ef = rand(T, in_edge_dim, E)
            pf = rand(T, pos_dim, N)
            fg = FeaturedGraph(adj, nf=nf, ef=ef, pf=pf) |> gpu
            fg_ = l(fg)
            nf_ = node_feature(fg_)
            pf_ = positional_feature(fg_)

            @test size(nf_) == (out_dim, N)
            @test size(pf_) == (pos_dim, N)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 13
        end

        @testset "layer without graph and edge feature" begin
            l = EEquivGraphConv(in_dim=>out_dim, pos_dim) |> gpu

            nf = rand(T, in_dim, N)
            pf = rand(T, pos_dim, N)
            fg = FeaturedGraph(adj, nf=nf, pf=pf) |> gpu
            fg_ = l(fg)
            nf_ = node_feature(fg_)
            pf_ = positional_feature(fg_)

            @test size(nf_) == (out_dim, N)
            @test size(pf_) == (pos_dim, N)

            g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
            @test length(g.grads) == 13
        end

        @testset "layer with static graph" begin
            nf = rand(T, in_dim, N, batch_size)
            ef = rand(T, in_edge_dim, E, batch_size)
            fg = FeaturedGraph(adj, pf = rand(T, pos_dim, N, batch_size))
            l = WithGraph(fg, EEquivGraphConv(in_dim=>out_dim, pos_dim, in_edge_dim)) |> gpu
            H, Y = l(nf |> gpu, ef |> gpu)
            @test size(H) == (out_dim, N, batch_size)
            @test size(Y) == (pos_dim, N, batch_size)

            g = gradient(() -> sum(l(nf |> gpu, ef |> gpu)[1]), Flux.params(l))
            @test length(g.grads) == 13
        end

        @testset "layer with static graph without edge feature" begin
            nf = rand(T, in_dim, N, batch_size)
            fg = FeaturedGraph(adj, pf = rand(T, pos_dim, N, batch_size))
            l = WithGraph(fg, EEquivGraphConv(in_dim=>out_dim, pos_dim)) |> gpu
            H, Y = l(nf |> gpu)
            @test size(H) == (out_dim, N, batch_size)
            @test size(Y) == (pos_dim, N, batch_size)

            g = gradient(() -> sum(l(nf |> gpu)[1]), Flux.params(l))
            @test length(g.grads) == 12
        end
    end
end
