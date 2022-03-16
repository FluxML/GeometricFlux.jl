@testset "models" begin
    batch_size = 10
    in_channel = 3
    out_channel = 5
    N = 4
    batch = 4
    T = Float32
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]

    fg = FeaturedGraph(adj)

    @testset "GAE" begin
        gc = WithGraph(fg, GCNConv(in_channel=>out_channel))
        gae = GAE(gc)
        X = rand(T, in_channel, N)
        Y = gae(X)
        @test size(Y) == (N, N)
    end

    @testset "VGAE" begin
        @testset "InnerProductDecoder" begin
           ipd = InnerProductDecoder(identity)
           X = rand(T, 1, N, batch)
           Y = ipd(X)
           @test size(Y) == (N, N, batch)

           X = rand(T, 1, N)
           fg = FeaturedGraph(adj, nf=X)
           fg_ = ipd(fg)
           Y = node_feature(fg_)
           @test size(Y) == (N, N)

           X = rand(T, in_channel, N)
           fg = FeaturedGraph(adj, nf=X)
           fg_ = ipd(fg)
           Y = node_feature(fg_)
           @test size(Y) == (N, N)
        end

        @testset "VariationalGraphEncoder" begin
            z_dim = 2
            gc = GCNConv(in_channel=>out_channel)
            ve = VariationalGraphEncoder(gc, out_channel, z_dim)
            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = ve(fg)
            Z = node_feature(fg_)
            @test size(Z) == (z_dim, N)
        end

        @testset "VGAE" begin
            z_dim = 2
            gc = GCNConv(in_channel=>out_channel)
            vgae = VGAE(gc, out_channel, z_dim)
            X = rand(T, in_channel, N)
            fg = FeaturedGraph(adj, nf=X)
            fg_ = vgae(fg)
            Y = node_feature(fg_)
            @test size(Y) == (N, N)
        end

        @testset "DeepSet" begin
            ϕ = Dense(64, 16)
            ρ = Dense(16, 4)
            @testset "layer without graph" begin
                deepset = DeepSet(ϕ, ρ)

                X = rand(T, 64, N)
                fg = FeaturedGraph(adj, nf=X)
                fg_ = deepset(fg)
                @test size(global_feature(fg_)) == (4, 1)

                g = Zygote.gradient(() -> sum(global_feature(deepset(fg))), Flux.params(deepset))
                @test length(g.grads) == 2
            end

            @testset "layer with static graph" begin
                X = rand(T, 64, N, batch_size)
                deepset = WithGraph(fg, DeepSet(ϕ, ρ))
                Y = deepset(X)
                @test size(Y) == (4, 1, batch_size)

                g = Zygote.gradient(() -> sum(deepset(X)), Flux.params(deepset))
                @test length(g.grads) == 0
            end
        end
    end
end
