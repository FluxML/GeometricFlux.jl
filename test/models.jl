using Flux: Dense

in_channel = 3
out_channel = 5
N = 4
adj = [0. 1. 0. 1.;
       1. 0. 1. 0.;
       0. 1. 0. 1.;
       1. 0. 1. 0.]

@testset "models" begin
    @testset "GAE" begin
        gc = GCNConv(adj, in_channel=>out_channel)
        gae = GAE(gc)
        X = rand(in_channel, N)
        Y = gae(X)
        @test size(Y) == (N, N)
    end

    @testset "VGAE" begin
        @testset "InnerProductDecoder" begin
           ipd = InnerProductDecoder(identity)
           Y = ipd(rand(1, N))
           @test size(Y) == (N, N)

           Y = ipd(rand(in_channel, N))
           @test size(Y) == (N, N)
        end

        @testset "VariationalEncoder" begin
            z_dim = 2
            gc = GCNConv(adj, in_channel=>out_channel)
            ve = VariationalEncoder(gc, out_channel, z_dim)
            X = rand(in_channel, N)
            Z = ve(X)
            @test size(Z) == (z_dim, N)
        end

        @testset "VGAE" begin
            z_dim = 2
            gc = GCNConv(adj, in_channel=>out_channel)
            vgae = VGAE(gc, out_channel, z_dim)
            X = rand(in_channel, N)
            Y = vgae(X)
            @test size(Y) == (N, N)
        end
    end
end
