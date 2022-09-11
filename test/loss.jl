@testset "loss" begin
    T = Float32
    N = 4
    adj = T[0. 1. 0. 1.;
            1. 0. 1. 0.;
            0. 1. 0. 1.;
            1. 0. 1. 0.]
    fg = FeaturedGraph(adj)

    @testset "laplacian_eig_loss" begin
        λ = 1f0
        k = 3
        p = rand(T, k, N)
        L = GraphSignals.laplacian_matrix(fg, T)
        @test laplacian_eig_loss(p, L, λ) > 0
        @test_throws AssertionError laplacian_eig_loss(p, L, -λ)
    end
end
