num_node = 5
A = rand([0, 1], 5, 5)
A = A .| A'
num_edge = sum(A)
nf = rand(6, num_node)
ef = rand(7, num_edge)
gf = rand(8)

fg = FeaturedGraph(A, nf, ef, gf)

@testset "selector" begin
    fs = FeatureSelector(:node)
    @test fs(fg) == nf

    fs = FeatureSelector(:edge)
    @test fs(fg) == ef

    fs = FeatureSelector(:global)
    @test fs(fg) == gf

    @test_throws ArgumentError FeatureSelector(:foo)
end
