@testset "qm7b" begin
    names, X, T = dataset(QM7b())
    @test typeof(names) == Vector{String}
    @test size(names) == (14,)
    @test typeof(X) == Array{Float32,3}
    @test size(X) == (7211, 23, 23)
    @test typeof(T) == Matrix{Float32}
    @test size(T) == (7211, 14)
end