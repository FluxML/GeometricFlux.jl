@testset "alias sampling" begin
    probs = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    J, q = GeometricFlux.alias_setup(probs)
    samples = [GeometricFlux.alias_sample(J, q) for _ in 1:1000]
    print
    @test length(J) == length(q) == length(probs)
    @test max(q...) <= 1.1
    @test min(q...) >= 0.0
    @test max(J...) <= length(probs)
    @test min(J...) >= 0.0
    @test length(samples) == 1000
end
