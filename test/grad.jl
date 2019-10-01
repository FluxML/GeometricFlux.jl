using Random

rng, M, N, P, Q = MersenneTwister(1234567), 2, 3, 4, 5
ys = randn(rng, M, Q)
us = randn(rng, M, N, P)
xs = rand(1:Q, N, P)

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs[1:end-1])
    for (x, Δ) in zip(xs[1:end-1], grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

gradcheck(f, xs...) = (all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...)[1:end-1], rtol = 1e-5, atol = 1e-5)))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)


@testset "gradtest" begin
    @test gradtest(scatter_add!, copy(ys), us, xs)
    @test gradtest(scatter_sub!, copy(ys), us, xs)
    @test gradtest(scatter_max!, copy(ys), us, xs)
    @test gradtest(scatter_min!, copy(ys), us, xs)
    @test gradtest(scatter_mul!, copy(ys), us, xs)
    @test gradtest(scatter_div!, copy(ys), us, xs)
    @test gradtest(scatter_mean!, copy(ys), us, xs)
end
