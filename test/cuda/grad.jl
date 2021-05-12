ys = cu([3. 3. 4. 4. 5.;
         5. 5. 6. 6. 7.])
us = cu(2*ones(2, 3, 4))
xs = CuArray{Int64}([1 2 3 4;
                     4 2 1 3;
                     3 5 5 3])

∇y_mul = [4. 4. 16. 4. 4.; 4. 4. 16. 4. 4.]
∇y_div = [.25 .25 .0625 .25 .25; .25 .25 .0625 .25 .25]
∇u_mean = cat([.5 .5 .25; .5 .5 .25], [.5 .5 .5; .5 .5 .5],
              [.25 .5 .5; .25 .5 .5], [.5 .25 .25; .5 .25 .25], dims=3)

@testset "cuda/grad" begin
    @testset "pool" begin
        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(+, x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(+, xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(-, x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(-, xs, x)), us) == (-ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(max, x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(max, xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(min, x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(GeometricFlux.scatter(min, xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(prodpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(prodpool(xs, x)), us) == (2048*ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(divpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(divpool(xs, x)), us) == (-ones(2, 3, 4)/8192,)

        @test Zygote.gradient(x -> sum(meanpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(meanpool(xs, x)), us) == (∇u_mean,)
    end
end
