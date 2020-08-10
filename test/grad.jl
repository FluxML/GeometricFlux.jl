ys = [3. 3. 4. 4. 5.;
      5. 5. 6. 6. 7.]
us = 2*ones(2, 3, 4)
xs = [1 2 3 4;
      4 2 1 3;
      3 5 5 3]

∇y_mul = [4. 4. 16. 4. 4.; 4. 4. 16. 4. 4.]
∇y_div = [.25 .25 .0625 .25 .25; .25 .25 .0625 .25 .25]
∇u_mean = cat([.5 .5 .25; .5 .5 .25], [.5 .5 .5; .5 .5 .5],
              [.25 .5 .5; .25 .5 .5], [.5 .25 .25; .5 .25 .25], dims=3)

@testset "grad" begin
    @testset "pool" begin
        @test Zygote.gradient(x -> sum(sumpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(sumpool(xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(subpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(subpool(xs, x)), us) == (-ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(maxpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(maxpool(xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(minpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(minpool(xs, x)), us) == (ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(prodpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(prodpool(xs, x)), us) == (2048*ones(2, 3, 4),)

        @test Zygote.gradient(x -> sum(divpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(divpool(xs, x)), us) == (-ones(2, 3, 4)/8192,)

        @test Zygote.gradient(x -> sum(meanpool(x, us)), xs) == (nothing,)
        @test Zygote.gradient(x -> sum(meanpool(xs, x)), us) == (∇u_mean,)
    end
end
