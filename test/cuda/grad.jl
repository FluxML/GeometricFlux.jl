using CuArrays

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
    @testset "scatter" begin
        @test Zygote.gradient(x -> sum(scatter_add!(x, us, xs)), ys) == (ones(2, 5),)
        @test Zygote.gradient(x -> sum(scatter_add!(copy(ys), x, xs)), us) == (ones(2, 3, 4),)
        @test Zygote.gradient(x -> sum(scatter_add!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_sub!(x, us, xs)), ys) == (ones(2, 5),)
        @test Zygote.gradient(x -> sum(scatter_sub!(copy(ys), x, xs)), us) == (-ones(2, 3, 4),)
        @test Zygote.gradient(x -> sum(scatter_sub!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_max!(x, us, xs)), ys) == (ones(2, 5),)
        @test Zygote.gradient(x -> sum(scatter_max!(copy(ys), x, xs)), us) == (zeros(2, 3, 4),)
        @test Zygote.gradient(x -> sum(scatter_max!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_min!(x, us, xs)), ys) == (zeros(2, 5),)
        @test Zygote.gradient(x -> sum(scatter_min!(copy(ys), x, xs)), us) == (ones(2, 3, 4),)
        @test Zygote.gradient(x -> sum(scatter_min!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_mul!(x, us, xs)), ys) == (∇y_mul,)
        @test Zygote.gradient(x -> sum(scatter_mul!(copy(ys), x, xs)), us) == (2048*gather(ys, xs),)
        @test Zygote.gradient(x -> sum(scatter_mul!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_div!(x, us, xs)), ys) == (∇y_div,)
        @test Zygote.gradient(x -> sum(scatter_div!(copy(ys), x, xs)), us) == (-gather(ys, xs)/8192,)
        @test Zygote.gradient(x -> sum(scatter_div!(copy(ys), us, x)), xs) == (nothing,)

        @test Zygote.gradient(x -> sum(scatter_mean!(x, us, xs)), ys) == (ones(2, 5),)
        @test Zygote.gradient(x -> sum(scatter_mean!(copy(ys), x, xs)), us) == (∇u_mean,)
        @test Zygote.gradient(x -> sum(scatter_mean!(copy(ys), us, x)), xs) == (nothing,)
    end

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
