using CuArrays

ys = cu([3 3 4 4 5;
         5 5 6 6 7])
us = cu(ones(Int, 2, 3, 4))
xs = CuArray{Int64}([1 2 3 4;
                     4 2 1 3;
                     3 5 5 3])


@testset "cuda/scatter" begin
    for T = [UInt32, UInt64, Int32, Int64]
        @testset "$(T)" begin
            @testset "add" begin
                ys_ = cu([5 5 8 6 7;
                          7 7 10 8 9])
                @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:add, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "sub" begin
                ys_ = cu([1 1 0 2 3;
                          3 3 2 4 5])
                @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:sub, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "max" begin
                ys_ = cu([3 3 4 4 5;
                          5 5 6 6 7])
                @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:max, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "min" begin
                ys_ = cu([1 1 1 1 1;
                          1 1 1 1 1])
                @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:min, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end


    for T = [Float32, Float64]
        @testset "$(T)" begin
            @testset "add" begin
                ys_ = cu([5 5 8 6 7;
                          7 7 10 8 9])
                @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:add, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "sub" begin
                ys_ = cu([1 1 0 2 3;
                          3 3 2 4 5])
                @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:sub, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "max" begin
                ys_ = cu([3 3 4 4 5;
                          5 5 6 6 7])
                @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:max, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "min" begin
                ys_ = cu([1 1 1 1 1;
                          1 1 1 1 1])
                @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:min, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "mul" begin
                ys_ = cu([3 3 4 4 5;
                          5 5 6 6 7])
                @test scatter_mul!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:mul, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "div" begin
                us_div = us .* 2
                ys_ = cu([0.75 0.75 0.25 1. 1.25;
                          1.25 1.25 0.375 1.5 1.75])
                @test scatter_div!(T.(copy(ys)), T.(us_div), xs) == T.(ys_)
                @test scatter!(:div, T.(copy(ys)), T.(us_div), xs) == T.(ys_)
            end

            @testset "mean" begin
                ys_ = cu([4 4 5 5 6;
                          6 6 7 7 8])
                @test scatter_mean!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter!(:mean, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end
end
