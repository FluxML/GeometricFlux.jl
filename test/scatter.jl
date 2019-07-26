ys = [3 3 4 4 5;
      5 5 6 6 7]
us = ones(Int, 2, 3, 4)
xs = [1 2 3 4;
      4 2 1 3;
      3 5 5 3]
types = [UInt8, UInt16, UInt32, UInt64,
         Int8, Int16, Int32, Int64, Int128,
         Float16, Float32, Float64]

@testset "Test Scatter Add" begin
    ys_ = [5 5 8 6 7;
           7 7 10 8 9]
    for T = types
        @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:add, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end

@testset "Test Scatter Sub" begin
    ys_ = [1 1 0 2 3;
           3 3 2 4 5]
    for T = types
        @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:sub, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end

@testset "Test Scatter Max" begin
    ys_ = [3 3 4 4 5;
           5 5 6 6 7]
    for T = types
        @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:max, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end

@testset "Test Scatter Min" begin
    ys_ = [1 1 1 1 1;
           1 1 1 1 1]
    for T = types
        @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:min, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end

@testset "Test Scatter Mul" begin
    ys_ = [3 3 4 4 5;
           5 5 6 6 7]
    for T = types
        @test scatter_mul!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:mul, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end

@testset "Test Scatter Div" begin
    us_div = us .* 2
    ys_ = [0.75 0.75 0.25 1. 1.25;
           1.25 1.25 0.375 1.5 1.75]
    for T = [Float16, Float32, Float64]
        @test scatter_div!(T.(copy(ys)), T.(us_div), xs) == T.(ys_)
        @test scatter!(:div, T.(copy(ys)), T.(us_div), xs) == T.(ys_)
    end
end

@testset "Test Scatter Mean" begin
    ys_ = [4 4 5 5 6;
           6 6 7 7 8]
    for T = [Float16, Float32, Float64]
        @test scatter_mean!(T.(copy(ys)), T.(us), xs) == T.(ys_)
        @test scatter!(:mean, T.(copy(ys)), T.(us), xs) == T.(ys_)
    end
end
