ys = [3 3 4 4 5;
      5 5 6 6 7]
us = ones(Int, 2, 3, 4)
xs = [1 2 3 4;
      4 2 1 3;
      3 5 5 3]
types = [UInt8, UInt16, UInt32, UInt64, UInt128,
         Int8, Int16, Int32, Int64, Int128, BigInt,
         Float16, Float32, Float64, BigFloat, Rational]

@testset "scatter" begin
    for T = types
        @testset "$T" begin
            PT = promote_type(T, Int)
            @testset "scatter_add!" begin
                ys_ = [5 5 8 6 7;
                       7 7 10 8 9]
                @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_add!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_add!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:add, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_sub!" begin
                ys_ = [1 1 0 2 3;
                       3 3 2 4 5]
                @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_sub!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_sub!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:sub, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_max!" begin
                ys_ = [3 3 4 4 5;
                       5 5 6 6 7]
                @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_max!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_max!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:max, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_min!" begin
                ys_ = [1 1 1 1 1;
                       1 1 1 1 1]
                @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_min!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_min!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:min, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_mul!" begin
                ys_ = [3 3 4 4 5;
                       5 5 6 6 7]
                @test scatter_mul!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mul!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mul!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:mul, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end

    for T = [Float16, Float32, Float64, BigFloat, Rational]
        @testset "$T" begin
            PT = promote_type(T, Float64)
            @testset "scatter_div!" begin
                us_div = us .* 2
                ys_ = [0.75 0.75 0.25 1. 1.25;
                       1.25 1.25 0.375 1.5 1.75]
                @test scatter_div!(T.(copy(ys)), T.(us_div), xs) == T.(ys_)
                @test scatter_div!(T.(copy(ys)), us_div, xs) == PT.(ys_)
                @test scatter_div!(copy(ys), T.(us_div), xs) == PT.(ys_)
                @test scatter!(:div, T.(copy(ys)), T.(us_div), xs) == T.(ys_)
            end

            @testset "scatter_mean!" begin
                ys_ = [4. 4. 5. 5. 6.;
                       6. 6. 7. 7. 8.]
                @test scatter_mean!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mean!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mean!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:mean, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end
end


# StaticArray
ys = @MMatrix [3 3 4 4 5; 5 5 6 6 7]
us = @MArray ones(Int, 2, 3, 4)
xs = @MMatrix [1 2 3 4; 4 2 1 3; 3 5 5 3]
types = [UInt8, UInt16, UInt32, UInt64, UInt128,
         Int8, Int16, Int32, Int64, Int128,
         Float16, Float32, Float64, Rational]

@testset "scatter-staticarray" begin
    for T = types
        @testset "$T" begin
            PT = promote_type(T, Int)
            @testset "scatter_add!" begin
                ys_ = @MMatrix [5 5 8 6 7; 7 7 10 8 9]
                @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_add!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_add!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:add, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_sub!" begin
                ys_ = @MMatrix [1 1 0 2 3; 3 3 2 4 5]
                @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_sub!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_sub!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:sub, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_max!" begin
                ys_ = @MMatrix [3 3 4 4 5; 5 5 6 6 7]
                @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_max!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_max!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:max, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_min!" begin
                ys_ = @MMatrix [1 1 1 1 1; 1 1 1 1 1]
                @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_min!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_min!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:min, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end

            @testset "scatter_mul!" begin
                ys_ = @MMatrix [3 3 4 4 5; 5 5 6 6 7]
                @test scatter_mul!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mul!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mul!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:mul, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end

    for T = [Float16, Float32, Float64, Rational]
        @testset "$T" begin
            PT = promote_type(T, Float64)
            @testset "scatter_div!" begin
                us_div = us .* 2
                ys_ = @MMatrix [0.75 0.75 0.25 1. 1.25; 1.25 1.25 0.375 1.5 1.75]
                @test scatter_div!(T.(copy(ys)), T.(us_div), xs) == T.(ys_)
                @test scatter_div!(T.(copy(ys)), us_div, xs) == PT.(ys_)
                @test scatter_div!(copy(ys), T.(us_div), xs) == PT.(ys_)
                @test scatter!(:div, T.(copy(ys)), T.(us_div), xs) == T.(ys_)
            end

            @testset "scatter_mean!" begin
                ys_ = @MMatrix [4. 4. 5. 5. 6.; 6. 6. 7. 7. 8.]
                @test scatter_mean!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mean!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mean!(copy(ys), T.(us), xs) == PT.(ys_)
                @test scatter!(:mean, T.(copy(ys)), T.(us), xs) == T.(ys_)
            end
        end
    end
end
