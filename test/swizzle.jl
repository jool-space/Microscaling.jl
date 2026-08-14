@testset "SwizzledArray" begin
    Random.seed!(1)

    @testset "f8_4x128 round trip" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = f8_4x128(scales)

        @test S == swizzle(scales, :f8_4x128)
        @test S isa SwizzledArray
        @test S isa F8_4x128Array
        @test size(S) == (8, 256)
        @test size(parent(S)) == (4, 4, 32, 2, 2)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "batched" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 4, 128, 3))
        S = swizzle(scales, :f8_4x128)

        @test size(S) == (4, 128, 3)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "wrapping existing storage" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(scales, :f8_4x128)
        W = SwizzledArray(parent(S), :f8_4x128)

        @test W == scales
        @test F8_4x128Array(parent(S)) == scales
    end

    @testset "generic patterns" begin
        x = rand(Float32, 6, 4)
        S = swizzle(x, (:a, (:b, :c)) --> (:b, :a, :c); b = 2)

        @test size(S) == (6, 4)
        @test size(parent(S)) == (2, 6, 2)
        @test S == x
        @test copy(S) == x
        @test SwizzledArray(parent(S), (:a, (:b, :c)) --> (:b, :a, :c)) == x

        # grouped storage side, with a batch ellipsis
        y = rand(Float32, 6, 4, 3)
        T = swizzle(y, ((:a, :b), :c, ..) --> (:c, (:b, :a), ..); a = 2)
        @test size(T) == (6, 4, 3)
        @test size(parent(T)) == (4, 6, 3)
        @test T == y
        @test copy(T) == y
    end

    @testset "as blockscaled scales" begin
        elements = Float8_E4M3FN.(randn(Float32, 256, 256))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-2:2, 8, 256))
        A = BlockscaledArray(swizzle(scales, :f8_4x128), elements)

        @test block_size(A) == (32, 1)
        @test A == dequantize(scales, elements, (32, 1))
    end

    @testset "validation" begin
        @test_throws ArgumentError swizzle(rand(Float32, 8), :f8_4x128)
        @test_throws ArgumentError swizzle(rand(Float32, 2, 128), :f8_4x128)
        @test_throws ArgumentError swizzle(rand(Float32, 8, 64), :f8_4x128)
        @test_throws ArgumentError SwizzledArray(rand(Float32, 4, 4, 32, 2), :f8_4x128)
        @test_throws DimensionMismatch SwizzledArray(rand(Float32, 2, 4, 32, 2, 2), :f8_4x128)
        @test_throws ArgumentError swizzle(rand(Float32, 4, 128), :nope)
        @test_throws ArgumentError swizzle(rand(Float32, 4, 4), (:a, :b) --> (:a, :c))
        @test_throws ArgumentError swizzle(rand(Float32, 4, 4), (:a, :b, ..) --> (:b, :a))
    end

    @testset "adapt" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(scales, :f8_4x128)
        S′ = Adapt.adapt(Array, S)

        @test S′ isa F8_4x128Array
        @test S′ == scales
    end
end

@testset "deprecated names" begin
    s = @test_deprecated sm1xx(rand(Float32, 4, 128))
    @test s isa F8_4x128Array
    @test Base.isdeprecated(Microscaling, :Sm1xxArray)
    @test Microscaling.Sm1xxArray === F8_4x128Array
end
