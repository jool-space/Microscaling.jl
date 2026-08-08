@testset "Sm1xxArray" begin
    Random.seed!(1)

    @testset "round trip" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = sm1xx(scales)

        @test S isa Sm1xxArray
        @test size(S) == (8, 256)
        @test size(parent(S)) == (4, 4, 32, 2, 2)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "batched" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 4, 128, 3))
        S = sm1xx(scales)

        @test size(S) == (4, 128, 3)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "as blockscaled scales" begin
        elements = Float8_E4M3FN.(randn(Float32, 256, 256))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-2:2, 8, 256))
        A = BlockscaledArray(sm1xx(scales), elements)

        @test block_size(A) == (32, 1)
        @test A == dequantize(scales, elements, (32, 1))
    end

    @testset "validation" begin
        @test_throws ArgumentError sm1xx(rand(Float32, 8))
        @test_throws ArgumentError sm1xx(rand(Float32, 2, 128))
        @test_throws ArgumentError sm1xx(rand(Float32, 8, 64))
        @test_throws ArgumentError Sm1xxArray(rand(Float32, 4, 4, 32, 2))
        @test_throws ArgumentError Sm1xxArray(rand(Float32, 2, 4, 32, 2, 2))
    end

    @testset "adapt" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = sm1xx(scales)
        S′ = Adapt.adapt(Array, S)

        @test S′ isa Sm1xxArray
        @test S′ == scales
    end
end
