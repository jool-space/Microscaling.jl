@testset "BlockscaledArray" begin
    Random.seed!(1)

    @testset "MXFP8 matrix" begin
        elements = Float8_E4M3FN.(randn(Float32, 64, 8))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-2:2, 2, 8))
        A = BlockscaledArray(scales, elements)
        reference = dequantize(scales, elements, (32, 1))

        @test A isa BlockscaledMatrix{Float32}
        @test size(A) == (64, 8)
        @test block_size(A) == (32, 1)
        @test block_size(A, 1) == 32
        @test scale_type(A) == Float8_E8M0FNU
        @test element_type(A) == Float8_E4M3FN

        @test A == reference
        @test copy(A) == reference
        @test A .* 2 == reference .* 2
        @test sum(A) ≈ sum(reference)
        @test map(abs, A) == abs.(reference)
        @test similar(A, Float64, (4, 2)) isa Matrix{Float64}
        @test occursin("BlockscaledMatrix", sprint(show, MIME("text/plain"), A))
    end

    @testset "MXFP4 with NarrowArray elements" begin
        data = Float4_E2M1FN.(randn(Float32, 64, 8))
        elements = NarrowArray{Float4_E2M1FN}(data)
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-1:1, 2, 8))
        A = BlockscaledArray{Float32}(scales, elements, (32, 1))

        @test A == dequantize(scales, data, (32, 1))
        @test copy(A) == dequantize(scales, data, (32, 1))
    end

    @testset "vector" begin
        elements = Float8_E4M3FN.(randn(Float32, 32))
        scales = Float8_E8M0FNU.([0.5f0])
        v = BlockscaledArray(scales, elements)

        @test v isa BlockscaledVector{Float32}
        @test block_size(v) == (32,)
        @test v == 0.5f0 .* Float32.(elements)
    end

    @testset "dimension-wide scales" begin
        elements = Float8_E4M3FN.(randn(Float32, 16, 4))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-1:1, 1, 4))
        A = BlockscaledArray{Float32}(scales, elements, (:, 1))

        @test block_size(A) == (:, 1)
        @test A == dequantize(scales, elements, (16, 1))
        @test copy(A) == dequantize(scales, elements, (16, 1))
    end

    @testset "element type promotion" begin
        elements = randn(Float32, 8)
        @test eltype(BlockscaledArray(Float16.(2.0f0 .^ (-2:1)), elements)) == Float32
        @test eltype(BlockscaledArray{Float16}(Float16.(2.0f0 .^ (-2:1)), elements)) == Float16
        # Microfloat scale/element pairs promote to an abstract type, so the
        # element type falls back to Float32.
        mx = BlockscaledArray(Float8_E8M0FNU.(2.0f0 .^ (-2:1)), Float8_E4M3FN.(elements))
        @test eltype(mx) == Float32
    end

    @testset "shape validation" begin
        elements = Float8_E4M3FN.(randn(Float32, 64, 8))
        @test_throws DimensionMismatch BlockscaledArray{Float32}(rand(Float32, 2, 8), elements, (16, 1))
        @test_throws DimensionMismatch BlockscaledArray{Float32}(rand(Float32, 2, 8), elements, (:, 1))
        @test_throws DimensionMismatch BlockscaledArray{Float32}(rand(Float32, 2, 4), elements, (32, 1))
    end

    @testset "adapt" begin
        elements = Float8_E4M3FN.(randn(Float32, 64, 8))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-2:2, 2, 8))
        A = BlockscaledArray(scales, elements)
        B = Adapt.adapt(Array, A)

        @test B isa BlockscaledMatrix{Float32}
        @test block_size(B) == block_size(A)
        @test B == A
    end
end
