using LinearAlgebra: mul!

@testset "cuBLASLt MXFP8 — mul!(C, W', X)" begin
    Random.seed!(2)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32

    @testset "M=$M, N=$N, K=$K" for (M, N, K) in (
        (128, 128, 128),
        (256, 256, 256),
        (256, 384, 512),
    )
        K_s = K ÷ block

        w_data  = Element.(randn(K, M))
        x_data  = Element.(randn(K, N))
        w_scale = Scale.(rand(K_s, M))
        x_scale = Scale.(rand(K_s, N))

        C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

        W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
        X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))
        C = CUDA.zeros(Float32, M, N)

        mul!(C, transpose(W), X, 1.0f0, 0.0f0)

        @test isapprox(Array(C), C_ref; rtol = 1e-5, atol = 1e-5)
    end
end

@testset "cuBLASLt MXFP8 — α/β accumulation" begin
    Random.seed!(3)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N))

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))

    @testset "α=$α, β=$β" for (α, β) in ((2.0f0, 0.0f0), (1.0f0, 1.0f0), (0.5f0, 0.5f0))
        C_prev = CUDA.rand(Float32, M, N)
        C_expected = α .* C_ref .+ β .* Array(C_prev)
        C = copy(C_prev)
        mul!(C, transpose(W), X, α, β)
        @test isapprox(Array(C), C_expected; rtol = 1e-5, atol = 1e-5)
    end
end

@testset "cuBLASLt MXFP8 — device scalar α/β" begin
    Random.seed!(9)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N))

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))

    @testset "α=$αv, β=$βv (device, device)" for (αv, βv) in ((1.0f0, 0.0f0), (2.0f0, 0.0f0), (0.5f0, 0.5f0))
        α = CuArray(fill(αv))
        β = CuArray(fill(βv))
        C_prev = CUDA.rand(Float32, M, N)
        C_expected = αv .* C_ref .+ βv .* Array(C_prev)
        C = copy(C_prev)
        mul!(C, transpose(W), X, α, β)
        @test isapprox(Array(C), C_expected; rtol = 1e-5, atol = 1e-5)
    end

end

@testset "cuBLASLt MXFP8 — mixed E4M3/E5M2" begin
    Random.seed!(6)

    Scale = Float8_E8M0FNU
    block = 32
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Float8_E5M2.(randn(K, M))
    x_data  = Float8_E4M3FN.(randn(K, N))
    w_scale = Scale.(rand(K_s, M) / √K)
    x_scale = Scale.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, CUDACore.BFloat16)
        C = CUDA.zeros(Dtype, M, N)
        mul!(C, transpose(W), X, 1.0f0, 0.0f0)
        @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt MXFP8 — output types" begin
    Random.seed!(7)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M) / √K)
    x_scale = Scale.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, CUDACore.BFloat16)
        C = CUDA.zeros(Dtype, M, N)
        mul!(C, transpose(W), X, 1.0f0, 0.0f0)
        @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt NVFP4 — mul!(C, W', X)" begin
    Random.seed!(5)

    Scale   = Float8_E4M3FN
    Element = Float4_E2M1FN
    block = 16

    @testset "M=$M, N=$N, K=$K" for (M, N, K) in (
        (128, 128, 128),
        (256, 256, 256),
        (256, 384, 512),
    )
        K_s = K ÷ block

        w_data  = Element.(randn(K, M))
        x_data  = Element.(randn(K, N))
        w_scale = Scale.(rand(K_s, M))
        x_scale = Scale.(rand(K_s, N))

        C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

        W = BlockscaledArray(sm1xx(CuArray(w_scale)), NarrowArray{Element}(CuArray(w_data)))
        X = BlockscaledArray(sm1xx(CuArray(x_scale)), NarrowArray{Element}(CuArray(x_data)))
        C = CUDA.zeros(Float32, M, N)

        mul!(C, transpose(W), X, 1.0f0, 0.0f0)

        @test isapprox(Array(C), C_ref; rtol = 1e-4, atol = 1e-4)
    end
end

@testset "cuBLASLt NVFP4 — output types" begin
    Random.seed!(8)

    Scale   = Float8_E4M3FN
    Element = Float4_E2M1FN
    block = 16
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M) / √K)
    x_scale = Scale.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), NarrowArray{Element}(CuArray(w_data)))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), NarrowArray{Element}(CuArray(x_data)))

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, CUDACore.BFloat16)
        C = CUDA.zeros(Dtype, M, N)
        mul!(C, transpose(W), X, 1.0f0, 0.0f0)
        @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt tensorwide FP8 — mul!(C, W', X)" begin
    Random.seed!(12)

    Element = Float8_E4M3FN
    M, N, K = 256, 256, 256

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Float32[0.5]
    x_scale = Float32[0.25]

    C_ref = Float32.(w_data) .* w_scale[1]
    C_ref = transpose(C_ref) * (Float32.(x_data) .* x_scale[1])

    W = BlockscaledArray{Float32}(CuArray(reshape(w_scale, 1, 1)), CuArray(w_data), (:, :))
    X = BlockscaledArray{Float32}(CuArray(reshape(x_scale, 1, 1)), CuArray(x_data), (:, :))
    C = CUDA.zeros(Float32, M, N)

    mul!(C, transpose(W), X, 1.0f0, 0.0f0)

    @test isapprox(Array(C), C_ref; rtol = 1e-5, atol = 1e-5)
end

@testset "cuBLASLt batched MXFP8 — batched_mul!" begin
    Random.seed!(13)

    Element = Float8_E4M3FN
    Scale   = Float8_E8M0FNU
    block = 32
    M, N, K = 256, 256, 256
    batch = 4
    K_s = K ÷ block

    w_data  = Element.(randn(K, M, batch))
    x_data  = Element.(randn(K, N, batch))
    w_scale = Scale.(rand(K_s, M, batch))
    x_scale = Scale.(rand(K_s, N, batch))

    D_ref = stack(1:batch) do b
        blockscaled_gemm_reference(
            w_data[:,:,b], w_scale[:,:,b],
            x_data[:,:,b], x_scale[:,:,b], block)
    end

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))
    D = CUDA.zeros(Float32, M, N, batch)

    batched_mul!(D, W, X, 1.0f0, 0.0f0)

    @test isapprox(Array(D), D_ref; rtol = 1e-5, atol = 1e-5)
end

if CUDA.capability(CUDA.device()).major == 9  # Hopper only

@testset "cuBLASLt VEC128 FP8 — mul!(C, W', X)" begin
    Random.seed!(10)

    Element = Float8_E4M3FN
    block = 128

    @testset "M=$M, N=$N, K=$K" for (M, N, K) in (
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    )
        K_s = K ÷ block

        w_data  = Element.(randn(K, M))
        x_data  = Element.(randn(K, N))
        w_scale = Float32.(rand(K_s, M) / √K)
        x_scale = Float32.(rand(K_s, N) / √K)

        C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

        W = BlockscaledArray{Float32}(CuArray(w_scale), CuArray(w_data))
        X = BlockscaledArray{Float32}(CuArray(x_scale), CuArray(x_data))
        C = CUDA.zeros(Float32, M, N)

        mul!(C, transpose(W), X, 1.0f0, 0.0f0)

        @test isapprox(Array(C), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt BLK128x128 × VEC128 — mul!(C, W', X)" begin
    Random.seed!(11)

    Element = Float8_E4M3FN
    M, N, K = 512, 512, 512
    K_s = K ÷ 128
    M_s = M ÷ 128

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Float32.(rand(K_s, M_s) / √K)
    x_scale = Float32.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, 128;
        x_block_size=(128, 128), y_block_size=128)

    W = BlockscaledArray{Float32}(CuArray(w_scale), CuArray(w_data), (128, 128))
    X = BlockscaledArray{Float32}(CuArray(x_scale), CuArray(x_data))
    C = CUDA.zeros(Float32, M, N)

    mul!(C, transpose(W), X, 1.0f0, 0.0f0)

    @test isapprox(Array(C), C_ref; rtol = 1e-2, atol = 1e-2)
end

end # Hopper only (sm_90)
