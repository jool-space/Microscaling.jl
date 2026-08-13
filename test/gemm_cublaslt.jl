using BFloat16s
using NNlib: batched_transpose

# Each capability section below states where support is *claimed*; inside the
# testsets, `@test matmul_supported(...)` holds the library to that claim, so
# a wrong gate fails loudly on capable hardware instead of erroring (too wide)
# or silently skipping (too narrow).
CC = CUDACore.capability(CUDACore.device())

if CC >= v"10.0"  # Blackwell: MXFP8/NVFP4 block-scaled formats

@testset "cuBLASLt MXFP8 — matmul!(C, W', X)" begin
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
        C = CUDACore.zeros(Float32, M, N)

        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)

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
        C_prev = CuArray(rand(Float32, M, N))
        C_expected = α .* C_ref .+ β .* Array(C_prev)
        C = copy(C_prev)
        @test cuBLASLt.matmul_supported(C, transpose(W), X; α, β)
        cuBLASLt.matmul!(C, transpose(W), X; α, β)
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
        C_prev = CuArray(rand(Float32, M, N))
        C_expected = αv .* C_ref .+ βv .* Array(C_prev)
        C = copy(C_prev)
        @test cuBLASLt.matmul_supported(C, transpose(W), X; α, β)
        cuBLASLt.matmul!(C, transpose(W), X; α, β)
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

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, BFloat16)
        C = CUDACore.zeros(Dtype, M, N)
        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)
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

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, BFloat16)
        C = CUDACore.zeros(Dtype, M, N)
        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)
        @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt NVFP4 — matmul!(C, W', X)" begin
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
        C = CUDACore.zeros(Float32, M, N)

        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)

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

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, BFloat16)
        C = CUDACore.zeros(Dtype, M, N)
        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)
        @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt batched MXFP8 — matmul!" begin
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

    @testset "$label" for (label, Wt) in (
        "batched_transpose" => batched_transpose(W),
        "PermutedDimsArray" => PermutedDimsArray(W, (2, 1, 3)),
    )
        D = CUDACore.zeros(Float32, M, N, batch)
        @test cuBLASLt.matmul_supported(D, Wt, X)
        cuBLASLt.matmul!(D, Wt, X)
        @test isapprox(Array(D), D_ref; rtol = 1e-5, atol = 1e-5)
    end
end

else
@info "skipping MXFP8/NVFP4 block-scaled matmul testsets (CC ≥ 10.0 required, device is $CC)"
end  # Blackwell (CC ≥ 10.0)

if CC >= v"8.9"  # Ada and later: FP8 with tensorwide scalar scales

@testset "cuBLASLt tensorwide FP8 — matmul!(C, W', X)" begin
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
    C = CUDACore.zeros(Float32, M, N)

    @test cuBLASLt.matmul_supported(C, transpose(W), X)
    cuBLASLt.matmul!(C, transpose(W), X)

    @test isapprox(Array(C), C_ref; rtol = 1e-3, atol = 1e-3)
end

else
@info "skipping tensorwide FP8 matmul testset (CC ≥ 8.9 required, device is $CC)"
end  # Ada and later (CC ≥ 8.9)

if v"9.0" <= CC < v"10.0"  # Hopper only: f32 128-block scale modes

# cuBLASLt reads :vec128_f32 scales outer-major: dense outer × K÷128 memory,
# the transpose of the (K÷128, outer) block-scale geometry
outer_major(scale) = transpose(CuArray(permutedims(scale)))

@testset "cuBLASLt VEC128 FP8 — matmul!(C, W', X)" begin
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

        W = BlockscaledArray{Float32}(outer_major(w_scale), CuArray(w_data))
        X = BlockscaledArray{Float32}(outer_major(x_scale), CuArray(x_data))
        C = CUDACore.zeros(Float32, M, N)

        @test cuBLASLt.matmul_supported(C, transpose(W), X)
        cuBLASLt.matmul!(C, transpose(W), X)

        @test isapprox(Array(C), C_ref; rtol = 1e-2, atol = 1e-2)
    end
end

@testset "cuBLASLt BLK128x128 × VEC128 — matmul!(C, W', X)" begin
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
    X = BlockscaledArray{Float32}(outer_major(x_scale), CuArray(x_data))
    C = CUDACore.zeros(Float32, M, N)

    @test cuBLASLt.matmul_supported(C, transpose(W), X)
    cuBLASLt.matmul!(C, transpose(W), X)

    @test isapprox(Array(C), C_ref; rtol = 1e-2, atol = 1e-2)
end

else
@info "skipping VEC128/BLK128x128 matmul testsets (Hopper only, device is $CC)"
end  # Hopper only (sm_90)

# Outer-vec is one of the 12.9 Hopper f32 scaling schemes (sm_89 and sm_121
# both answer matmul_supported = false); per-batch scalar's support surface is
# undocumented, with datacenter Blackwell the remaining untested claim.
if v"9.0" <= CC < v"10.0"

@testset "cuBLASLt outer-vec FP8 — matmul!(C, W', X)" begin
    Random.seed!(14)

    Element = Float8_E4M3FN
    M, N, K = 256, 256, 256

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Float32.(rand(1, M) / √K)
    x_scale = Float32.(rand(1, N) / √K)

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, (K, 1))

    W = BlockscaledArray{Float32}(CuArray(w_scale), CuArray(w_data), (:, 1))
    X = BlockscaledArray{Float32}(CuArray(x_scale), CuArray(x_data), (:, 1))
    C = CUDACore.zeros(Float32, M, N)

    @test cuBLASLt.matmul_supported(C, transpose(W), X)
    cuBLASLt.matmul!(C, transpose(W), X)

    @test isapprox(Array(C), C_ref; rtol = 1e-2, atol = 1e-2)
end

else
@info "skipping outer-vec FP8 matmul testset (Hopper only claimed, device is $CC)"
end

if v"10.0" <= CC < v"12.0" && cuBLASLt.version() >= v"13.1"

@testset "cuBLASLt per-batch scalar FP8 — matmul!" begin
    Random.seed!(15)

    Element = Float8_E4M3FN
    M, N, K, batch = 256, 256, 256, 4

    w_data  = Element.(randn(K, M, batch))
    x_data  = Element.(randn(K, N, batch))
    w_scale = Float32.(rand(1, 1, batch))
    x_scale = Float32.(rand(1, 1, batch))

    D_ref = stack(1:batch) do b
        (w_scale[b] * x_scale[b]) *
            (transpose(Float32.(w_data[:, :, b])) * Float32.(x_data[:, :, b]))
    end

    W = BlockscaledArray{Float32}(CuArray(w_scale), CuArray(w_data), (:, :, 1))
    X = BlockscaledArray{Float32}(CuArray(x_scale), CuArray(x_data), (:, :, 1))
    D = CUDACore.zeros(Float32, M, N, batch)

    @test cuBLASLt.matmul_supported(D, batched_transpose(W), X)
    cuBLASLt.matmul!(D, batched_transpose(W), X)

    @test isapprox(Array(D), D_ref; rtol = 1e-3, atol = 1e-3)
end

else
@info "skipping per-batch scalar FP8 matmul testset (10.0 ≤ CC < 12.0 and cuBLASLt ≥ 13.1 claimed)"
end

@testset "cuBLASLt — unswizzled block scales are rejected" begin
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 128, 128, 128
    K_s = K ÷ block

    w_scale = CuArray(Scale.(rand(K_s, M)))
    x_scale = CuArray(Scale.(rand(K_s, N)))
    w_data  = CuArray(Element.(randn(K, M)))
    x_data  = CuArray(Element.(randn(K, N)))

    W = BlockscaledArray(sm1xx(w_scale), w_data)
    @test cuBLASLt.scale_mode(W) == :vec32_ue8m0

    # raw (unswizzled) scale arrays must be refused before they reach the
    # library, which would silently misread them through its tiled layout
    W_raw = BlockscaledArray(w_scale, w_data)
    X_raw = BlockscaledArray(x_scale, x_data)
    @test_throws ArgumentError cuBLASLt.scale_mode(W_raw)
    C = CUDACore.zeros(Float32, M, N)
    @test_throws ArgumentError cuBLASLt.matmul!(C, transpose(W_raw), X_raw)
end

@testset "cuBLASLt — scale mode mapping" begin
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    M, K, batch = 64, 128, 4

    data(dims...) = CuArray(Element.(randn(dims...)))

    # one Float32 scale per outer element (channelwise/rowwise)
    W = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, M)), data(K, M), (:, 1))
    @test cuBLASLt.scale_mode(W) == :outer_vec_f32

    # one Float32 scalar per batch entry
    Wb = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, 1, batch)),
                                   data(K, M, batch), (:, :, 1))
    @test cuBLASLt.scale_mode(Wb) == :per_batch_scalar_f32

    # a tensorwide scalar spans the batch dimension too
    Wt = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, 1, 1)),
                                   data(K, M, batch), (:, :, :))
    @test cuBLASLt.scale_mode(Wt) == :scalar_f32

    # batched per-matrix modes must repeat their scales per batch entry
    W_shared = BlockscaledArray{Float32}(CuArray(Scale.(rand(K ÷ 32, M, 1))),
                                         data(K, M, batch), (32, 1, :))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_shared)

    # a scalar blocking the batch by anything but 1 or : is meaningless
    W_pair = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, 1, batch ÷ 2)),
                                       data(K, M, batch), (:, :, 2))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_pair)

    # outer-vec scales must be Float32
    W_bad = BlockscaledArray{Float32}(CuArray(Scale.(rand(1, M))), data(K, M), (:, 1))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_bad)

    # scale types with no cuBLASLt mode are refused, not MethodError'd
    W_e5 = BlockscaledArray{Float32}(CuArray(Float8_E5M2.(rand(K ÷ 32, M))),
                                     data(K, M), (32, 1))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_e5)

    # cuBLASLt matmul has no layout for 4D and beyond
    W4 = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, M, 2, 2)),
                                   data(K, M, 2, 2), (:, 1, 1, 1))
    @test_throws ArgumentError cuBLASLt.scale_mode(W4)

    # Hopper f32 layout contracts (per cuBLAS "Scaling factors layouts"):
    # :vec128_f32 scales are outer-major — the transpose of the natural
    # (K÷128, outer) geometry — so a plain K-major array must be refused;
    # the two layouts only coincide for K ≤ 128
    Kb, Mb = 256, 256
    W_vec = BlockscaledArray{Float32}(transpose(CuArray(rand(Float32, Mb, 2))),
                                      data(Kb, Mb))
    @test cuBLASLt.scale_mode(W_vec) == :vec128_f32
    W_kmaj = BlockscaledArray{Float32}(CuArray(rand(Float32, 2, Mb)), data(Kb, Mb))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_kmaj)

    # :blk128x128_f32 scales are K-major with the K÷128 dimension padded to a
    # multiple of 4: a view into padded storage passes, unpadded is refused
    W_pad = BlockscaledArray{Float32}(view(CuArray(rand(Float32, 4, 2)), 1:2, :),
                                      data(Kb, Mb), (128, 128))
    @test cuBLASLt.scale_mode(W_pad) == :blk128x128_f32
    W_nopad = BlockscaledArray{Float32}(CuArray(rand(Float32, 2, 2)),
                                        data(Kb, Mb), (128, 128))
    @test_throws ArgumentError cuBLASLt.scale_mode(W_nopad)
end
