import cuDNN
using cuDNN: Graph, tensor, tensor!, matmul!, execute!, is_supported,
    block_scale_quantize!

@assert !isnothing(Base.get_extension(Microscaling, :cuDNNExt))

# The Microscaling cuDNN extension is hooks-only: `tensor!(g, ::BlockscaledArray)`
# adds a block-scaled operand (data + swizzled scale tensors + dequantize node)
# to a graph, and bindings accept the BlockscaledArray itself. This helper is
# the canonical dequantize→matmul graph built from those hooks.
lift3(t) = length(t) == 2 ? (t..., 1) : t

function blockscaled_matmul_graph(W, X, Dtype)
    K, M, batch = lift3(size(W.p))
    N = size(X.p, 2)
    Wt = PermutedDimsArray(W, ndims(W) == 2 ? (2, 1) : (2, 1, 3))
    g = Graph(io_dtype=Dtype, intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, Wt; name="A")
    b = tensor!(g, X; name="B")
    c = tensor!(g; dims=(M, N, batch), dtype=Dtype, output=true, name="C")
    matmul!(g, a, b; c)
    return g
end

function blockscaled_matmul!(C, g, W, X)
    execute!(g, tensor(g, "A.data") => W, tensor(g, "A.scale") => W,
                tensor(g, "B.data") => X, tensor(g, "B.scale") => X,
                tensor(g, "C") => reshape(C, lift3(size(C))))
    return C
end

# cuDNN's block-scale graph ops only have Blackwell engines; the graph builds
# everywhere, so `is_supported` holds the capability claim on any device — a
# wrong gate fails loudly instead of silently skipping.
CC = CUDA.capability(CUDA.device())
cudnn_blockscale_claimed = CC >= v"10.0" && cuDNN.functional()

@testset "cuDNN MXFP8 — dequantize→matmul graph" begin
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

        g = blockscaled_matmul_graph(W, X, Float32)
        @test is_supported(g) == cudnn_blockscale_claimed
        if cudnn_blockscale_claimed
            blockscaled_matmul!(C, g, W, X)
            @test isapprox(Array(C), C_ref; rtol = 1e-5, atol = 1e-5)
        end
    end
end

@testset "cuDNN MXFP8 — output types" begin
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
        C = CUDA.zeros(Dtype, M, N)
        g = blockscaled_matmul_graph(W, X, Dtype)
        if is_supported(g)
            blockscaled_matmul!(C, g, W, X)
            @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
        else
            @test_skip cudnn_blockscale_claimed
        end
    end
end

@testset "cuDNN NVFP4 — dequantize→matmul graph" begin
    Random.seed!(5)

    Scale   = Float8_E4M3FN
    Element = Float4_E2M1FN
    block = 16
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N))

    C_ref = blockscaled_gemm_reference(w_data, w_scale, x_data, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), NarrowArray{Element}(CuArray(w_data)))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), NarrowArray{Element}(CuArray(x_data)))
    C = CUDA.zeros(Float32, M, N)

    g = blockscaled_matmul_graph(W, X, Float32)
    if is_supported(g)
        blockscaled_matmul!(C, g, W, X)
        @test isapprox(Array(C), C_ref; rtol = 1e-4, atol = 1e-4)
    else
        @test_skip cudnn_blockscale_claimed
    end
end

@testset "cuDNN batched MXFP8 — dequantize→matmul graph" begin
    Random.seed!(13)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
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

    g = blockscaled_matmul_graph(W, X, Float32)
    if is_supported(g)
        blockscaled_matmul!(D, g, W, X)
        @test isapprox(Array(D), D_ref; rtol = 1e-5, atol = 1e-5)
    else
        @test_skip cudnn_blockscale_claimed
    end
end

@testset "cuDNN broadcast-batch MXFP8 — shared W, batched X" begin
    # the inference shape: one quantized weight matrix against a batch of
    # activations, batch-1 operand broadcast by the matmul. Engine support for
    # broadcasting a dequantize output is unverified, so this skips (rather
    # than claims) when unsupported.
    Random.seed!(17)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 256, 256, 256
    batch = 4
    K_s = K ÷ block

    w_data  = Element.(randn(K, M))
    x_data  = Element.(randn(K, N, batch))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N, batch))

    D_ref = stack(1:batch) do b
        blockscaled_gemm_reference(
            w_data, w_scale, x_data[:,:,b], x_scale[:,:,b], block)
    end

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_data))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))
    D = CUDA.zeros(Float32, M, N, batch)

    g = Graph(io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, PermutedDimsArray(W, (2, 1)); name="A")    # (M, K, 1)
    b = tensor!(g, X; name="B")                               # (K, N, batch)
    c = tensor!(g; dims=(M, N, batch), dtype=Float32, output=true, name="C")
    matmul!(g, a, b; c)

    # measured supported on sm_121 / cuDNN 9.24
    @test is_supported(g) == cudnn_blockscale_claimed
    if cudnn_blockscale_claimed
        blockscaled_matmul!(D, g, W, X)
        @test isapprox(Array(D), D_ref; rtol = 1e-5, atol = 1e-5)
    end
end

@testset "cuDNN MXFP8 — fused matmul→quantize pipeline" begin
    # the full MXFP8 pipeline in one graph: dequantize both operands, matmul
    # into a virtual f32 tensor, and quantize it back to MXFP8 (elements +
    # swizzled scales written straight into a BlockscaledArray). The
    # dequantized result must reproduce the f32 gemm within block-quantization
    # error; wrong scale bytes (a mislaid swizzle) miss by powers of two, so
    # the loose tolerance still catches layout errors. This is also the
    # positive control for the quantize node: NVIDIA's sample only blesses
    # quantize fused after a matmul, and this test proving the node numerically
    # is what licenses reading the standalone-quantize claim below as "no
    # engine" rather than "wrong graph".
    Random.seed!(19)

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

    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, PermutedDimsArray(W, (2, 1)); name="A")
    b = tensor!(g, X; name="B")
    c = matmul!(g, a, b)                       # virtual (M, N, 1) f32
    ty, tscale = block_scale_quantize!(g, c; block_size=block, block_dim=1,
                                       dtype=Element, scale_dtype=Scale)

    # quantized output blocks along M (dimension 1), feeding a next gemm that
    # reduces over M; scales land in the swizzled layout
    D = BlockscaledArray(
        Sm1xxArray(CuArray{Scale}(undef, 4, 4, 32, (M ÷ block) ÷ 4, N ÷ 128)),
        CuArray{Element}(undef, M, N))

    # measured supported on sm_121 / cuDNN 9.24
    @test is_supported(g) == cudnn_blockscale_claimed
    if cudnn_blockscale_claimed
        execute!(g, tensor(g, "A.data") => W, tensor(g, "A.scale") => W,
                    tensor(g, "B.data") => X, tensor(g, "B.scale") => X,
                    ty => D, tscale => D)
        @test isapprox(Float32.(Array(copy(D))), C_ref; rtol = 0.15, atol = 0.15)
    end
end

@testset "cuDNN MXFP8 — standalone quantize has no engine" begin
    # a lone quantize graph finalizes (the descriptors are valid) but no
    # engine takes it, measured on sm_121 / cuDNN 9.24 — consistent with
    # NVIDIA's sample, which only shows quantize fused after a matmul. If this
    # starts failing on newer cuDNN or other hardware, standalone quantize has
    # gained an engine: flip the claim and add numerics.
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    K, N = 256, 256

    g = Graph()
    tx = tensor!(g; dims=(K, N, 1), dtype=Float32, name="X")
    block_scale_quantize!(g, tx; block_size=32, block_dim=1,
                          dtype=Element, scale_dtype=Scale)
    @test !is_supported(g)
end

@testset "cuDNN — tensor presentation" begin
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    M, N, K = 128, 128, 128

    W = BlockscaledArray(sm1xx(CuArray(Scale.(rand(K ÷ 32, M)))),
                         CuArray(Element.(randn(K, M))))

    g = Graph()
    a = tensor!(g, PermutedDimsArray(W, (2, 1)); name="A")
    b = tensor!(g, W; name="B")
    @test a.dims == [M, K, 1]
    @test b.dims == [K, M, 1]

    # bindings dispatch on dtype: the same BlockscaledArray backs both tensors
    @test cuDNN.checked_array_pointer(tensor(g, "A.data"), W) == pointer(W.p)
    @test cuDNN.checked_array_pointer(tensor(g, "A.scale"), W) == pointer(parent(W.x))
end

@testset "cuDNN — invalid operands are rejected" begin
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    M, K = 128, 128

    g = Graph()

    # raw (unswizzled) scale arrays must be refused before they reach the
    # library, which would silently misread them through its tiled layout
    W_raw = BlockscaledArray(CuArray(Scale.(rand(K ÷ 32, M))),
                             CuArray(Element.(randn(K, M))))
    @test_throws ArgumentError tensor!(g, W_raw; name="A")

    # scale types with no cuDNN block-scale mode are refused, not MethodError'd
    W_f32 = BlockscaledArray{Float32}(CUDA.rand(Float32, 1, M),
                                      CuArray(Element.(randn(K, M))), (:, 1))
    @test_throws ArgumentError tensor!(g, W_f32; name="A")

    # cuDNN matmul has no layout for 4D and beyond
    W4 = BlockscaledArray{Float32}(CUDA.rand(Float32, 1, M, 2, 2),
                                   CuArray(Element.(randn(K, M, 2, 2))), (:, 1, 1, 1))
    @test_throws ArgumentError tensor!(g, W4; name="A")

    # mismatched inner dimensions surface from the graph matmul
    W = BlockscaledArray(sm1xx(CuArray(Scale.(rand(K ÷ 32, M)))),
                         CuArray(Element.(randn(K, M))))
    X_mismatch = BlockscaledArray(sm1xx(CuArray(Scale.(rand(2K ÷ 32, M)))),
                                  CuArray(Element.(randn(2K, M))))
    g2 = Graph()
    a = tensor!(g2, PermutedDimsArray(W, (2, 1)); name="A")
    b = tensor!(g2, X_mismatch; name="B")
    @test_throws DimensionMismatch matmul!(g2, a, b)
end
