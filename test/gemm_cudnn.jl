import cuDNN
using cuDNN: Graph, tensor, tensor!, matmul!, execute!, is_supported,
    block_scale_quantize!

@assert !isnothing(Base.get_extension(Microscaling, :cuDNNExt))

# The Microscaling cuDNN extension is hooks-only: `tensor!(g, ::BlockscaledArray)`
# adds a block-scaled operand (element + swizzled scale tensors + dequantize node)
# to a graph at the array's own rank, and bindings take the storage components —
# `elements(A)` for the element tensor, `scales(A)` for the scale tensor. Nothing lifts ranks:
# cuDNN matmul operands carry an explicit batch dimension, so gemm operands are
# built 3-D. This helper is the canonical dequantize→matmul graph from those
# hooks.
batch1(a) = reshape(a, size(a)..., 1)   # append an explicit batch-1 axis

# FP4 elements are sub-byte and live bit-packed on the GPU
gpu_elements(a) = eltype(a) === Float4_E2M1FN ?
    NarrowArray{Float4_E2M1FN}(CuArray(a)) : CuArray(a)

function blockscaled_matmul_graph(W, X, Dtype)
    K, M, batch = size(elements(W))
    N = size(elements(X), 2)
    Wt = PermutedDimsArray(W, (2, 1, 3))
    g = Graph(io_dtype=Dtype, intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, Wt; name="A")
    b = tensor!(g, X; name="B")
    c = tensor!(g; dims=(M, N, batch), dtype=Dtype, output=true, name="C")
    matmul!(g, a, b; c)
    return g
end

# C may be 2-D: binding compares memory layouts with singleton dims dropped
function blockscaled_matmul!(C, g, W, X)
    execute!(g, tensor(g, "A.element") => elements(W), tensor(g, "A.scale") => scales(W),
                tensor(g, "B.element") => elements(X), tensor(g, "B.scale") => scales(X),
                tensor(g, "C") => C)
    return C
end

# cuDNN's block-scale graph ops only have Blackwell engines; the graph builds
# everywhere, so `is_supported` holds the capability claim on any device — a
# wrong gate fails loudly instead of silently skipping.
CC = CUDACore.capability(CUDACore.device())
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

        w_element  = Element.(randn(K, M))
        x_element  = Element.(randn(K, N))
        w_scale = Scale.(rand(K_s, M))
        x_scale = Scale.(rand(K_s, N))

        C_ref = blockscaled_gemm_reference(w_element, w_scale, x_element, x_scale, block)

        W = BlockscaledArray(sm1xx(CuArray(batch1(w_scale))), CuArray(batch1(w_element)))
        X = BlockscaledArray(sm1xx(CuArray(batch1(x_scale))), CuArray(batch1(x_element)))
        C = CUDACore.zeros(Float32, M, N)

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

    w_element  = Element.(randn(K, M))
    x_element  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M) / √K)
    x_scale = Scale.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_element, w_scale, x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(batch1(w_scale))), CuArray(batch1(w_element)))
    X = BlockscaledArray(sm1xx(CuArray(batch1(x_scale))), CuArray(batch1(x_element)))

    @testset "Dtype=$Dtype" for Dtype in (Float32, Float16, BFloat16)
        C = CUDACore.zeros(Dtype, M, N)
        g = blockscaled_matmul_graph(W, X, Dtype)
        if is_supported(g)
            blockscaled_matmul!(C, g, W, X)
            @test isapprox(Float32.(Array(C)), C_ref; rtol = 1e-2, atol = 1e-2)
        else
            @test_skip cudnn_blockscale_claimed
        end
    end
end

# every (element A × element B, scale, block) combination cuDNN's block-scale
# matmul recipes accept; MXFP8 e4m3×e4m3 additionally sweeps shapes above
@testset "cuDNN $label — dequantize→matmul graph" for (label, ElemA, ElemB, Scale, block) in (
    ("NVFP4",           Float4_E2M1FN, Float4_E2M1FN, Float8_E4M3FN,  16),
    ("MXFP4",           Float4_E2M1FN, Float4_E2M1FN, Float8_E8M0FNU, 32),
    ("MXFP8 e4m3×e5m2", Float8_E4M3FN, Float8_E5M2,   Float8_E8M0FNU, 32),
    ("MXFP8 e5m2×e4m3", Float8_E5M2,   Float8_E4M3FN, Float8_E8M0FNU, 32),
    ("MXFP8 e5m2×e5m2", Float8_E5M2,   Float8_E5M2,   Float8_E8M0FNU, 32),
)
    Random.seed!(5)

    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_element  = ElemA.(randn(K, M))
    x_element  = ElemB.(randn(K, N))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N))

    C_ref = blockscaled_gemm_reference(w_element, w_scale, x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(batch1(w_scale))), gpu_elements(batch1(w_element)))
    X = BlockscaledArray(sm1xx(CuArray(batch1(x_scale))), gpu_elements(batch1(x_element)))
    C = CUDACore.zeros(Float32, M, N)

    g = blockscaled_matmul_graph(W, X, Float32)
    @test is_supported(g) == cudnn_blockscale_claimed
    if cudnn_blockscale_claimed
        blockscaled_matmul!(C, g, W, X)
        @test isapprox(Array(C), C_ref; rtol = 1e-4, atol = 1e-4)
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

    w_element  = Element.(randn(K, M, batch))
    x_element  = Element.(randn(K, N, batch))
    w_scale = Scale.(rand(K_s, M, batch))
    x_scale = Scale.(rand(K_s, N, batch))

    D_ref = stack(1:batch) do b
        blockscaled_gemm_reference(
            w_element[:,:,b], w_scale[:,:,b],
            x_element[:,:,b], x_scale[:,:,b], block)
    end

    W = BlockscaledArray(sm1xx(CuArray(w_scale)), CuArray(w_element))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_element))
    D = CUDACore.zeros(Float32, M, N, batch)

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

    w_element  = Element.(randn(K, M))
    x_element  = Element.(randn(K, N, batch))
    w_scale = Scale.(rand(K_s, M))
    x_scale = Scale.(rand(K_s, N, batch))

    D_ref = stack(1:batch) do b
        blockscaled_gemm_reference(
            w_element, w_scale, x_element[:,:,b], x_scale[:,:,b], block)
    end

    W = BlockscaledArray(sm1xx(CuArray(batch1(w_scale))), CuArray(batch1(w_element)))
    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_element))
    D = CUDACore.zeros(Float32, M, N, batch)

    g = Graph(io_dtype=Float32, intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, PermutedDimsArray(W, (2, 1, 3)); name="A") # (M, K, 1)
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
    # the loose tolerance still catches layout errors.
    Random.seed!(19)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block = 32
    M, N, K = 256, 256, 256
    K_s = K ÷ block

    w_element  = Element.(randn(K, M))
    x_element  = Element.(randn(K, N))
    w_scale = Scale.(rand(K_s, M) / √K)
    x_scale = Scale.(rand(K_s, N) / √K)

    C_ref = blockscaled_gemm_reference(w_element, w_scale, x_element, x_scale, block)

    W = BlockscaledArray(sm1xx(CuArray(batch1(w_scale))), CuArray(batch1(w_element)))
    X = BlockscaledArray(sm1xx(CuArray(batch1(x_scale))), CuArray(batch1(x_element)))

    g = Graph(intermediate_dtype=Float32, compute_dtype=Float32)
    a = tensor!(g, PermutedDimsArray(W, (2, 1, 3)); name="A")
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
        execute!(g, tensor(g, "A.element") => elements(W), tensor(g, "A.scale") => scales(W),
                    tensor(g, "B.element") => elements(X), tensor(g, "B.scale") => scales(X),
                    ty => elements(D), tscale => scales(D))
        @test isapprox(Float32.(Array(copy(D))), C_ref; rtol = 0.15, atol = 0.15)
    end
end

@testset "cuDNN MXFP8 — standalone quantize" begin
    # a lone quantize graph finalizes (the descriptors are valid); whether
    # any engine takes it is cuDNN's concern, not ours to pin — none does on
    # sm_121 / cuDNN 9.24, consistent with NVIDIA's sample, which only shows
    # quantize fused after a matmul
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    K, N = 256, 256

    g = Graph()
    tx = tensor!(g; dims=(K, N, 1), dtype=Float32, name="X")
    block_scale_quantize!(g, tx; block_size=32, block_dim=1,
                          dtype=Element, scale_dtype=Scale)
    @test is_supported(g) isa Bool
end

@testset "cuDNN — tensor presentation" begin
    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    M, N, K = 128, 128, 128

    W = BlockscaledArray(sm1xx(CuArray(Scale.(rand(K ÷ 32, M)))),
                         CuArray(Element.(randn(K, M))))

    # tensors are declared at the array's own rank: a 2-D operand stays rank 2
    g = Graph()
    a = tensor!(g, PermutedDimsArray(W, (2, 1)); name="A")
    b = tensor!(g, W; name="B")
    @test a.dims == [M, K]
    @test b.dims == [K, M]

    # dense element arrays go through cuDNN's own layout-checked binding (the
    # permuted presentation matches as a memory layout); the swizzled scales
    # go through the extension's Sm1xxArray method
    @test cuDNN.checked_array_pointer(tensor(g, "A.element"), elements(W)) == pointer(elements(W))
    @test cuDNN.checked_array_pointer(tensor(g, "A.scale"), scales(W)) == pointer(parent(scales(W)))

    # a higher rank lifts by trailing singletons, for callers keeping a
    # graph's tensors at one uniform rank
    g5 = Graph()
    d = tensor!(g5, PermutedDimsArray(W, (2, 1)); name="D", rank=3)
    @test d.dims == [M, K, 1]
    @test cuDNN.tensor(g5, "D.scale").dims == [M, K ÷ 32, 1]
    @test_throws ArgumentError tensor!(g5, W; name="E", rank=1)

    # rank is not capped to matmul's 3: block scaling is not just for matmul
    W4 = BlockscaledArray(sm1xx(CuArray(Scale.(rand(K ÷ 32, M, 2, 3)))),
                          CuArray(Element.(randn(K, M, 2, 3))))
    g4 = Graph()
    c = tensor!(g4, W4; name="C")
    @test c.dims == [K, M, 2, 3]
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
    W_f32 = BlockscaledArray{Float32}(CuArray(rand(Float32, 1, M)),
                                      CuArray(Element.(randn(K, M))), (:, 1))
    @test_throws ArgumentError tensor!(g, W_f32; name="A")

    # UE8M0 recipes are block-32 only: block-16 e8m0 graphs never get an
    # engine (verified across output types and shapes), so refuse up front
    W16 = BlockscaledArray(sm1xx(CuArray(Float8_E8M0FNU.(rand(K ÷ 16, M)))),
                           NarrowArray{Float4_E2M1FN}(CuArray(Float4_E2M1FN.(randn(K, M)))))
    @test_throws ArgumentError tensor!(g, W16; name="A")

    # mismatched inner dimensions surface from the graph matmul
    W = BlockscaledArray(sm1xx(CuArray(batch1(Scale.(rand(K ÷ 32, M))))),
                         CuArray(batch1(Element.(randn(K, M)))))
    X_mismatch = BlockscaledArray(sm1xx(CuArray(batch1(Scale.(rand(2K ÷ 32, M))))),
                                  CuArray(batch1(Element.(randn(2K, M)))))
    g2 = Graph()
    a = tensor!(g2, PermutedDimsArray(W, (2, 1, 3)); name="A")
    b = tensor!(g2, X_mismatch; name="B")
    @test_throws DimensionMismatch matmul!(g2, a, b)

    # 2-D operands declare rank-2 tensors, which cuDNN matmul rejects — the
    # batch axis is the caller's to add
    W2 = BlockscaledArray(sm1xx(CuArray(Scale.(rand(K ÷ 32, M)))),
                          CuArray(Element.(randn(K, M))))
    g3 = Graph()
    a2 = tensor!(g3, PermutedDimsArray(W2, (2, 1)); name="A")
    b2 = tensor!(g3, W2; name="B")
    @test_throws ArgumentError matmul!(g3, a2, b2)
end
