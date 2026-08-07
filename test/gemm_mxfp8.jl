using Einops: @rearrange

const k1 = 4
const m2 = 4
const m1 = 32

unswizzle_4_4_32(tile) = @rearrange(tile, "(k1 m2 m1) k0 m0 -> (k1 k0) (m1 m2 m0)"; k1, m2)

function gemm_mxfp8(X, X_scale, Y, Y_scale, Z, TM::Int, TN::Int, TK::Int)
    i, j = ct.bid(1), ct.bid(2)
    num_k = cld(size(X, 1), TK)
    z = zeros(Float32, (TM, TN))
    for k in Int32(1):Int32(num_k)
        x = ct.load(X, (i, k), (TM, TK), order = (2, 1))
        y = ct.load(Y, (k, j), (TK, TN))
        x_s = ct.load(X_scale, (1, k, i), (k1 * m2 * m1, TK ÷ 128, TM ÷ 128)) |> unswizzle_4_4_32
        y_s = ct.load(Y_scale, (1, k, j), (k1 * m2 * m1, TK ÷ 128, TN ÷ 128)) |> unswizzle_4_4_32
        z = ct.muladd_scaled(x, transpose(x_s), y, y_s, z)
    end
    ct.store(Z, (i, j), z)
    return
end

@testset "MXFP8 GEMM — manual swizzle" begin
    Random.seed!(0)

    Scale   = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block_size = 32
    M, N, K = 512, 512, 512
    K_s = K ÷ block_size
    TM, TN, TK = 128, 128, 128

    x_data  = Element.(randn(K, M))
    y_data  = Element.(randn(K, N))
    x_scale = Scale.(rand(K_s, M))
    y_scale = Scale.(rand(K_s, N))

    Z_ref = blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size)

    X = BlockscaledArray(sm1xx(CuArray(x_scale)), CuArray(x_data))
    Y = BlockscaledArray(sm1xx(CuArray(y_scale)), CuArray(y_data))
    Z = CUDA.zeros(Float32, M, N)

    CUDA.@sync @cuda backend=ct blocks=(cld(M, TM), cld(N, TN)) gemm_mxfp8(
        X.p,
        @rearrange(X.x.x, "k1 m2 m1 k0 m0 -> (k1 m2 m1) k0 m0"),
        Y.p,
        @rearrange(Y.x.x, "k1 m2 m1 k0 m0 -> (k1 m2 m1) k0 m0"),
        Z,
        ct.Constant(TM), ct.Constant(TN), ct.Constant(TK),
    )

    @test isapprox(Array(Z), Z_ref; rtol = 1e-5, atol = 1e-5)
end
