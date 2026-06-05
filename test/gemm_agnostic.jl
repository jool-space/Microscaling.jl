using Einops: @rearrange

function gemm_agnostic(X, Y, Z, TM::Int, TN::Int, TK::Int)
    i, j = ct.bid(1), ct.bid(2)
    num_k = cld(size(X, 1), TK)
    z = zeros(Float32, (TM, TN))
    for k in Int32(1):Int32(num_k)
        x = ct.load(X, (k, i), (TK, TM))
        y = ct.load(Y, (k, j), (TK, TN))
        z = muladd(transpose(x), y, z)
    end
    ct.store(Z, (i, j), z)
    return nothing
end

@testset "MXFP8 GEMM — BlockscaledArray wrapper" begin
    Random.seed!(0)

    Scale = Float8_E8M0FNU
    Element = Float8_E4M3FN
    block_size = 32
    M, N, K = 256, 384, 512
    K_s = K ÷ block_size
    TM, TN, TK = 128, 128, 128

    format = BlockscalingFormat(block_size, Scale, Element)

    x_data  = Element.(randn(K, M))
    y_data  = Element.(randn(K, N))
    x_scale = Scale.(rand(K_s, M))
    y_scale = Scale.(rand(K_s, N))

    Z_ref = blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size)

    X = BlockscaledArray(format, (CuArray(x_scale)), CuArray(x_data))
    Y = BlockscaledArray(format, (CuArray(y_scale)), CuArray(y_data))
    Z = CUDA.zeros(Float32, M, N)

    CUDA.@sync @cuda backend=ct blocks=(cld(M, TM), cld(N, TN)) gemm_agnostic(
        X, Y, Z, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK),
    )

    @test isapprox(Array(Z), Z_ref; rtol = 1e-5, atol = 1e-5)
end
