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
    return
end

Random.seed!(1)

@testset for (M, N, K) in (
                (128, 128, 128),
                (256, 384, 512)),
             (TM, TN, TK) in (
                (128, 128, 128),
                (128, 256, 128)),
             (block_size, Scale, Element) in (
                (32, Float8_E8M0FNU, Float8_E4M3FN),
                (32, Float8_E8M0FNU, Float4_E2M1FN),
                (16, Float8_E4M3FN, Float4_E2M1FN)),
             scale_wrapper in (identity, sm1xx)

    K_s = K ÷ block_size

    x_data  = Element.(randn(K, M))
    y_data  = Element.(randn(K, N))
    x_scale = Scale.(rand(K_s, M))
    y_scale = Scale.(rand(K_s, N))

    Z_ref = blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size)

    X_scales = scale_wrapper(CuArray(x_scale))
    Y_scales = scale_wrapper(CuArray(y_scale))

    if Element <: Float4_E2M1FN
        X_elements = Narrow{Element}.(CuArray(x_data))
        Y_elements = Narrow{Element}.(CuArray(y_data))
    else
        X_elements = CuArray(x_data)
        Y_elements = CuArray(y_data)
    end

    X = BlockscaledArray(X_scales, X_elements)
    Y = BlockscaledArray(Y_scales, Y_elements)
    Z = CUDA.zeros(Float32, M, N)

    CUDA.@sync @cuda backend=ct blocks=(cld(M, TM), cld(N, TN)) gemm_agnostic(
        X, Y, Z, ct.Constant(TM), ct.Constant(TN), ct.Constant(TK),
    )

    @test isapprox(Array(Z), Z_ref; rtol = 1e-5, atol = 1e-5)
end
