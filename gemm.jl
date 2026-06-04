using CUDA
import cuTile as ct

using Einops: @rearrange

M, N, K = 3072, 3072, 3072

SCALE_BLOCK_SIZE = 32
K_s = K ÷ SCALE_BLOCK_SIZE
TM, TN, TK = 128, 128, 128

using Microscaling

Scale = Float8_E8M0FNU
Element = Float8_E4M3FN
format = BlockscalingFormat(SCALE_BLOCK_SIZE, Scale, Element)

x = CuArray(randn(Element, K_s * SCALE_BLOCK_SIZE, M));
x_scale = sm1xx(CuArray(rand(Scale, K_s, M)));

y = CuArray(randn(Element, K_s * SCALE_BLOCK_SIZE, N));
y_scale = sm1xx(CuArray(rand(Scale, K_s, N)));

X = BlockscaledArray(format, x_scale, x);
Y = BlockscaledArray(format, y_scale, y);

Z = CUDA.zeros(Float32, M, N);

const k1, m2, m1 = 4, 4, 32

using Einops: @rearrange

unswizzle_4_4_32(tile) = @rearrange(tile, "(k1 m2 m1) k0 m0 -> (k1 k0) (m1 m2 m0)"; k1=4, m2=4)

function mxfp8_gemm(X, X_scale, Y, Y_scale, Z, TM::Int, TN::Int, TK::Int)
    i, j = ct.bid(1), ct.bid(2)
    num_k = cld(size(X, 1), TK)
    z = zeros(Float32, (TM, TN))
    for k in Int32(1):Int32(num_k)
        x = ct.load(X, (k, i), (TK, TM))
        y = ct.load(Y, (k, j), (TK, TN))
        x_s = ct.load(X_scale, (1, k, i), (k1 * m2 * m1, TK ÷ 128, TM ÷ 128)) |> unswizzle_4_4_32
        y_s = ct.load(Y_scale, (1, k, j), (k1 * m2 * m1, TK ÷ 128, TN ÷ 128)) |> unswizzle_4_4_32
        z = ct.muladd_scaled(transpose(x), transpose(x_s), y, y_s, z)
    end
    ct.store(Z, (i, j), z)
    return nothing
end

bench = @be CUDA.@sync @cuda backend=ct blocks=(cld(M, TM), cld(N, TN)) mxfp8_gemm(
    X.p,
    @rearrange(X.x.x, "k1 m2 m1 k0 m0 -> (k1 m2 m1) k0 m0"),
    Y.p,
    @rearrange(Y.x.x, "k1 m2 m1 k0 m0 -> (k1 m2 m1) k0 m0"),
    Z,
    ct.Constant(TM),
    ct.Constant(TN),
    ct.Constant(TK)
)

@show M*N*K*2 / minimum(x -> x.time, bench.samples)