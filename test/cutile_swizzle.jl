# f8_4x128 scales through cuTile: whole-tile requests ride the merged 512-byte
# tile view; sub-tile requests (powers of two within one tile, at runtime
# tile-aligned positions) box the storage factors. Padded storage needs no
# special handling in either path.

function swz_copy2(S, D, TK::Int, TM::Int)
    i, j = ct.bid(1), ct.bid(2)
    ct.store(D, (i, j), ct.load(S, (i, j), (TK, TM)))
    return
end
function swz_copy3(S, D, TK::Int, TM::Int, TB::Int)
    i, j, b = ct.bid(1), ct.bid(2), ct.bid(3)
    ct.store(D, (i, j, b), ct.load(S, (i, j, b), (TK, TM, TB)))
    return
end

# swizzled → dense, tile by tile over the padded extents; dense must equal the
# logical values with zeros in the pad
function swz_load_roundtrip(K_s, M, TK, TM)
    x = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, K_s, M))
    S = f8_4x128(CuArray(x))
    pk, pm = Microscaling.padded_size(S)
    nk, nm = cld(pk, TK), cld(pm, TM)
    D = CUDACore.zeros(Float8_E8M0FNU, nk * TK, nm * TM)
    CUDACore.@sync @cuda backend=ct blocks=(nk, nm) swz_copy2(
        S, D, ct.Constant(TK), ct.Constant(TM))
    ref = zeros(Float8_E8M0FNU, nk * TK, nm * TM)
    ref[1:K_s, 1:M] .= x
    return Array(D) == ref
end

# dense → swizzled; the storage must match what the host swizzle produces
function swz_store_roundtrip(K_s, M, TK, TM)
    x = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, K_s, M))
    S = f8_4x128(CuArray(x))
    T = f8_4x128(CUDACore.zeros(Float8_E8M0FNU, K_s, M))
    pk, pm = Microscaling.padded_size(S)
    nk, nm = cld(pk, TK), cld(pm, TM)
    Dp = CUDACore.zeros(Float8_E8M0FNU, nk * TK, nm * TM)
    Dp[1:K_s, 1:M] .= CuArray(x)
    CUDACore.@sync @cuda backend=ct blocks=(nk, nm) swz_copy2(
        Dp, T, ct.Constant(TK), ct.Constant(TM))
    return Array(parent(T)) == Array(parent(S))
end

@testset "f8_4x128 cuTile loads and stores" begin
    Random.seed!(0)
    @testset "load (K_s=$K_s, M=$M) as ($TK, $TM)" for (K_s, M, TK, TM) in (
            # whole tiles, unpadded and over padded storage
            (8, 256, 8, 256), (8, 256, 4, 128), (6, 200, 4, 128), (6, 200, 8, 256),
            # sub-tile along m at runtime positions (m1 boxes, then m2 boxes)
            (8, 256, 4, 1), (8, 256, 4, 16), (8, 256, 8, 16), (8, 256, 4, 32), (8, 256, 4, 64),
            # sub-tile along k, with m whole or sub
            (8, 256, 2, 128), (8, 256, 2, 256), (8, 256, 1, 128), (8, 256, 2, 16), (8, 256, 1, 1),
            # activation-shaped: many k tiles, tiny m padded to 128
            (80, 1, 4, 1), (80, 16, 4, 16), (80, 16, 8, 16), (80, 3, 4, 4), (2, 64, 2, 64))
        @test swz_load_roundtrip(K_s, M, TK, TM)
    end

    @testset "store (K_s=$K_s, M=$M) as ($TK, $TM)" for (K_s, M, TK, TM) in (
            (8, 256, 4, 128), (8, 256, 4, 16), (8, 256, 2, 64),
            (80, 16, 4, 16), (80, 1, 4, 1), (6, 200, 4, 128))
        @test swz_store_roundtrip(K_s, M, TK, TM)
    end

    @testset "batch dims pass through" begin
        x = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 128, 3))
        S = f8_4x128(CuArray(x))
        for (TK, TM) in ((4, 128), (4, 16))
            D = CUDACore.zeros(Float8_E8M0FNU, 8, 128, 3)
            CUDACore.@sync @cuda backend=ct blocks=(8 ÷ TK, 128 ÷ TM, 3) swz_copy3(
                S, D, ct.Constant(TK), ct.Constant(TM), ct.Constant(1))
            @test Array(D) == x
        end
    end

    @testset "requests that are neither whole tiles nor within one" begin
        S = f8_4x128(CUDACore.zeros(Float8_E8M0FNU, 8, 256))
        D = CUDACore.zeros(Float8_E8M0FNU, 8, 256)
        @test_throws Exception CUDACore.@sync @cuda backend=ct blocks=(1, 1) swz_copy2(
            S, D, ct.Constant(8), ct.Constant(192))
    end
end
