@testset "SwizzledArray" begin
    Random.seed!(1)

    @testset "f8_4x128 round trip" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = f8_4x128(scales)

        @test S == swizzle(scales, :f8_4x128)
        @test S isa SwizzledArray
        @test S isa F8_4x128Array
        @test size(S) == (8, 256)
        @test size(parent(S)) == (4, 4, 32, 2, 2)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "batched" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 4, 128, 3))
        S = swizzle(scales, :f8_4x128)

        @test size(S) == (4, 128, 3)
        @test S == scales
        @test copy(S) == scales
    end

    @testset "wrapping existing storage" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(scales, :f8_4x128)
        W = SwizzledArray(parent(S), :f8_4x128)

        @test W == scales
        @test F8_4x128Array(parent(S)) == scales
    end

    @testset "generic patterns" begin
        x = rand(Float32, 6, 4)
        S = swizzle(x, (:a, (:b, :c)) --> (:b, :a, :c); b = 2)

        @test size(S) == (6, 4)
        @test size(parent(S)) == (2, 6, 2)
        @test S == x
        @test copy(S) == x
        @test SwizzledArray(parent(S), (:a, (:b, :c)) --> (:b, :a, :c)) == x

        # grouped storage side, with a batch ellipsis
        y = rand(Float32, 6, 4, 3)
        T = swizzle(y, ((:a, :b), :c, ..) --> (:c, (:b, :a), ..); a = 2)
        @test size(T) == (6, 4, 3)
        @test size(parent(T)) == (4, 6, 3)
        @test T == y
        @test copy(T) == y
    end

    @testset "structural dispatch" begin
        # factor names are canonicalized out of the type: any naming of the
        # f8_4x128 grouping/permutation with the declared 4/4/32 tile sizes
        # is the same concrete type and dispatches as F8_4x128Array
        x = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(x, ((:a, :b), (:c, :d, :e), ..) --> (:a, :d, :c, :b, :e, ..);
                    a = 4, d = 4, c = 32)
        @test S isa F8_4x128Array
        @test typeof(S) === typeof(swizzle(x, :f8_4x128))
        @test S == x

        # a crossed mapping is a different layout, not the f8 swizzle
        C = swizzle(x, ((:a, :b), (:c, :d, :e), ..) --> (:d, :a, :c, :b, :e, ..);
                    a = 4, d = 4, c = 32)
        @test !(C isa F8_4x128Array)
        @test C == x

        # same structure but different declared tile sizes is not f8 either
        D = swizzle(x, ((:a, :b), (:c, :d, :e), ..) --> (:a, :d, :c, :b, :e, ..);
                    a = 8, d = 4, c = 16)
        @test !(D isa F8_4x128Array)
        @test D == x
    end

    @testset "tile padding" begin
        # rows pad to whole 128-tiles: logical 200 -> physical 256
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 200))
        S = swizzle(scales, :f8_4x128)   # padding is the default

        @test S isa F8_4x128Array
        @test size(S) == (8, 200)
        @test Microscaling.padded_size(S) == (8, 256)
        @test size(parent(S)) == (4, 4, 32, 2, 2)
        @test S == scales
        @test copy(S) == scales
        @test_throws BoundsError S[1, 201]

        # pad=false refuses the same shape
        @test_throws ArgumentError swizzle(scales, :f8_4x128; pad=false)

        # the shorthand forwards kwargs
        @test f8_4x128(scales) == scales
        @test_throws ArgumentError f8_4x128(scales; pad=false)

        # wrapping pre-padded storage takes the logical dims as a kwarg
        W = SwizzledArray(parent(S), :f8_4x128; dims=(8, 200))
        @test size(W) == (8, 200) && W == scales
        @test_throws DimensionMismatch SwizzledArray(parent(S), :f8_4x128;
                                                     dims=(8, 300))

        # scale-column axis pads to whole 4-groups, zero-filled
        x = rand(Float32, 6, 128) .+ 1
        S2 = swizzle(x, :f8_4x128)
        @test size(S2) == (6, 128) && Microscaling.padded_size(S2) == (8, 128)
        @test S2 == x && copy(S2) == x
        @test count(!iszero, parent(S2)) == 6 * 128

        # batch dims pass through unpadded
        b = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 4, 100, 3))
        B = swizzle(b, :f8_4x128)
        @test size(B) == (4, 100, 3) && Microscaling.padded_size(B) == (4, 128, 3)
        @test B == b && copy(B) == b

        # generic patterns pad too
        g = rand(Float32, 5, 3)
        G = swizzle(g, (:a, (:b, :c)) --> (:b, :a, :c); b = 2)
        @test size(G) == (5, 3) && Microscaling.padded_size(G) == (5, 4)
        @test G == g && copy(G) == g
    end

    @testset "as blockscaled scales" begin
        elements = Float8_E4M3FN.(randn(Float32, 256, 256))
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-2:2, 8, 256))
        A = BlockscaledArray(swizzle(scales, :f8_4x128), elements)

        @test block_size(A) == (32, 1)
        @test A == dequantize(scales, elements, (32, 1))
    end

    @testset "validation" begin
        @test_throws ArgumentError swizzle(rand(Float32, 8), :f8_4x128)
        @test_throws ArgumentError swizzle(rand(Float32, 2, 128), :f8_4x128;
                                           pad=false)
        @test_throws ArgumentError swizzle(rand(Float32, 8, 64), :f8_4x128;
                                           pad=false)
        @test_throws ArgumentError SwizzledArray(rand(Float32, 4, 4, 32, 2), :f8_4x128)
        @test_throws DimensionMismatch SwizzledArray(rand(Float32, 2, 4, 32, 2, 2), :f8_4x128)
        @test_throws ArgumentError swizzle(rand(Float32, 4, 128), :nope)
        # mistyped keywords must not be silently swallowed as context
        @test_throws ArgumentError swizzle(rand(Float32, 8, 256), :f8_4x128;
                                           padded=true)
        @test_throws ArgumentError SwizzledArray(rand(Float32, 4, 4, 32, 2, 2),
                                                 :f8_4x128; blocksize=32)
        @test_throws ArgumentError swizzle(rand(Float32, 4, 4), (:a, :b) --> (:a, :c))
        @test_throws ArgumentError swizzle(rand(Float32, 4, 4), (:a, :b, ..) --> (:b, :a))
    end

    @testset "display and pattern recovery" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(scales, :f8_4x128)

        @test Einops.ArrowPattern(S) ==
              (((:k1, :k0), (:m1, :m2, :m0)) --> (:k1, :m2, :m1, :k0, :m0))

        str = summary(S)
        @test startswith(str, "8×256 F8_4x128Array{")   # alias-aware type display
        @test occursin("\"(k1 k0) (m1 m2 m0) -> k1 m2 m1 k0 m0\"", str)
        @test occursin("where k1=4, m1=32, m2=4", str)

        # non-alias patterns elide the Pattern parameters; nothing declared,
        # so no where clause
        G = swizzle(rand(Float32, 6, 4), (:a, :b) --> (:b, :a))
        @test occursin("SwizzledArray{Float32, 2, Pattern{…}, ", summary(G))
        @test occursin("\"a b -> b a\"", summary(G))
        @test !occursin("where", summary(G))

        # batch dims render as a trailing ellipsis
        B = swizzle(Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 4, 128, 3)), :f8_4x128)
        @test occursin("\"(k1 k0) (m1 m2 m0) ... -> k1 m2 m1 k0 m0 ...\"", summary(B))
    end

    @testset "adapt" begin
        scales = Float8_E8M0FNU.(2.0f0 .^ rand(-3:3, 8, 256))
        S = swizzle(scales, :f8_4x128)
        S′ = Adapt.adapt(Array, S)

        @test S′ isa F8_4x128Array
        @test S′ == scales
    end
end

@testset "deprecated names" begin
    s = @test_deprecated sm1xx(rand(Float32, 4, 128))
    @test s isa F8_4x128Array
    @test Base.isdeprecated(Microscaling, :Sm1xxArray)
    @test Microscaling.Sm1xxArray === F8_4x128Array
end
