using Microscaling

import cuTile as ct
import Adapt
using Einops: @reshape, @rearrange

struct F8_4x128TileArray{T,N,A<:ct.TileArray{T},B<:ct.TileArray{T}} <: ct.AbstractTileArray{T,N}
    size::NTuple{N,Int}
    tiles::A
    factors::B
end

const K1, M2, M1 = 4, 4, 32
const KTILE, MTILE = K1, M2 * M1   # 4 scale columns × 128 rows per tile

Base.parent(s::F8_4x128TileArray) = s.tiles
Base.size(s::F8_4x128TileArray) = s.size
Base.eltype(::F8_4x128TileArray{T}) where T = T
Base.ndims(::F8_4x128TileArray{T,N}) where {T,N} = N

function Adapt.adapt_structure(to::ct.KernelAdaptor,
        x::F8_4x128Array{T,N,X}) where {T,N,X<:AbstractArray{T}}
    storage = parent(x)
    tiles = @reshape(storage, "k1 m2 m1 k0 m0 ... -> (k1 m2 m1) k0 m0 ...")
    return F8_4x128TileArray(size(x), Adapt.adapt(to, tiles), Adapt.adapt(to, storage))
end

whole_tiles(shape) = shape[1] % KTILE == 0 && shape[2] % MTILE == 0

function check_request(shape)
    ok(extent, tile) = extent % tile == 0 || (ispow2(extent) && extent < tile)
    ok(shape[1], KTILE) && ok(shape[2], MTILE) || throw(ArgumentError(
        "f8_4x128 requests must be whole tiles or a power of two within one \
         tile along each swizzled axis (tile = $KTILE × $MTILE); got $(shape[1:2])"))
    return nothing
end

function factor_box(extent, index, I)
    offset = (index - 1) * extent
    if extent <= I
        inner_extent = extent
        outer_extent = 1
        inner_index = (offset % I) ÷ extent + 1
        outer_index = offset ÷ I + 1
    else
        inner_extent = I
        outer_extent = extent ÷ I
        inner_index = 1
        outer_index = index
    end
    return (inner_extent, outer_extent), (inner_index, outer_index)
end

function m_box(extent, index)
    offset = (index - 1) * extent
    if extent <= M1
        r = offset % MTILE
        (extent, 1, 1), ((r % M1) ÷ extent + 1, r ÷ M1 + 1, offset ÷ MTILE + 1)
    elseif extent < MTILE
        r = offset % MTILE
        (M1, extent ÷ M1, 1), (1, (r ÷ M1) ÷ (extent ÷ M1) + 1, offset ÷ MTILE + 1)
    else
        (M1, M2, extent ÷ MTILE), (1, 1, index)
    end
end

function factor_request(index, shape)
    (k1e, k0e), (k1i, k0i) = factor_box(shape[1], index[1], K1)
    (m1e, m2e, m0e), (m1i, m2i, m0i) = m_box(shape[2], index[2])
    shape′ = (k1e, m2e, m1e, k0e, m0e, shape[3:end]...)
    index′ = (k1i, m2i, m1i, k0i, m0i, index[3:end]...)
    return index′, shape′
end

function ct.load(arr::F8_4x128TileArray, index, shape; kws...)
    if whole_tiles(shape)
        shape′ = (K1 * M2 * M1, shape[1] ÷ KTILE, shape[2] ÷ MTILE, shape[3:end]...)
        tile′ = ct.load(arr.tiles, (1, index...), shape′; kws...)
        return @rearrange(tile′, "(k1 m2 m1) k0 m0 ... -> (k1 k0) (m1 m2 m0) ..."; k1=K1, m2=M2)
    else
        check_request(shape)
        index′, shape′ = factor_request(index, shape)
        tile′ = ct.load(arr.factors, index′, shape′; kws...)
        return @rearrange(tile′, "k1 m2 m1 k0 m0 ... -> (k1 k0) (m1 m2 m0) ...")
    end
end

function ct.store(arr::F8_4x128TileArray, index, tile; kws...)
    shape = size(tile)
    if whole_tiles(shape)
        tile′ = @rearrange(tile, "(k1 k0) (m1 m2 m0) ... -> (k1 m2 m1) k0 m0 ..."; k1=K1, m2=M2, m1=M1)
        return ct.store(arr.tiles, (1, index...), tile′; kws...)
    else
        check_request(shape)
        index′, shape′ = factor_request(index, shape)
        k1, m2, m1 = shape′
        tile′ = @rearrange(tile, "(k1 k0) (m1 m2 m0) ... -> k1 m2 m1 k0 m0 ..."; k1, m2, m1)
        return ct.store(arr.factors, index′, tile′; kws...)
    end
end
