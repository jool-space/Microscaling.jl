using Microscaling

import cuTile as ct
using cuTile: KernelAdaptor, TileArray
using Adapt: Adapt, adapt
using Einops: @rearrange

struct Sm1xxTileArray{T,N,A<:TileArray{T}}
    size::NTuple{N,Int}
    parent::A
end

Base.parent(s::Sm1xxTileArray) = s.parent
Base.size(s::Sm1xxTileArray) = s.size
Base.eltype(::Sm1xxTileArray{T}) where T = T
Base.ndims(::Sm1xxTileArray{T,N}) where {T,N} = N

function Adapt.adapt_structure(to::KernelAdaptor, arr::Sm1xxArray)
    return Sm1xxTileArray(
        size(arr),
        Adapt.adapt_structure(
            to,
            @rearrange(parent(arr), "k1 m2 m1 k0 m0 ... -> (k1 m2 m1) k0 m0 ...")
        )
    )
end

function ct.load(
    arr::Sm1xxTileArray,
    index, shape;
    kws...
)
    k1, m2, m1 = 4, 4, 32
    shape′ = (
        k1 * m2 * m1,
        shape[1] ÷ k1,
        shape[2] ÷ (m2 * m1),
        shape[3:end]...
    )
    tile′ = ct.load(parent(arr), (1, index...), shape′; kws...)
    tile = @rearrange(tile′, "(k1 m2 m1) k0 m0 ... -> (k1 k0) (m1 m2 m0) ..."; k1, m2)
    return tile
end

function ct.store(
    arr::Sm1xxTileArray,
    index, tile;
    kws...
)
    k1, m2, m1 = 4, 4, 32
    tile′ = @rearrange(tile, "(k1 k0) (m1 m2 m0) ... -> (k1 m2 m1) k0 m0 ..."; k1, m2, m1)
    return ct.store(parent(arr), (1, index...), tile′; kws...)
end