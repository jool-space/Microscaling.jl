using Microscaling: F8_4x128Array

import cuTile as ct
import Adapt
using Einops: @reshape, @rearrange

struct F8_4x128TileArray{T,N,A<:ct.TileArray{T}} <: ct.AbstractTileArray{T,N}
    size::NTuple{N,Int}
    parent::A
end

Base.parent(s::F8_4x128TileArray) = s.parent
Base.size(s::F8_4x128TileArray) = s.size
Base.eltype(::F8_4x128TileArray{T}) where T = T
Base.ndims(::F8_4x128TileArray{T,N}) where {T,N} = N

function Adapt.adapt_structure(to::ct.KernelAdaptor,
        x::F8_4x128Array{T,N,X}) where {T,N,X<:AbstractArray{T}}
    # for coalesced loads
    x′ = @reshape(parent(x), "k1 m2 m1 k0 m0 ... -> (k1 m2 m1) k0 m0 ...")
    return F8_4x128TileArray(size(x), Adapt.adapt(to, x′))
end

function ct.load(
    arr::F8_4x128TileArray,
    index, shape; kws...
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
    arr::F8_4x128TileArray,
    index, tile;
    kws...
)
    k1, m2, m1 = 4, 4, 32
    tile′ = @rearrange(tile, "(k1 k0) (m1 m2 m0) ... -> (k1 m2 m1) k0 m0 ..."; k1, m2, m1)
    return ct.store(parent(arr), (1, index...), tile′; kws...)
end
