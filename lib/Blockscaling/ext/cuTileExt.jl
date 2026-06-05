module cuTileExt

using Blockscaling

import cuTile as ct
using cuTile: KernelAdaptor, TileArray, Tile
using Adapt: Adapt, adapt
import Blockscaling: block_dim, block_size

struct BlockscaledTileArray{T,BD,F,X,P}
    eltype::Val{T}
    block_dim::Val{BD}
    format::F
    x::X
    p::P
end

function Adapt.adapt_structure(to::KernelAdaptor, arr::BlockscaledArray)
    return BlockscaledTileArray(
        Val(eltype(arr)), arr.block_dim, arr.format,
        Adapt.adapt_structure(to, arr.x),
        Adapt.adapt_structure(to, arr.p)
    )
end

Base.size(arr::BlockscaledTileArray, args...) = size(arr.p, args...)
Base.eltype(::BlockscaledTileArray{T}) where T = T
Base.ndims(arr::BlockscaledTileArray) = ndims(arr.p)
block_dim(::BlockscaledTileArray{T,BD}) where {T,BD} = BD
block_size(arr::BlockscaledTileArray) = block_size(arr.format)

struct BlockscaledTile{T,X<:Tile,P<:Tile}
    eltype::Val{T}
    x::X
    p::P
end

Base.size(tile::BlockscaledTile) = size(tile.p)
Base.size(tile::BlockscaledTile, i::Integer) = size(tile.p, i)
Base.ndims(tile::BlockscaledTile) = ndims(tile.p)
Base.eltype(::BlockscaledTile{T}) where T = T
Base.transpose(tile::BlockscaledTile) =
    BlockscaledTile(tile.eltype, transpose(tile.x), transpose(tile.p))

@inline function ct.load(
    arr::BlockscaledTileArray,
    index, shape;
    x_kws=(;), p_kws=(;)
)
    x = ct.load(arr.x, index, ntuple(Val(ndims(arr))) do i
            i == block_dim(arr) ? shape[i] ÷ block_size(arr) : shape[i]
        end; x_kws...
    )
    p = ct.load(arr.p, index, shape; p_kws...)
    return BlockscaledTile(Val(eltype(arr)), x, p)
end

function Base.muladd(
    a::BlockscaledTile,
    b::BlockscaledTile,
    acc::Tile
)
    return ct.muladd_scaled(
        a.p, a.x,
        b.p, b.x,
        acc
    )
end

function Broadcast.broadcastable(tile::BlockscaledTile{T}) where {T}
    p, x = tile.p, tile.x
    inner = ntuple(i -> size(p, i) ÷ size(x, i), Val(ndims(p)))
    return T.(p) .* T.(repeat(x; inner))
end

end
