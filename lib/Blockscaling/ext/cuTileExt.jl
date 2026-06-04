module cuTileExt

using Blockscaling

import cuTile as ct
using cuTile: KernelAdaptor, TileArray, Tile
using Adapt: Adapt, adapt
import Blockscaling: block_dim, block_size

struct BlockscaledTileArray{T,BD,F,X,P}
    block_dim::Val{BD}
    format::F
    x::X
    p::P
end

BlockscaledTileArray{T}(block_dim::Val{BD}, format::F, x::X, p::P) where {T,BD,F,X,P} =
    BlockscaledTileArray{T,BD,F,X,P}(block_dim, format, x, p)

function Adapt.adapt_structure(to::KernelAdaptor, arr::BlockscaledArray)
    return BlockscaledTileArray{eltype(arr)}(
        arr.block_dim, arr.format,
        Adapt.adapt_structure(to, arr.x),
        Adapt.adapt_structure(to, arr.p)
    )
end

Base.size(arr::BlockscaledTileArray, args...) = size(arr.p, args...)
Base.eltype(::BlockscaledTileArray{T}) where {T} = T
Base.ndims(arr::BlockscaledTileArray) = ndims(arr.p)
block_dim(::BlockscaledTileArray{T,BD}) where {T,BD} = BD
block_size(arr::BlockscaledTileArray) = block_size(arr.format)

struct BlockscaledTile{T,X<:Tile,P<:Tile}
    eltype::Type{T}
    x::X
    p::P
end

Base.size(tile::BlockscaledTile) = size(tile.p)
Base.size(tile::BlockscaledTile, i::Integer) = size(tile.p, i)
Base.ndims(tile::BlockscaledTile) = ndims(tile.p)
Base.eltype(::BlockscaledTile{T}) where T = T
Base.transpose(tile::BlockscaledTile) =
    BlockscaledTile(tile.eltype, transpose(tile.x), transpose(tile.p))

function ct.load(
    arr::BlockscaledTileArray,
    index, shape;
    x=(;), p=(;)
)
    return BlockscaledTile(
        eltype(arr),
        ct.load(arr.x, index, ntuple(Val(ndims(arr))) do i
                i == block_dim(arr) ? shape[i] ÷ block_size(arr) : shape[i]
            end; x...
        ),
        ct.load(arr.p, index, shape; p...)
    )
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

end
