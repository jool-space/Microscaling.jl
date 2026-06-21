using Microscaling.Blockscaling
import Microscaling.Blockscaling: block_size, scale_type, element_type

import cuTile as ct
using cuTile: KernelAdaptor, TileArray, Tile
using Adapt: Adapt, adapt

struct BlockscaledTileArray{T,K,X,P}
    x::X
    p::P
end

function Adapt.adapt_structure(
    to::KernelAdaptor,
    arr::BlockscaledArray{T,N,K}
) where {T,N,K}
    x = Adapt.adapt_structure(to, arr.x)
    p = Adapt.adapt_structure(to, arr.p)
    return BlockscaledTileArray{T,K,typeof(x),typeof(p)}(x, p)
end

Base.size(arr::BlockscaledTileArray, args...) = size(arr.p, args...)
Base.eltype(::BlockscaledTileArray{T}) where T = T
Base.ndims(arr::BlockscaledTileArray) = ndims(arr.p)
block_size(::BlockscaledTileArray{T,K}) where {T,K} = Tuple(K.parameters)
block_size(arr::BlockscaledTileArray, i::Integer) = block_size(arr)[i]
scale_type(arr::BlockscaledTileArray) = eltype(arr.x)
element_type(arr::BlockscaledTileArray) = eltype(arr.p)

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

function ct.load(
    arr::BlockscaledTileArray,
    index, shape;
    x_kws=(;), p_kws=(;)
)
    x = ct.load(arr.x, index, ntuple(Val(ndims(arr))) do i
            k = block_size(arr, i)
            isone(k) ? shape[i] : shape[i] ÷ k
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
