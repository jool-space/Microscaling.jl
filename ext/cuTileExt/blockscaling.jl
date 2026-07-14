using Microscaling
import Microscaling: block_size

import cuTile as ct
import Adapt

struct BlockscaledTileArray{T,N,K,X,P} <: ct.AbstractTileArray{T,N}
    x::X
    p::P
end

function Adapt.adapt_structure(
    to::ct.KernelAdaptor, arr::BlockscaledArray{T,N,K}
) where {T,N,K}
    any(s -> s isa Colon, block_size(arr)) && throw(ArgumentError("Colon block size is not supported."))
    x = Adapt.adapt(to, arr.x)
    p = Adapt.adapt(to, arr.p)
    return BlockscaledTileArray{T,N,K,typeof(x),typeof(p)}(x, p)
end

Base.size(arr::BlockscaledTileArray, args...) = size(arr.p, args...)

block_size(::BlockscaledTileArray{T,N,K}) where {T,N,K} = Tuple(K.parameters)
block_size(arr::BlockscaledTileArray, i::Integer) = block_size(arr)[i]

struct BlockscaledTile{T,X<:ct.Tile,P<:ct.Tile}
    x::X
    p::P
end

BlockscaledTile{T}(x, p) where T = BlockscaledTile{T,typeof(x),typeof(p)}(x, p)

Base.size(tile::BlockscaledTile) = size(tile.p)
Base.size(tile::BlockscaledTile, i::Integer) = size(tile.p, i)
Base.ndims(tile::BlockscaledTile) = ndims(tile.p)
Base.eltype(::BlockscaledTile{T}) where T = T

function Base.convert(::Type{ct.Tile{T}}, tile::BlockscaledTile{T}) where T
    p, x = tile.p, tile.x
    inner = ntuple(i -> size(p, i) ÷ size(x, i), Val(ndims(p)))
    return T.(p) .* T.(repeat(x; inner))
end

Base.convert(::Type{ct.Tile}, tile::BlockscaledTile{T}) where T = convert(ct.Tile{T}, tile)

function ct.load(
    arr::BlockscaledTileArray,
    index, shape;
    scale_args=(;), element_args=(;), kws...
)
    scale_shape = ntuple(Val(ndims(arr))) do i
        k = block_size(arr, i)
        isone(k) ? shape[i] : shape[i] ÷ k
    end
    x = ct.load(arr.x, index, scale_shape; scale_args..., kws...)
    p = ct.load(arr.p, index, shape; element_args..., kws...)
    return BlockscaledTile{eltype(arr)}(x, p)
end

function Base.muladd(
    a::BlockscaledTile{T},
    b::BlockscaledTile{T},
    acc::ct.Tile{T}
) where T
    return ct.muladd_scaled(
        a.p, a.x,
        b.p, b.x,
        acc
    )
end

Broadcast.broadcastable(tile::BlockscaledTile) = convert(ct.Tile, tile)

# `reshape` is excluded because a split can cross a block boundary
# (it needs intent the raw target shape doesn't carry).
Base.transpose(tile::BlockscaledTile{T}) where T =
    BlockscaledTile{T}(transpose(tile.x), transpose(tile.p))

Base.permutedims(tile::BlockscaledTile{T}, perm) where T =
    BlockscaledTile{T}(permutedims(tile.x, perm), permutedims(tile.p, perm))

Base.repeat(tile::BlockscaledTile{T}, counts::Integer...) where T =
    BlockscaledTile{T}(repeat(tile.x, counts...), repeat(tile.p, counts...))

Base.repeat(tile::BlockscaledTile{T}; inner = nothing, outer = nothing) where T =
    BlockscaledTile{T}(repeat(tile.x; inner, outer), repeat(tile.p; inner, outer))

for op in (:+, :-)
    @eval begin
        Base.$op(a::BlockscaledTile, b::BlockscaledTile) = $op(convert(ct.Tile, a), convert(ct.Tile, b))
        Base.$op(a::BlockscaledTile, b::ct.Tile)         = $op(convert(ct.Tile, a), b)
        Base.$op(a::ct.Tile,         b::BlockscaledTile) = $op(a, convert(ct.Tile, b))
    end
end
