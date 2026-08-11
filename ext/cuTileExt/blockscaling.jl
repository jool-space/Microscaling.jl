using Microscaling
import Microscaling: block_size, elements, scales

import cuTile as ct
import Adapt

struct BlockscaledTileArray{T,N,K,S,E} <: ct.AbstractTileArray{T,N}
    scale::S
    element::E
end

function Adapt.adapt_structure(
    to::ct.KernelAdaptor, arr::BlockscaledArray{T,N,K}
) where {T,N,K}
    any(s -> s isa Colon, block_size(arr)) && throw(ArgumentError("Colon block size is not supported."))
    scale = Adapt.adapt(to, scales(arr))
    element = Adapt.adapt(to, elements(arr))
    return BlockscaledTileArray{T,N,K,typeof(scale),typeof(element)}(scale, element)
end

elements(arr::BlockscaledTileArray) = arr.element
scales(arr::BlockscaledTileArray) = arr.scale

Base.size(arr::BlockscaledTileArray, args...) = size(elements(arr), args...)

block_size(::BlockscaledTileArray{T,N,K}) where {T,N,K} = Tuple(K.parameters)
block_size(arr::BlockscaledTileArray, i::Integer) = block_size(arr)[i]

Base.transpose(arr::BlockscaledTileArray{<:Any,2}) = PermutedTileArray(arr, (2,1))

struct BlockscaledTile{T,S<:ct.Tile,E<:ct.Tile}
    scale::S
    element::E
end

BlockscaledTile{T}(scale, element) where T =
    BlockscaledTile{T,typeof(scale),typeof(element)}(scale, element)

elements(tile::BlockscaledTile) = tile.element
scales(tile::BlockscaledTile) = tile.scale

Base.size(tile::BlockscaledTile) = size(elements(tile))
Base.size(tile::BlockscaledTile, i::Integer) = size(elements(tile), i)
Base.ndims(tile::BlockscaledTile) = ndims(elements(tile))
Base.eltype(::BlockscaledTile{T}) where T = T

function Base.convert(::Type{ct.Tile{T}}, tile::BlockscaledTile{T}) where T
    element, scale = elements(tile), scales(tile)
    inner = ntuple(i -> size(element, i) ÷ size(scale, i), Val(ndims(element)))
    return T.(element) .* T.(repeat(scale; inner))
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
    scale = ct.load(scales(arr), index, scale_shape; scale_args..., kws...)
    element = ct.load(elements(arr), index, shape; element_args..., kws...)
    return BlockscaledTile{eltype(arr)}(scale, element)
end

function Base.muladd(
    a::BlockscaledTile{T},
    b::BlockscaledTile{T},
    acc::ct.Tile{T}
) where T
    return ct.muladd_scaled(
        elements(a), scales(a),
        elements(b), scales(b),
        acc
    )
end

Broadcast.broadcastable(tile::BlockscaledTile) = convert(ct.Tile, tile)

# `reshape` is excluded because a split can cross a block boundary
# (it needs intent the raw target shape doesn't carry).
Base.transpose(tile::BlockscaledTile{T}) where T =
    BlockscaledTile{T}(transpose(scales(tile)), transpose(elements(tile)))

Base.permutedims(tile::BlockscaledTile{T}, perm) where T =
    BlockscaledTile{T}(permutedims(scales(tile), perm), permutedims(elements(tile), perm))

Base.repeat(tile::BlockscaledTile{T}, counts::Integer...) where T =
    BlockscaledTile{T}(repeat(scales(tile), counts...), repeat(elements(tile), counts...))

Base.repeat(tile::BlockscaledTile{T}; inner = nothing, outer = nothing) where T =
    BlockscaledTile{T}(repeat(scales(tile); inner, outer), repeat(elements(tile); inner, outer))

for op in (:+, :-)
    @eval begin
        Base.$op(a::BlockscaledTile, b::BlockscaledTile) = $op(convert(ct.Tile, a), convert(ct.Tile, b))
        Base.$op(a::BlockscaledTile, b::ct.Tile)         = $op(convert(ct.Tile, a), b)
        Base.$op(a::ct.Tile,         b::BlockscaledTile) = $op(a, convert(ct.Tile, b))
    end
end
