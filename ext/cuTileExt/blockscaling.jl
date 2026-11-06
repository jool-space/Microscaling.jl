using Microscaling
import Microscaling: block_size, elements, scales

import cuTile as ct
import Adapt

struct BlockScaledTileArray{T,N,K,S,E} <: ct.AbstractTileArray{T,N}
    scale::S
    element::E
end

function Adapt.adapt_structure(
    to::ct.KernelAdaptor, arr::BlockScaledArray{T,N,K}
) where {T,N,K}
    any(s -> s isa Colon, block_size(arr)) && throw(ArgumentError("Colon block size is not supported."))
    scale = Adapt.adapt(to, scales(arr))
    element = Adapt.adapt(to, elements(arr))
    return BlockScaledTileArray{T,N,K,typeof(scale),typeof(element)}(scale, element)
end

elements(arr::BlockScaledTileArray) = arr.element
scales(arr::BlockScaledTileArray) = arr.scale

Base.size(arr::BlockScaledTileArray, args...) = size(elements(arr), args...)

block_size(::BlockScaledTileArray{T,N,K}) where {T,N,K} = Tuple(K.parameters)
block_size(arr::BlockScaledTileArray, i::Integer) = block_size(arr)[i]

Base.transpose(arr::BlockScaledTileArray{<:Any,2}) = PermutedTileArray(arr, (2,1))

struct BlockScaledTile{T,S<:ct.Tile,E<:ct.Tile}
    scale::S
    element::E
end

BlockScaledTile{T}(scale, element) where T =
    BlockScaledTile{T,typeof(scale),typeof(element)}(scale, element)

elements(tile::BlockScaledTile) = tile.element
scales(tile::BlockScaledTile) = tile.scale

Base.size(tile::BlockScaledTile) = size(elements(tile))
Base.size(tile::BlockScaledTile, i::Integer) = size(elements(tile), i)
Base.ndims(tile::BlockScaledTile) = ndims(elements(tile))
Base.eltype(::BlockScaledTile{T}) where T = T

function Base.convert(::Type{ct.Tile{T}}, tile::BlockScaledTile{T}) where T
    element, scale = elements(tile), scales(tile)
    inner = ntuple(i -> size(element, i) ÷ size(scale, i), Val(ndims(element)))
    return T.(element) .* T.(repeat(scale; inner))
end

Base.convert(::Type{ct.Tile}, tile::BlockScaledTile{T}) where T = convert(ct.Tile{T}, tile)

function ct.load(
    arr::BlockScaledTileArray,
    index, shape;
    scale_args=(;), element_args=(;), kws...
)
    scale_shape = ntuple(Val(ndims(arr))) do i
        k = block_size(arr, i)
        isone(k) ? shape[i] : shape[i] ÷ k
    end
    scale = ct.load(scales(arr), index, scale_shape; scale_args..., kws...)
    element = ct.load(elements(arr), index, shape; element_args..., kws...)
    return BlockScaledTile{eltype(arr)}(scale, element)
end

function Base.muladd(
    a::BlockScaledTile{T},
    b::BlockScaledTile{T},
    acc::ct.Tile{T}
) where T
    return ct.muladd_scaled(
        elements(a), scales(a),
        elements(b), scales(b),
        acc
    )
end

Broadcast.broadcastable(tile::BlockScaledTile) = convert(ct.Tile, tile)

# `reshape` is excluded because a split can cross a block boundary
# (it needs intent the raw target shape doesn't carry).
Base.transpose(tile::BlockScaledTile{T}) where T =
    BlockScaledTile{T}(transpose(scales(tile)), transpose(elements(tile)))

Base.permutedims(tile::BlockScaledTile{T}, perm) where T =
    BlockScaledTile{T}(permutedims(scales(tile), perm), permutedims(elements(tile), perm))

Base.repeat(tile::BlockScaledTile{T}, counts::Integer...) where T =
    BlockScaledTile{T}(repeat(scales(tile), counts...), repeat(elements(tile), counts...))

Base.repeat(tile::BlockScaledTile{T}; inner = nothing, outer = nothing) where T =
    BlockScaledTile{T}(repeat(scales(tile); inner, outer), repeat(elements(tile); inner, outer))

for op in (:+, :-)
    @eval begin
        Base.$op(a::BlockScaledTile, b::BlockScaledTile) = $op(convert(ct.Tile, a), convert(ct.Tile, b))
        Base.$op(a::BlockScaledTile, b::ct.Tile)         = $op(convert(ct.Tile, a), b)
        Base.$op(a::ct.Tile,         b::BlockScaledTile) = $op(a, convert(ct.Tile, b))
    end
end
