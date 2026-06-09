module cuTileExt

using Narrow

import cuTile as ct
using cuTile: KernelAdaptor, TileArray
using Adapt: Adapt, adapt

struct ReinterpretTileArray{T,N,A<:TileArray{UInt8,N}}
    eltype::Val{T}
    parent::A
end

Base.eltype(::ReinterpretTileArray{T}) where T = T
Base.ndims(::ReinterpretTileArray{T,N}) where {T,N} = N

function Base.size(arr::ReinterpretTileArray{T,N}, i::Integer) where {T,N}
    ratio = 8 ÷ ct.bitwidth(T)
    return i == 1 ? size(arr.parent, i) * ratio : size(arr.parent, i)
end
Base.size(arr::ReinterpretTileArray{T,N}) where {T,N} = ntuple(i -> size(arr, i), Val(N))

function Adapt.adapt_storage(to::KernelAdaptor, arr::PackedArray)
    return ReinterpretTileArray(
        Val(eltype(arr)),
        Adapt.adapt_storage(to, reinterpret(UInt8, arr))
    )
end

function ct.store(arr::ReinterpretTileArray{T,N}, index, tile; kws...) where {T,N}
    return ct.store(arr.parent, index, reinterpret(UInt8, tile); kws...)
end

function ct.load(
    arr::ReinterpretTileArray{T,N},
    index, shape; kws...
) where {T,N}
    ratio = 8 ÷ ct.bitwidth(T)
    shape′ = ntuple(Val(N)) do i
        i == 1 ? shape[i] ÷ ratio : shape[i]
    end
    byte_tile = ct.load(arr.parent, index, shape′; kws...)
    tile = reinterpret(T, byte_tile)
    return tile
end

end
