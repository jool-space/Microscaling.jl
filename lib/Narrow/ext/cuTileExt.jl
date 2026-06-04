module cuTileExt

using Narrow

import cuTile as ct
using cuTile: KernelAdaptor, TileArray
using Adapt: Adapt, adapt

struct ReinterpretTileArray{T,N,A<:TileArray{UInt8,N}}
    parent::A
end

function ReinterpretTileArray{T}(parent::A) where {T,N,A<:TileArray{UInt8,N}}
    return ReinterpretTileArray{T,N,A}(parent)
end

Base.eltype(::ReinterpretTileArray{T}) where T = T
Base.ndims(::ReinterpretTileArray{T,N}) where {T,N} = N

function Adapt.adapt_storage(to::KernelAdaptor, arr::PackedArray)
    return ReinterpretTileArray{T}(
        Adapt.adapt_storage(to, reinterpret(UInt8, arr))
    )
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
