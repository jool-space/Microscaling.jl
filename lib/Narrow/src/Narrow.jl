module Narrow

using Republic

@public bitwidth
export NarrowArray, NarrowVector, NarrowMatrix
export narrow
export PackedArray, PackedVector, PackedMatrix

bitwidth(::Type{T}) where T = sizeof(T) * 8
bitwidth(::T) where T = bitwidth(T)
bitwidth(::Type{Bool}) = 1

include("packing.jl")
include("NarrowArray.jl")
include("PackedArray.jl")

end
