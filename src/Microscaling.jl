module Microscaling using Republic

@reexport using Narrow
@reexport using Microfloats
@reexport using Blockscaling

include("ScaleArray.jl")
export ScaleArray
public Dense, Sm1xx
public relayout

# DenseBlockscaledArray?

const ColMajorBlockscaledMatrix{T,A<:BlockscaledMatrix{T,1}} = Union{PermutedDimsArray{T,2,(1,2),(1,2),A}, BlockscaledMatrix{T,1}}
const RowMajorBlockscaledMatrix{T,A<:BlockscaledMatrix{T,1}} = PermutedDimsArray{T,2,(2,1),(2,1),A}

end
