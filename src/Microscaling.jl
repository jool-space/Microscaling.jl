module Microscaling

using Republic

@reexport using Narrow
@reexport using Microfloats
@reexport using Blockscaling

include("Sm1xxArray.jl")
export Sm1xxArray, sm1xx

const ColMajorBlockscaledMatrix{T,A<:BlockscaledMatrix{T,1}} = Union{PermutedDimsArray{T,2,(1,2),(1,2),A}, BlockscaledMatrix{T,1}}
const RowMajorBlockscaledMatrix{T,A<:BlockscaledMatrix{T,1}} = PermutedDimsArray{T,2,(2,1),(2,1),A}

end
