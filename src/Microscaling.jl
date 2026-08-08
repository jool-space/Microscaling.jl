module Microscaling

using Republic
import Adapt

@reexport using Microfloats

@reexport using BitPacking:
    NarrowArray, NarrowVector, NarrowMatrix,
    Narrow, bitwidth

include("blockscaling.jl")
export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
public block_size, scale_type, element_type

include("sm1xx.jl")
export Sm1xxArray, sm1xx

end
