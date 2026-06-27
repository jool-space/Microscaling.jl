module Microscaling

using Republic

@reexport using Microfloats

using BitPacking

include("blockscaling.jl")
export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
export GlobalScaleArray, GlobalScaleVector, GlobalScaleMatrix
public block_size, scale_type, element_type

include("sm1xx.jl")
export Sm1xxArray, sm1xx

function batched_mul! end
export batched_mul!

end
