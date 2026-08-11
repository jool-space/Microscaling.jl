module Microscaling

using Republic

@republic using Microfloats:
    Microfloat,
    Float8_E4M3FN, Float8_E5M2,   # MXFP8
    Float6_E2M3FN, Float6_E3M2FN, # MXFP6
    Float4_E2M1FN,                # MXFP4
    Float8_E8M0FNU                # MX scale

@republic using BitPacking:
    Narrow, bitwidth,
    NarrowArray, NarrowVector, NarrowMatrix

include("blockscaling.jl")
export BlockscaledArray, BlockscaledVector, BlockscaledMatrix
public elements, scales
public block_size, scale_type, element_type

include("sm1xx.jl")
export Sm1xxArray, sm1xx

end
