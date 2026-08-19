```@meta
CurrentModule = Microscaling
```

# Microscaling

Array types for the [OCP Microscaling Formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

## Installation

```julia
using Pkg
Registry.add(url="https://registry.jool.space")
Pkg.add("Microscaling")
```

## Blockscaled arrays

```@docs
BlockscaledArray
BlockscaledVector
BlockscaledMatrix
scales
elements
block_size
scale_type
element_type
```

## Swizzled scale layout

Blackwell block-scaled matmuls read scale factors through a hardware tiled
layout ("128×4 tiled" in NVIDIA's docs, `F8_128x4` in cuDNN, Sm1xx in
CUTLASS); [`swizzle`](@ref) rearranges dense scale arrays into such layouts
and presents them at their logical shape.

```@docs
SwizzledArray
swizzle
f8_4x128
F8_4x128Array
ArrowPattern(::SwizzledArray)
```

## Number formats

MX-relevant types from [Microfloats.jl](https://github.com/jool-space/Microfloats.jl), available as `Microscaling.Float8_E4M3FN` etc.

```@docs
Microfloat
Float8_E4M3FN
Float8_E5M2
Float6_E2M3FN
Float6_E3M2FN
Float4_E2M1FN
Float8_E8M0FNU
```

## Packed storage

Sub-byte element storage from [BitPacking.jl](https://github.com/jool-space/BitPacking.jl), available as `Microscaling.NarrowArray` etc.

```@docs
NarrowArray
Narrow
bitwidth
```
