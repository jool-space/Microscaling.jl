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
block_size
scale_type
element_type
```

## Sm1xx scale layout

```@docs
Sm1xxArray
sm1xx
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
