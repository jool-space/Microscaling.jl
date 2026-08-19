# Microscaling.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Microscaling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Microscaling.jl/dev/)
[![Build Status](https://github.com/jool-space/Microscaling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Microscaling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Microscaling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Microscaling.jl)

Array types for the [OCP Microscaling Formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):

- `BlockscaledArray` pairs elements of any type with per-block scales; compose with `NarrowArray` to pack sub-byte elements
- `SwizzledArray` presents storage rearranged by an Einops pattern at its logical shape; `swizzle(x, :f8_4x128)` — shorthand `f8_4x128(x)` — produces the 128×4 tiled scale layout Blackwell tensor cores read, zero-padding to whole tiles by default.
