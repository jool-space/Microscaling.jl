# Microscaling.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.jool.space/Microscaling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Microscaling.jl/dev/)
[![Build Status](https://github.com/jool-space/Microscaling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Microscaling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/Microscaling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/Microscaling.jl)

Array types for the [OCP Microscaling Formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):

- `BlockscaledArray` pairs elements of any type with per-block scales; compose with `NarrowArray` to pack sub-byte elements
- `Sm1xxArray` presents the swizzled scale layout used by SM 1xx (Blackwell) tensor cores.
