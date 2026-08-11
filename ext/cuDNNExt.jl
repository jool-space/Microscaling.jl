module cuDNNExt

using Microscaling:
    Sm1xxArray,
    Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN,
    BlockscaledArray,
    block_size, scale_type, element_type, elements, scales

using BitPacking: NarrowArray

import cuDNN
using cuDNN:
    Graph, Tensor,
    CUDNN_TENSOR_REORDERING_F8_128x4

blockstr(t::Tuple) = "(" * join((b isa Colon ? ":" : b for b in t), ",") * ")"

# Both gemm operands store K in dimension 1 and the outer M/N in dimension 2;
# dimension 3, when present, is the batch. cuDNN's graph API only has the two
# Blackwell MX recipes — MXFP8 (32-element blocks, UE8M0 scales) and NVFP4
# (16-element blocks, UE4M3 scales); the cuBLASLt-only Float32 modes have no
# cuDNN equivalent.
function mx_block_size(A::BlockscaledArray{<:Any,N}) where {N}
    2 <= N <= 3 || throw(ArgumentError(
        "cuDNN block-scaled matmul takes matrices or 3-dimensional batches; got $(N)D"))
    blk = block_size(A)
    k, outer = blk[1], blk[2]
    N == 3 && blk[3] != 1 && throw(ArgumentError(
        "batched block scales must repeat per batch entry (batch block 1); " *
        "got $(blockstr(blk))"))
    S, E = scale_type(A), element_type(A)
    if S === Float8_E8M0FNU
        k == 32 && outer == 1 || throw(ArgumentError(
            "UE8M0 scales expect (32,1) blocks, got $(blockstr((k, outer)))"))
        E in (Float8_E4M3FN, Float8_E5M2) || throw(ArgumentError(
            "MXFP8 expects Float8_E4M3FN or Float8_E5M2 elements, got $E"))
        return 32
    elseif S === Float8_E4M3FN
        k == 16 && outer == 1 || throw(ArgumentError(
            "UE4M3 scales expect (16,1) blocks, got $(blockstr((k, outer)))"))
        E === Float4_E2M1FN || throw(ArgumentError(
            "NVFP4 expects Float4_E2M1FN elements, got $E"))
        return 16
    end
    throw(ArgumentError(
        "no cuDNN block-scale mode for $S scales with $(blockstr(blk)) blocks"))
end

# dense storage strides of the underlying (unpermuted) array, then present
# both dims and strides through `perm`
function present(dims::NTuple{N,Int}, perm) where {N}
    strides = cumprod((1, Base.front(dims)...))
    return map(i -> dims[i], perm), map(i -> strides[i], perm)
end

"""
    tensor!(g::Graph, A::BlockscaledArray; name)
    tensor!(g::Graph, PermutedDimsArray(A, perm); name)

Add a block-scaled operand to a cuDNN graph: an element tensor `"\$name.element"`, a
scale tensor `"\$name.scale"` in the swizzled 128×4 tile layout, and a
block-scale-dequantize node joining them. Returns the virtual dequantized
tensor.

Tensors are declared at the array's own rank; nothing is lifted. cuDNN's
matmul takes rank-3 `(rows, cols, batch)` operands, so gemm operands are 3-D
BlockscaledArrays with K in dimension 1 (batch extent 1 for a plain gemm),
and the `a` operand of `matmul!` — which wants M first — is passed as
`PermutedDimsArray(A, (2, 1, 3))`.

The bare array is presented in storage order; wrapping it in a
`PermutedDimsArray` presents the same bytes with permuted dimensions.

On `execute!`, bind the storage components: `elements(A)` to the element
tensor and `scales(A)` to the scale tensor.
"""
cuDNN.tensor!(g::Graph, A::BlockscaledArray{<:Any,N}; name::String="") where {N} =
    block_scale_tensors!(g, A, ntuple(identity, N); name)

cuDNN.tensor!(g::Graph,
              Ap::PermutedDimsArray{<:Any,N,perm,<:Any,<:BlockscaledArray{<:Any,N}};
              name::String="") where {N,perm} =
    block_scale_tensors!(g, parent(Ap), perm; name)

function block_scale_tensors!(g::Graph, A::BlockscaledArray{<:Any,N},
                              perm::NTuple{N,Int}; name::String) where {N}
    bs = mx_block_size(A)
    scales(A) isa Sm1xxArray || throw(ArgumentError(
        "cuDNN accesses block scales through a swizzled tiled layout, " *
        "but the scales are a $(nameof(typeof(scales(A)))); wrap them with `sm1xx`"))
    edims, estrides = present(size(elements(A)), perm)
    sdims, sstrides = present(size(scales(A)), perm)
    element = cuDNN.tensor!(g; dims=edims, strides=estrides, dtype=element_type(A),
                            name=name * ".element")
    scale = cuDNN.tensor!(g; dims=sdims, strides=sstrides, dtype=scale_type(A),
                          name=name * ".scale",
                          reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    return cuDNN.block_scale_dequantize!(g, element, scale; block_size=bs, name)
end

# Packed and swizzled storage cannot satisfy the dense layout comparison —
# NarrowArray elements are sub-byte, and an Sm1xxArray's logical presentation
# is not strided (the F8_128x4 reordering carries the real layout) — so
# binding checks element type and count and hands over the parent's pointer.
function cuDNN.binding_pointer(t::Tensor, a::Union{Sm1xxArray,NarrowArray})
    parent(a) isa cuDNN.DenseCuArray || throw(ArgumentError(
        "binding for $(t.name) needs GPU-backed storage, got $(typeof(parent(a)))"))
    cuDNN.graph_dtype(eltype(a)) == t.dtype || throw(ArgumentError(
        "binding for $(t.name) has eltype $(eltype(a)), expected $(t.dtype)"))
    length(a) == prod(t.dims) || throw(DimensionMismatch(
        "binding for $(t.name) has $(length(a)) elements, expected $(prod(t.dims))"))
    return pointer(parent(a))
end

end
