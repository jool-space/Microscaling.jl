module cuDNNExt

using Microscaling:
    Sm1xxArray,
    Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN,
    BlockscaledArray,
    block_size, scale_type, element_type

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

lift3(t::NTuple{2,Int}) = (t..., 1)
# dense storage strides of the underlying (unpermuted) array, then present
# both dims and strides through `perm`, lifted to cuDNN matmul's rank 3
function present(dims::NTuple{2,Int}, perm)
    strides = (1, dims[1])
    return (dims[perm[1]], dims[perm[2]], 1),
           (strides[perm[1]], strides[perm[2]], dims[1] * dims[2])
end
function present(dims::NTuple{3,Int}, perm)
    strides = (1, dims[1], dims[1] * dims[2])
    return map(i -> dims[i], perm), map(i -> strides[i], perm)
end

"""
    tensor!(g::Graph, A::BlockscaledArray; name)
    tensor!(g::Graph, PermutedDimsArray(A, perm); name)

Add a block-scaled operand to a cuDNN graph: a data tensor `"\$name.data"`, a
scale tensor `"\$name.scale"` in the swizzled 128×4 tile layout, and a
block-scale-dequantize node joining them. Returns the virtual dequantized
tensor.

The bare array is presented in storage order; wrapping it in a
`PermutedDimsArray` presents the same bytes with permuted dimensions. Blockscaled
gemm operands store K in dimension 1, so the `a` operand of `matmul!` — which
wants M first — is passed as `PermutedDimsArray(A, (2, 1))` (or `(2, 1, 3)`
when batched).

Bind the `BlockscaledArray` itself to both created tensors on `execute!`.
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
    A.x isa Sm1xxArray || throw(ArgumentError(
        "cuDNN accesses block scales through a swizzled tiled layout, " *
        "but the scales are a $(nameof(typeof(A.x))); wrap them with `sm1xx`"))
    ddims, dstrides = present(size(A.p), perm)
    sdims, sstrides = present(size(A.x), perm)
    data = cuDNN.tensor!(g; dims=ddims, strides=dstrides, dtype=element_type(A),
                         name=name * ".data")
    scale = cuDNN.tensor!(g; dims=sdims, strides=sstrides, dtype=scale_type(A),
                          name=name * ".scale",
                          reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    return cuDNN.block_scale_dequantize!(g, data, scale; block_size=bs, name)
end

# Tensors created by `tensor!(g, ::BlockscaledArray)` present lifted or
# permuted dims over packed (NarrowArray) or swizzled (Sm1xxArray) storage,
# so the dense checks cannot apply; the layout was validated when the graph was
# built. The element and scale types of a block-scale mode always differ, so
# the tensor's dtype picks which buffer backs it.
function cuDNN.checked_array_pointer(t::Tensor, A::BlockscaledArray)
    if t.dtype == cuDNN.graph_dtype(element_type(A))
        buffer, n = A.p, length(A.p)
    elseif t.dtype == cuDNN.graph_dtype(scale_type(A))
        buffer, n = A.x, length(A.x)
    else
        throw(ArgumentError(
            "binding for $(t.name): tensor data type matches neither the element " *
            "nor the scale type of the BlockscaledArray"))
    end
    n == prod(t.dims) || throw(DimensionMismatch(
        "binding for $(t.name) has $n elements, expected $(prod(t.dims))"))
    return device_pointer(buffer)
end

device_pointer(a::cuDNN.DenseCuArray) = pointer(a)
device_pointer(a::NarrowArray) = device_pointer(parent(a))
device_pointer(a::Sm1xxArray) = device_pointer(parent(a))
device_pointer(a) = throw(ArgumentError(
    "block-scaled cuDNN bindings need GPU-backed storage, got $(typeof(a))"))

end
