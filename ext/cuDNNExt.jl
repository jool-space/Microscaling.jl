module cuDNNExt

using Microscaling:
    F8_4x128Array, NarrowArray,
    Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, Float4_E2M1FN,
    BlockscaledArray,
    block_size, scale_type, element_type, elements, scales

using cuDNN:
    cuDNN, Graph, Tensor,
    CUDNN_TENSOR_REORDERING_F8_128x4

blockstr(t::Tuple) = "(" * join((b isa Colon ? ":" : b for b in t), ",") * ")"

# cuDNN's graph API only has the Blackwell MX recipes — MXFP8/MXFP4
# (32-element blocks, UE8M0 scales) and NVFP4 (16-element blocks, UE4M3
# scales) — all blocking the innermost storage dimension; the cuBLASLt-only
# Float32 modes have no cuDNN equivalent.
function mx_block_size(A::BlockscaledArray)
    blk = block_size(A)
    k = blk[1]
    all(==(1), Base.tail(blk)) || throw(ArgumentError(
        "cuDNN block scales tile the innermost dimension only; got $(blockstr(blk)) blocks"))
    S, E = scale_type(A), element_type(A)
    if S === Float8_E8M0FNU
        k == 32 || throw(ArgumentError(
            "UE8M0 scales expect 32-element blocks, got $k"))
        E in (Float8_E4M3FN, Float8_E5M2, Float4_E2M1FN) || throw(ArgumentError(
            "UE8M0 scales expect FP8 (E4M3/E5M2) or FP4 (E2M1) elements, got $E"))
        return 32
    elseif S === Float8_E4M3FN
        k == 16 || throw(ArgumentError(
            "UE4M3 scales expect 16-element blocks, got $k"))
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

# Add a block-scaled operand to a cuDNN graph: an element tensor `"\$name.element"`, a
# scale tensor `"\$name.scale"` in the swizzled 128×4 tile layout, and a
# block-scale-dequantize node joining them. Returns the virtual dequantized
# tensor.
#
# Tensors are declared at the array's own rank by default; a higher `rank`
# lifts them by trailing singleton dimensions (free in column-major storage),
# for callers that keep every tensor of a graph at one uniform rank. cuDNN's
# matmul takes rank-3 `(rows, cols, batch)` operands, so gemm operands are 3-D
# BlockscaledArrays with K in dimension 1 (batch extent 1 for a plain gemm),
# and the `a` operand of `matmul!` — which wants M first — is passed as
# `PermutedDimsArray(A, (2, 1, 3))`.
#
# On `execute!`, bind the storage components: `elements(A)` to the element
# tensor and `scales(A)` to the scale tensor.
cuDNN.tensor!(g::Graph, A::BlockscaledArray{<:Any,N};
              name::String="", rank::Integer=N) where {N} =
    block_scale_tensors!(g, A, ntuple(identity, N); name, rank)

cuDNN.tensor!(g::Graph,
              Ap::PermutedDimsArray{<:Any,N,perm,<:Any,<:BlockscaledArray{<:Any,N}};
              name::String="", rank::Integer=N) where {N,perm} =
    block_scale_tensors!(g, parent(Ap), perm; name, rank)

lift(dims, strides, rank) = (
    (dims..., ntuple(_ -> 1, rank - length(dims))...),
    (strides..., ntuple(_ -> max(prod(dims), 1), rank - length(dims))...),
)

function block_scale_tensors!(g::Graph, A::BlockscaledArray{<:Any,N},
                              perm::NTuple{N,Int}; name::String,
                              rank::Integer=N) where {N}
    rank >= N || throw(ArgumentError(
        "cannot declare a $N-dimensional block-scaled operand at rank $rank"))
    bs = mx_block_size(A)
    scales(A) isa F8_4x128Array || throw(ArgumentError(
        "cuDNN accesses block scales through a swizzled tiled layout, " *
        "but the scales are a $(nameof(typeof(scales(A)))); wrap them with `f8_4x128`"))
    edims, estrides = lift(present(size(elements(A)), perm)..., rank)
    sdims, sstrides = lift(present(size(scales(A)), perm)..., rank)
    element = cuDNN.tensor!(g; dims=edims, strides=estrides, dtype=element_type(A),
                            name=name * ".element")
    scale = cuDNN.tensor!(g; dims=sdims, strides=sstrides, dtype=scale_type(A),
                          name=name * ".scale",
                          reordering=CUDNN_TENSOR_REORDERING_F8_128x4)
    return cuDNN.block_scale_dequantize!(g, element, scale; block_size=bs, name)
end

# swizzled scales and sub-byte packed elements both present a layout the dense
# stride comparison cannot express; check what their storage admits
function cuDNN.checked_array_pointer(t::Tensor, a::Union{F8_4x128Array,NarrowArray})
    parent(a) isa cuDNN.DenseCuArray || throw(ArgumentError(
        "binding for $(t.name) needs GPU-backed storage, got $(typeof(parent(a)))"))
    cuDNN.graph_dtype(eltype(a)) == t.dtype || throw(ArgumentError(
        "binding for $(t.name) has eltype $(eltype(a)), expected $(t.dtype)"))
    length(a) == prod(t.dims) || throw(DimensionMismatch(
        "binding for $(t.name) has $(length(a)) elements, expected $(prod(t.dims))"))
    return pointer(parent(a))
end

end
