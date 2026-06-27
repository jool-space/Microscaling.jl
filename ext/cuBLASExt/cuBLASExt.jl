module cuBLASExt

# cuBLASLt block-scaled matmul for Microscaling's BlockscaledArray.
#
# FP8 elements (E4M3, E5M2): VEC32_UE8M0, block=32, E8M0 scales.
#   E5M2×E5M2 is not supported; one operand must be E4M3.
# FP4 elements (E2M1): VEC16_UE4M3, block=16, UE4M3 scales.
# Mixing FP8/FP4 scale modes is not supported by cuBLASLt.
# Per-call descriptor build (no plan cache yet).

using LinearAlgebra: LinearAlgebra, Transpose

using Microscaling:
    Sm1xxArray,
    Float8_E4M3FN, Float8_E5M2, Float4_E2M1FN, Float8_E8M0FNU,
    BlockscaledArray, BlockscaledVector, BlockscaledMatrix,
    block_size, scale_type, element_type
import Microscaling: batched_mul!

using BitPacking: NarrowArray

using CUDACore: CUDACore, CuArray, CuMatrix, CuPtr, cudaDataType,
    R_8F_E4M3, R_8F_E5M2, R_4F_E2M1
using cuBLAS: cublasLtHandle_t, cublasLtCreate,
    cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
    cublasLtMatrixLayoutSetAttribute,
    cublasLtMatmulPreference_t, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatmulAlgoGetHeuristic, cublasLtMatmulHeuristicResult_t, cublasLtMatmul,
    CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_COMPUTE_32F,
    CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB,
    CUBLASLT_MATMUL_DESC_POINTER_MODE,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F, CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    CUBLASLT_POINTER_MODE_HOST, CUBLASLT_POINTER_MODE_DEVICE,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET

# ---------------------------------------------------------------------------
# handle (crude per-process global; a per-context HandleCache is a later step)
# ---------------------------------------------------------------------------

const _LT_HANDLE = Ref{cublasLtHandle_t}(cublasLtHandle_t(C_NULL))

function lt_handle()
    if _LT_HANDLE[] == cublasLtHandle_t(C_NULL)
        ref = Ref{cublasLtHandle_t}()
        cublasLtCreate(ref)
        _LT_HANDLE[] = ref[]
    end
    return _LT_HANDLE[]
end

setattr!(desc, attr, val::T) where {T} =
    cublasLtMatmulDescSetAttribute(desc, attr, Ref(val), sizeof(T))
setlayout!(layout, attr, val::T) where {T} =
    cublasLtMatrixLayoutSetAttribute(layout, attr, Ref(val), sizeof(T))

# ---------------------------------------------------------------------------
# glue: Microscaling types -> cuBLASLt enums
# ---------------------------------------------------------------------------

Base.convert(::Type{cudaDataType}, ::Type{Float8_E4M3FN}) = R_8F_E4M3
Base.convert(::Type{cudaDataType}, ::Type{Float8_E5M2})   = R_8F_E5M2
Base.convert(::Type{cudaDataType}, ::Type{Float4_E2M1FN}) = R_4F_E2M1

# block size + scale element -> cuBLASLt scale mode
function scale_mode(A::BlockscaledArray{<:Any,N}) where {N}
    N >= 2 || throw(ArgumentError("scale_mode requires at least 2D"))
    _scale_mode(block_size(A, 1), block_size(A, 2), scale_type(A))
end
function _scale_mode(k::Integer, m::Integer, ::Type{Float8_E8M0FNU})
    k == 32 && m == 1 || throw(ArgumentError("UE8M0 scales expect (32,1) blocks, got ($k,$m)"))
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
end
function _scale_mode(k::Integer, m::Integer, ::Type{Float8_E4M3FN})
    k == 16 && m == 1 || throw(ArgumentError("UE4M3 scales expect (16,1) blocks, got ($k,$m)"))
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
end
function _scale_mode(k::Integer, m::Integer, ::Type{Float32})
    k == 128 || throw(ArgumentError("Float32 scales expect k=128, got $k"))
    m == 1   && return CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
    m == 128 && return CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F
    throw(ArgumentError("Float32 scales expect m=1 or m=128, got $m"))
end
function _scale_mode(::Colon, ::Colon, ::Type{Float32})
    return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F
end

cvoidptr(x) = reinterpret(CuPtr{Cvoid}, pointer(x))
cvoidptr(x::NarrowArray) = cvoidptr(parent(x))
cvoidptr(x::Sm1xxArray) = cvoidptr(parent(x))
elem_ptr(A::BlockscaledArray)  = cvoidptr(A.p)
scale_ptr(A::BlockscaledArray) = cvoidptr(A.x)

# ---------------------------------------------------------------------------
# D = α * Aᵀ * B + β * C, block-scaled
#
# A (K×M[×batch]) and B (K×N[×batch]) are K-major (TN orientation).
# C (M×N[×batch]) is the read-input for β-accumulation.
# D (M×N[×batch]) is the write-output; may alias C.
# ---------------------------------------------------------------------------

const DeviceScalar{T<:Number} = CuArray{T,0}

function _blockscaled_matmul!(D::CuArray, C::CuArray,
                              A::BlockscaledArray, B::BlockscaledArray,
                              α, β)
    K, M = size(A.p, 1), size(A.p, 2)
    Kb, N = size(B.p, 1), size(B.p, 2)
    K == Kb || throw(DimensionMismatch("contraction mismatch: A is $(size(A.p)), B is $(size(B.p))"))

    batched = ndims(A.p) == 3
    batch = batched ? size(A.p, 3) : 1
    if batched
        size(B.p, 3) == batch || throw(DimensionMismatch("batch mismatch: A has $(batch), B has $(size(B.p, 3))"))
        size(D) == (M, N, batch) || throw(DimensionMismatch("D is $(size(D)), expected ($M, $N, $batch)"))
        size(C) == (M, N, batch) || throw(DimensionMismatch("C is $(size(C)), expected ($M, $N, $batch)"))
    else
        size(D) == (M, N) || throw(DimensionMismatch("D is $(size(D)), expected ($M, $N)"))
        size(C) == (M, N) || throw(DimensionMismatch("C is $(size(C)), expected ($M, $N)"))
    end

    Atype = convert(cudaDataType, element_type(A))
    Btype = convert(cudaDataType, element_type(B))
    Ctype = convert(cudaDataType, eltype(C))
    Dtype = convert(cudaDataType, eltype(D))
    smA = scale_mode(A)
    smB = scale_mode(B)

    H = lt_handle()
    md = Ref{cublasLtMatmulDesc_t}()
    cublasLtMatmulDescCreate(md, CUBLAS_COMPUTE_32F, convert(cudaDataType, Float32))
    desc = md[]
    la = Ref{cublasLtMatrixLayout_t}()
    lb = Ref{cublasLtMatrixLayout_t}()
    lc = Ref{cublasLtMatrixLayout_t}()
    ld = Ref{cublasLtMatrixLayout_t}()
    pref = Ref{cublasLtMatmulPreference_t}()
    try
        setattr!(desc, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T)
        setattr!(desc, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N)
        if α isa DeviceScalar
            setattr!(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE)
        end
        setattr!(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, smA)
        setattr!(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, smB)
        setattr!(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_ptr(A))
        setattr!(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_ptr(B))

        cublasLtMatrixLayoutCreate(la, Atype, UInt64(K), UInt64(M), Int64(K))
        cublasLtMatrixLayoutCreate(lb, Btype, UInt64(K), UInt64(N), Int64(K))
        cublasLtMatrixLayoutCreate(lc, Ctype, UInt64(M), UInt64(N), Int64(M))
        cublasLtMatrixLayoutCreate(ld, Dtype, UInt64(M), UInt64(N), Int64(M))

        if batched
            for l in (la, lb, lc, ld)
                setlayout!(l[], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, Cint(batch))
            end
            setlayout!(la[], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, Int64(K * M))
            setlayout!(lb[], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, Int64(K * N))
            setlayout!(lc[], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, Int64(M * N))
            setlayout!(ld[], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, Int64(M * N))
        end

        cublasLtMatmulPreferenceCreate(pref)
        setattr_pref!(pref[], CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, Csize_t(1 << 24))

        res = Vector{cublasLtMatmulHeuristicResult_t}(undef, 1)
        cnt = Ref{Cint}(0)
        cublasLtMatmulAlgoGetHeuristic(H, desc, la[], lb[], ld[], lc[], pref[],
                                       Cint(1), res, cnt)
        cnt[] == 0 && error("cuBLASLt found no algorithm for this block-scaled matmul")

        ws = CuArray{UInt8}(undef, Int(res[1].workspaceSize))
        algo = Ref(res[1].algo)
        if α isa DeviceScalar
            GC.@preserve α β ws begin
                cublasLtMatmul(H, desc, cvoidptr(α),
                    elem_ptr(A), la[], elem_ptr(B), lb[], cvoidptr(β),
                    cvoidptr(C), lc[], cvoidptr(D), ld[],
                    algo, cvoidptr(ws), Csize_t(length(ws)), CUDACore.stream())
            end
        else
            αh = Float32[α]; βh = Float32[β]
            GC.@preserve αh βh ws begin
                cublasLtMatmul(H, desc, convert(Ptr{Cvoid}, pointer(αh)),
                    elem_ptr(A), la[], elem_ptr(B), lb[], convert(Ptr{Cvoid}, pointer(βh)),
                    cvoidptr(C), lc[], cvoidptr(D), ld[],
                    algo, cvoidptr(ws), Csize_t(length(ws)), CUDACore.stream())
            end
        end
    finally
        pref[] == cublasLtMatmulPreference_t(C_NULL) || cublasLtMatmulPreferenceDestroy(pref[])
        ld[]   == cublasLtMatrixLayout_t(C_NULL)     || cublasLtMatrixLayoutDestroy(ld[])
        lc[]   == cublasLtMatrixLayout_t(C_NULL)     || cublasLtMatrixLayoutDestroy(lc[])
        lb[]   == cublasLtMatrixLayout_t(C_NULL)     || cublasLtMatrixLayoutDestroy(lb[])
        la[]   == cublasLtMatrixLayout_t(C_NULL)     || cublasLtMatrixLayoutDestroy(la[])
        cublasLtMatmulDescDestroy(desc)
    end
    return D
end

setattr_pref!(pref, attr, val::T) where {T} =
    cublasLtMatmulPreferenceSetAttribute(pref, attr, Ref(val), sizeof(T))

# ---------------------------------------------------------------------------
# LinearAlgebra entry points
#
# α/β: Number → POINTER_MODE_HOST (cuBLAS sees values, can optimize β=0).
#       DeviceScalar → POINTER_MODE_DEVICE (graph-safe, read at execution time).
#       Mixed host/device not supported by cuBLASLt for block-scaled matmul.
# ---------------------------------------------------------------------------

function LinearAlgebra.mul!(C::CuMatrix, Wt::Transpose{<:Any,<:BlockscaledMatrix},
                            X::BlockscaledMatrix, α::Number, β::Number)
    return _blockscaled_matmul!(C, C, parent(Wt), X, α, β)
end

function LinearAlgebra.mul!(C::CuMatrix, Wt::Transpose{<:Any,<:BlockscaledMatrix},
                            X::BlockscaledMatrix, α::DeviceScalar, β::DeviceScalar)
    return _blockscaled_matmul!(C, C, parent(Wt), X, α, β)
end

function batched_mul!(D::CuArray{<:Any,3},
                                   A::BlockscaledArray{<:Any,3},
                                   B::BlockscaledArray{<:Any,3},
                                   α::Number, β::Number)
    return _blockscaled_matmul!(D, D, A, B, α, β)
end

function batched_mul!(D::CuArray{<:Any,3},
                                   A::BlockscaledArray{<:Any,3},
                                   B::BlockscaledArray{<:Any,3},
                                   α::DeviceScalar, β::DeviceScalar)
    return _blockscaled_matmul!(D, D, A, B, α, β)
end

end # module
