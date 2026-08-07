module cuBLASLtExt

using LinearAlgebra: LinearAlgebra, Transpose

using Microscaling:
    Sm1xxArray,
    Float8_E4M3FN, Float8_E8M0FNU,
    BlockscaledArray, BlockscaledMatrix,
    block_size, scale_type

using CUDACore: CuArray, CuMatrix
import cuBLASLt

cuBLASLt.ltptr(A::Sm1xxArray) = cuBLASLt.ltptr(parent(A))

cuBLASLt.ltdata(A::BlockscaledArray) = A.p
cuBLASLt.ltscale(A::BlockscaledArray) = A.x

function cuBLASLt.scale_mode(A::BlockscaledArray{<:Any,N}) where {N}
    N >= 2 || throw(ArgumentError("scale_mode requires at least 2D"))
    mode = _scale_mode(block_size(A, 1), block_size(A, 2), scale_type(A))
    check_swizzled(A.x, mode)
    return mode
end

# cuBLASLt accesses block-mode scales through its swizzled 128×4-entry tiled
# layout, but only ever sees a raw pointer, so it cannot verify the layout
# itself: unswizzled scales are silently misread, or read out of bounds. An
# Sm1xxArray is swizzled and tile-padded by construction.
function check_swizzled(x, mode::Symbol)
    mode in (:vec32_ue8m0, :vec16_ue4m3) || return nothing
    x isa Sm1xxArray || throw(ArgumentError(
        "cuBLASLt accesses :$mode scales through a swizzled tiled layout, " *
        "but the scales are a $(nameof(typeof(x))); wrap them with `sm1xx`"))
    return nothing
end

function _scale_mode(k::Integer, m::Integer, ::Type{Float8_E8M0FNU})
    k == 32 && m == 1 || throw(ArgumentError(
        "UE8M0 scales expect (32,1) blocks, got ($k,$m)"))
    return :vec32_ue8m0
end

function _scale_mode(k::Integer, m::Integer, ::Type{Float8_E4M3FN})
    k == 16 && m == 1 || throw(ArgumentError(
        "UE4M3 scales expect (16,1) blocks, got ($k,$m)"))
    return :vec16_ue4m3
end

function _scale_mode(k::Integer, m::Integer, ::Type{Float32})
    k == 128 || throw(ArgumentError("Float32 scales expect k=128, got $k"))
    m == 1 && return :vec128_f32
    m == 128 && return :blk128x128_f32
    throw(ArgumentError("Float32 scales expect m=1 or m=128, got $m"))
end

_scale_mode(::Colon, ::Colon, ::Type{Float32}) = :scalar_f32

const DeviceScalar{T<:Number} = CuArray{T,0}

_mul!(C, Wt, X, α, β) = cuBLASLt.matmul!(C, Wt, X; α, β)

function LinearAlgebra.mul!(C::CuMatrix,
                            Wt::Transpose{<:Any,<:BlockscaledMatrix},
                            X::BlockscaledMatrix,
                            α::Number, β::Number)
    return _mul!(C, Wt, X, α, β)
end

function LinearAlgebra.mul!(C::CuMatrix,
                            Wt::Transpose{<:Any,<:BlockscaledMatrix},
                            X::BlockscaledMatrix,
                            α::DeviceScalar, β::DeviceScalar)
    return _mul!(C, Wt, X, α, β)
end

end
