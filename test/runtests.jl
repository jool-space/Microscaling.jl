using Microscaling
using Test

using CUDA
import cuTile as ct
import CUDACore
using Random

function blockscaled_gemm_reference(x_data, x_scale, y_data, y_scale, block_size)
    expand(s) = repeat(s, inner = (block_size, 1))
    dqX = Float32.(x_data) .* Float32.(expand(x_scale))
    dqY = Float32.(y_data) .* Float32.(expand(y_scale))
    return transpose(dqX) * dqY
end

@testset "Microscaling.jl" begin
    if CUDA.functional()
        include("gemm_agnostic.jl")
        include("gemm_mxfp8.jl")
        include("gemm_cublaslt.jl")
    end
end
