module Microscaling

using Republic

include("Blockscaling/Blockscaling.jl")
@reexport inherit=:public using .Blockscaling

include("sm1xx.jl")
export Sm1xxArray, sm1xx

include("Narrow/Narrow.jl")
@reexport inherit=:public using .Narrow

@reexport using Microfloats

Narrow.bitwidth(::Type{T}) where T<:Microfloat = Microfloats.bitwidth(T)

end
