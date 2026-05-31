module MicrofloatsExt

using Narrow
using Microfloats

Narrow.bitwidth(::Type{T}) where T<:Microfloat = Microfloats.bitwidth(T)

end