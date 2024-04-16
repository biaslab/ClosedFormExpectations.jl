export Logpdf

import Distributions: Distribution, logpdf

struct Logpdf{D}
    dist::D
end

function (f::Logpdf{D})(args...) where {D <: Distribution}
    return logpdf(f.dist, args...;)
end

# function convert(::Type{Logpdf}, fixed_call::Fix1{typeof{logpdf}, D}) where {D <: Distribution}
#     return Logpdf(D)
# end