export Logpdf

import Distributions: Distribution, logpdf

"""
    Logpdf

A structure to represent the logpdf function of a distribution.
"""
struct Logpdf{D}
    dist::D
end

function (f::Logpdf{D})(args...) where {D <: Distribution}
    return logpdf(f.dist, args...;)
end

mean(::ClosedFormExpectation, f::Base.Fix1{typeof(logpdf), D}, q) where {D} = mean(ClosedFormExpectation(), Logpdf(f.x), q)
