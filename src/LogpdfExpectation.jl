module LogpdfExpectation

import Distributions: mean

export ClosedFormExpectation, meanlog

"""
    ClosedFormExpectation
"""
struct ClosedFormExpectation end

"""
    meanlog(::ClosedFormExpectation, q, f)

Compute the E_q[log f(x)] where q is a distribution and f is a function.
"""
function meanlog(::ClosedFormExpectation, ::Nothing, ::Nothing) end

include("Exponential/Exponential.jl")

end
