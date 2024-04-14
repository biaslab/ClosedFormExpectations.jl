module LogExpectations

import Distributions: mean

export ClosedFormExpectation, meanlog
export ExpLogSquare

"""
    ClosedFormExpectation
"""
struct ClosedFormExpectation end

"""
    meanlog(::ClosedFormExpectation, q, f)

Compute the E_q[log f(x)] where q is a distribution and f is a function.
"""
function meanlog(::ClosedFormExpectation, ::Nothing, ::Nothing) end


# """
#     meanlog(::ClosedFormExpectation, q::ContinuousDistribution, p::ContinuousDistribution)

# Compute E_q[log p(x)] where q and p are both continous distributions from Distributions julia package.
# """
# function meanlog(::ClosedFormExpectation, ::ContinuousDistribution, ::ContinuousDistribution) end


abstract type LogExpression end

"""
    ExpLogSquare(μ, σ)

ExpLogSquare is a type that represents the exp(-(log(x) - μ)^2/(2σ^2)) function.
"""
struct ExpLogSquare <: LogExpression
    μ
    σ
end

function (p::ExpLogSquare)(x)
    return exp(-(log(x) - p.μ)^2/(2*p.σ^2))
end

include("Exponential/Exponential.jl")
include("Normal/UnivariateNormal.jl")

end
