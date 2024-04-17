using StaticArrays
import SpecialFunctions: digamma, polygamma
import Distributions: Gamma, shape, rate
import LogExpFunctions: xlogx

function mean(::ClosedFormExpectation, p::typeof(log), q::Gamma)
    return digamma(shape(q)) + log(scale(q))
end

function mean(::ClosedWilliamsProduct, p::typeof(log), q::Gamma)
    return @SVector [
        polygamma(1, shape(q)),
        1/scale(q)
    ]
end

function mean(::ClosedFormExpectation, p::typeof(xlogx), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * (digamma(shape(q)+1) - log(rate(q))) 
end