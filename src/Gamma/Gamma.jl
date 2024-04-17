import SpecialFunctions: digamma
import Distributions: Gamma, shape, rate

function mean(::ClosedFormExpectation, p::typeof(log), q::Gamma)
    return digamma(shape(q)) + log(scale(q))
end

function mean(::ClosedWilliamsProduct, p::typeof(log), q::Gamma)
    return [
        polygamma(1, shape(q)),
        1/scale(q)
    ]
end