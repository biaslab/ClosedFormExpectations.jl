using StaticArrays
import SpecialFunctions: trigamma, digamma, polygamma
import Distributions: Gamma, shape, rate
import LogExpFunctions: xlogx

function mean(::ClosedFormExpectation, ::typeof(log), q::Gamma)
    return digamma(shape(q)) + log(scale(q))
end

function mean(::ClosedWilliamsProduct, ::typeof(log), q::Gamma)
    return @SVector [
        polygamma(1, shape(q)),
        1/scale(q)
    ]
end

function mean(::ClosedFormExpectation, ::typeof(xlogx), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * (digamma(shape(q)+1) - log(rate(q))) 
end

function mean(::ClosedFormExpectation, ::typeof(xlog2x), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * ((digamma(shape(q)+1) - log(rate(q)))^2 + trigamma(shape(q)+1))
end

function mean(::ClosedFormExpectation, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return trigamma(shape(q)) + (digamma(shape(q)) - log(rate(q)))^2
end

function mean(strategy::ClosedFormExpectation, ::ComposedFunction{Power{Val{3}}, typeof(log)}, q::Gamma)
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return polygamma(2, shape(q)) + 3*Elogx * Elog2x - 2*Elogx^3
end


function mean(strategy::ClosedFormExpectation, p::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = p.inner.μ, p.inner.σ
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return -1/(2*σ^2)*(μ^2 - 2*μ*Elogx + Elog2x)
end