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

function mean(::ClosedWilliamsProduct, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return @SVector [
        polygamma(2, shape(q)) + 2 * (digamma(shape(q)) - log(rate(q)))*trigamma(shape(q)),
        2 * (digamma(shape(q)) - log(rate(q))) * rate(q)
    ]
end

function mean(strategy::ClosedFormExpectation, ::ComposedFunction{Power{Val{3}}, typeof(log)}, q::Gamma)
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return polygamma(2, shape(q)) + 3*Elogx * Elog2x - 2*Elogx^3
end

function mean(strategy::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return -1/(2*σ^2)*(μ^2 - 2*μ*Elogx + Elog2x)
end

function mean(strategy::ClosedWilliamsProduct, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return (2*μ*Elogx - Elog2x)/(2*σ^2)
end

function mean(strategy::ClosedFormExpectation, f::Logpdf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    E_logexplogsquare = mean(strategy, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = mean(strategy, log, q)
    return E_logexplogsquare - E_logx - log(σ) - 0.5*log(2pi)
end

function mean(strategy::ClosedWilliamsProduct, f::Logpdf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    E_logexplogsquare = mean(strategy, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = mean(strategy, log, q)
    return E_logexplogsquare - E_logx
end