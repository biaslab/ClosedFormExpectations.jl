import Distributions: Laplace, Normal, Rayleigh, params, cdf, std
import ExponentialFamily: GaussianDistributionsFamily
using SpecialFunctions: erf, loggamma

@supports_expectation function expectation(::ClosedFormExpectation, p::LogpdfOf{NormalType}, q::GaussianDistributionsFamily) where {NormalType <: GaussianDistributionsFamily}
    μ_q, σ_q = mean(q), std(q)
    μ_p, σ_p = mean(p.x), std(p.x)
    return - 1/2 * log(2 * π * σ_p^2) - (σ_q^2 + (μ_p - μ_q)^2) / (2 * σ_p^2)
end

@supports_expectation function expectation(backend::ClosedFormExpectation, p::LogpdfOf{Laplace{T}}, q::GaussianDistributionsFamily) where {T}
    (loc, θ_p) = Distributions.params(p.x)
    normal = Normal(mean(q) - loc, std(q))
    return -log(2*θ_p) - θ_p^(-1) * expectation(backend, Abs(), normal)
end 

@supports_expectation function expectation(::ClosedFormExpectation, f::Abs, q::GaussianDistributionsFamily)
    μ, σ = mean(q), std(q)
    return μ*erf(μ/(sqrt(2)*σ)) + sqrt(2/π)*std(q)*exp(-μ^2/(2*σ^2))
end

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{LogGamma{T}}, q::GaussianDistributionsFamily) where {T}
    α, β = Distributions.params(f.x)
    μ, σ = mean(q), std(q)
    return β*μ - exp(μ + σ^2/2 - log(α)) - β * log(α) - loggamma(β)
end