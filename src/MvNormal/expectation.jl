import Distributions: Laplace, Normal, Rayleigh, params, cdf, std
import ExponentialFamily: MultivariateNormalDistributionsFamily
using SpecialFunctions: erf, loggamma

@supports_expectation function expectation(::ClosedFormExpectation, p::LogpdfOf{T1}, q::T2) where {T1 <: MultivariateNormalDistributionsFamily, T2 <: MultivariateNormalDistributionsFamily}
    -kldivergence(q, p.x) - entropy(q)
end

@supports_expectation function expectation(backend::ClosedFormExpectation, p::LogpdfOf{T1}, q::T2) where {T1 <: LinearLogGamma, T2 <: MultivariateNormalDistributionsFamily}
    μ, Σ = mean(q), cov(q)
    lin_mean = dot(p.x.weights, μ)
    lin_var = dot(p.x.weights, Σ, p.x.weights)
    return expectation(backend, Base.Fix1(logpdf, LogGamma(p.x.α, p.x.β)), NormalMeanVariance(lin_mean, lin_var))
end
