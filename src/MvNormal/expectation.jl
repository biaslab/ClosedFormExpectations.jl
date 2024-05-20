import Distributions: Laplace, Normal, Rayleigh, params, cdf, std
import ExponentialFamily: MultivariateNormalDistributionsFamily
using SpecialFunctions: erf, loggamma

function mean(::ClosedFormExpectation, p::Logpdf{T1}, q::T2) where {T1 <: MultivariateNormalDistributionsFamily, T2 <: MultivariateNormalDistributionsFamily}
    -kldivergence(q, p.dist) - entropy(q)
end

function mean(::ClosedFormExpectation, p::Logpdf{T1}, q::T2) where {T1 <: LinearLogGamma, T2 <: MultivariateNormalDistributionsFamily}
    μ, Σ = mean(q), cov(q)
    lin_mean = dot(p.dist.weights, μ)
    lin_var = dot(p.dist.weights, Σ, p.dist.weights)
    return mean(ClosedFormExpectation(), Logpdf(LogGamma(p.dist.α, p.dist.β)), NormalMeanVariance(lin_mean, lin_var))
end