import Distributions: Laplace, Normal, Rayleigh, params, cdf, std
import ExponentialFamily: MultivariateNormalDistributionsFamily
using SpecialFunctions: erf, loggamma

function mean(::ClosedFormExpectation, p::Logpdf{T1}, q::T2) where {T1 <: MultivariateNormalDistributionsFamily, T2 <: MultivariateNormalDistributionsFamily}
    -kldivergence(q, p.dist) - entropy(q)
end
