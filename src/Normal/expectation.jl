import Distributions: Laplace, Normal, Rayleigh, params, cdf, std
import ExponentialFamily: GaussianDistributionsFamily
using SpecialFunctions: erf

function mean(::ClosedFormExpectation, p::Logpdf{NormalType}, q::GaussianDistributionsFamily) where {NormalType <: GaussianDistributionsFamily}
    μ_q, σ_q = mean(q), std(q)
    μ_p, σ_p = mean(p.dist), std(p.dist)
    return - 1/2 * log(2 * π * σ_p^2) - (σ_q^2 + (μ_p - μ_q)^2) / (2 * σ_p^2)
end

function mean(::ClosedFormExpectation, p::Logpdf{Laplace{T}}, q::GaussianDistributionsFamily) where {T}
    (loc, θ_p) = params(p.dist)
    normal = Normal(mean(q) - loc, std(q))
    return -log(2*θ_p) - θ_p^(-1) * mean(ClosedFormExpectation(), Abs(), normal)
end 

function mean(::ClosedFormExpectation, f::Abs, q::GaussianDistributionsFamily)
    μ, σ = mean(q), std(q)
    return μ*erf(μ/(sqrt(2)*σ)) + sqrt(2/π)*std(q)*exp(-μ^2/(2*σ^2))
end
