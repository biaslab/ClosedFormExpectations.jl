import Distributions: Normal, std
import ExponentialFamily: UnivariateNormalDistributionsFamily
using StaticArrays
using SpecialFunctions: erfc, erf

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, p::Abs, q::Normal)
    μ, σ = q.μ, q.σ
    return @SVector [
        erf(μ/(sqrt(2)*σ)),
        sqrt(2/π)*exp(-μ^2/(2*σ^2))
    ]
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, p::LogpdfOf{NormalType}, q::Normal{T}) where {T, NormalType <: UnivariateNormalDistributionsFamily}
    μ_q, σ_q = mean(q), std(q)
    μ_p, σ_p = mean(p.x), std(p.x)
    return @SVector [
        (μ_p - μ_q)/σ_p^2,
        -σ_q/σ_p^2
    ]
end

@supports_expectation_gradient function expectation_gradient(backend::ClosedFormExpectation, p::LogpdfOf{Laplace{T}}, q::Normal{T}) where {T}
    (loc, θ) = params(p.x)
    normal = Normal(mean(q) - loc, std(q))
    abs_mean = expectation_gradient(backend, Abs(), normal)
    return -1/θ * abs_mean
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, f::LogpdfOf{LogGamma{T}}, q::Normal) where {T}
    α, β = params(f.x)
    μ, σ = mean(q), std(q)
    # Gradient of E[log p(x)] with respect to [μ, σ]
    # E[log p(x)] = β*μ - exp(μ + σ²/2 - log(α)) - β * log(α) - loggamma(β)
    exp_term = exp(μ + σ^2/2 - log(α))
    ∂_μ = β - exp_term
    ∂_σ = -σ * exp_term
    return @SVector [∂_μ, ∂_σ]
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, f::LogGammaKernel, q::Normal)
    α, β = params(f)
    μ, σ = mean(q), std(q)
    exp_term = exp(μ + σ^2 / 2 - log(α))
    return @SVector [β - exp_term, -σ * exp_term]
end
