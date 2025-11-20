import Distributions: Normal, std
import ExponentialFamily: GaussianDistributionsFamily
using StaticArrays
using SpecialFunctions: erfc, erf

function mean(::ClosedWilliamsProduct, p::Abs, q::Normal)
    μ, σ = q.μ, q.σ
    return @SVector [
        erf(μ/(sqrt(2)*σ)),
        sqrt(2/π)*exp(-μ^2/(2*σ^2))
    ]
end

function mean(::ClosedWilliamsProduct, p::Logpdf{NormalType}, q::Normal{T}) where {T, NormalType <: GaussianDistributionsFamily}
    μ_q, σ_q = mean(q), std(q)
    μ_p, σ_p = mean(p.dist), std(p.dist)
    return @SVector [
        (μ_p - μ_q)/σ_p^2,
        -σ_q/σ_p^2
    ]
end

function mean(::ClosedWilliamsProduct, p::Logpdf{Laplace{T}}, q::Normal{T}) where {T}
    (loc, θ) = params(p.dist)
    normal = Normal(mean(q) - loc, std(q))
    abs_mean = mean(ClosedWilliamsProduct(), Abs(), normal)
    return -1/θ * abs_mean
end

function mean(::ClosedWilliamsProduct, f::Logpdf{LogGamma{T}}, q::Normal) where {T}
    α, β = params(f.dist)
    μ, σ = mean(q), std(q)
    # Gradient of E[log p(x)] with respect to [μ, σ]
    # E[log p(x)] = β*μ - exp(μ + σ²/2 - log(α)) - β * log(α) - loggamma(β)
    exp_term = exp(μ + σ^2/2 - log(α))
    ∂_μ = β - exp_term
    ∂_σ = -σ * exp_term
    return @SVector [∂_μ, ∂_σ]
end
