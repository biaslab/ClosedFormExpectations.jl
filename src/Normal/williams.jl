import Distributions: Normal, std
import ExponentialFamily: GaussianDistributionsFamily
using StaticArrays
using SpecialFunctions: erfc, erf

function mean(::ClosedWilliamsProduct, p::Logpdf{NormalType}, q::Normal{T}) where {T, NormalType <: GaussianDistributionsFamily}
    μ_q, σ_q = mean(q), std(q)
    μ_p, σ_p = mean(p.dist), std(p.dist)
    return @SVector [
        -(μ_p - μ_q)/σ_p^2,
        -σ_q/σ_p^2
    ]
end

function mean(::ClosedWilliamsProduct, p::Logpdf{Laplace{T}}, q::Normal{T}) where {T}
    μ, σ = mean(q), std(q)
    (loc, θ) = params(p.dist)
    diff = loc - μ
    exp_part = exp(-diff^2/(2*σ^2))
    erf_part = erf((-loc + μ)/(sqrt(2) * σ))
    return @SVector [
        -(1 + (exp_part * sqrt(2/π) * diff)/σ + erf_part)/(2*θ),
        (exp_part * (-diff^2 - 2*σ^2))/(sqrt(2*π) * θ * σ^2)
    ]
end