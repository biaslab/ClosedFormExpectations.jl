import Distributions: Normal
using StaticArrays
using SpecialFunctions: erfc, erf

function mean(::ClosedWilliamsProduct, p::Logpdf{Normal{T}}, q::Normal) where {T}
    μ_q, σ_q = q.μ, q.σ
    μ_p, σ_p = p.dist.μ, p.dist.σ
    return @SVector [
        -(μ_p - μ_q)/σ_p^2,
        -σ_q/σ_p^2
    ]
end

function mean(::ClosedWilliamsProduct, p::Logpdf{Laplace{T}}, q::Normal) where {T}
    μ, σ = q.μ, q.σ
    (loc, θ) = params(p.dist)
    diff = loc - μ
    exp_part = exp(-diff^2/(2*σ^2))
    erf_part = erf((-loc + μ)/(sqrt(2) * σ))
    return @SVector [
        -(1 + (exp_part * sqrt(2/π) * diff)/σ + erf_part)/(2*θ),
        (exp_part * (-diff^2 - 2*σ^2))/(sqrt(2*π) * θ * σ^2)
    ]
end