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