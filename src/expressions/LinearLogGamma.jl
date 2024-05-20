export LinearLogGamma

using LinearAlgebra

import Distributions: ContinuousMultivariateDistribution, params

struct LinearLogGamma{T<:Real, C <: AbstractVector} <: ContinuousMultivariateDistribution
    α::T
    β::T
    weights::C
end

### Construction
function LinearLogGamma(α::T, β::T, weights::AbstractVector{T}) where {T<:Real}
    return LinearLogGamma{T, typeof(weights)}(α, β, weights)
end

function LinearLogGamma(α::Real, β::Real, weights::AbstractVector{<:Real})
    R = Base.promote_eltype(α, β, weights)
    LinearLogGamma(convert(R, α), convert(R, β), convert(AbstractVector{R}, weights))
end

function Distributions.params(d::LinearLogGamma)
    return d.α, d.β, d.weights
end

# logpdf
function Distributions.logpdf(d::LinearLogGamma, x::AbstractVector)
    α, β, weights = Distributions.params(d)
    loggamma = LogGamma(α, β)
    return logpdf(loggamma, dot(weights, x))
end