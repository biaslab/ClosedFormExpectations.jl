export LinearLogGamma

using LinearAlgebra

import Distributions: ContinuousMultivariateDistribution, params

"""
    LinearLogGamma(α, β, weights)

An unnormalized multivariate distribution derived from the LogGamma distribution. (see LogGamma)

The LinearLogGamma distribution is an distribution on a multidimensional `x`, derived from the LogGamma distribution. It is defined as:
$\widetilde{\mathcal{LG}}(x \mid \alpha, \beta, w) = \mathcal{LG}(x^{\top} w \mid a, b)$, 
where weights is a fixed vector of covariates, and \alpha and \beta are the scale and shape parameters of the LogGamma distribution, respectively.
Fields

α::T: The scale parameter of the LogGamma distribution.

β::T: The shape parameter of the LogGamma distribution.

weights::C: The fixed vector of covariates.
"""
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