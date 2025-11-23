export LogGamma

using Distributions
import Distributions: ContinuousUnivariateDistribution, @distr_support, @check_args, mean, var

"""
    LogGamma(α, β)

The LogGamma distribution is a continuous probability distribution on the real numbers. It is defined as:

The probability density function of the LogGamma distribution is defined as:

    ```math
    \\mathcal{LG}(x \\mid a, b) = \\frac{e^{b x} e^{-e^{x}/a}}{a^{b} \\Gamma(b)}, \\quad -\\infty < x < \\infty, a > 0, b > 0.
    ```
Ref:
    https://www.math.wm.edu/~leemis/chart/UDR/PDFs/Loggamma.pdf
"""
struct LogGamma{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    function LogGamma{T}(α::T, β::T) where {T}
        new{T}(α, β)
    end
end

function LogGamma(α::T, β::T; check_args=true) where {T<:Real}
    @check_args LogGamma (α > zero(α) && β > zero(β))
    return LogGamma{T}(α, β)
end

LogGamma(α::Real, β::Real; check_args::Bool=true) = LogGamma(promote(α, β)...; check_args=check_args)
LogGamma(α::Integer, β::Integer; check_args::Bool=true) = LogGamma(float(α), float(β); check_args=check_args)

@distr_support LogGamma -Inf Inf

#### Conversions
Base.convert(::Type{LogGamma{T}}, α::S, β::S) where {T <: Real, S <: Real} = LogGamma(T(α), T(β))
Base.convert(::Type{LogGamma{T}}, d::LogGamma) where {T<:Real} = LogGamma{T}(T(d.α), T(d.β))
Base.convert(::Type{LogGamma{T}}, d::LogGamma{T}) where {T<:Real} = d

#### Parameters
Distributions.params(d::LogGamma) = (d.α, d.β)
Distributions.partype(::LogGamma{T}) where {T} = Tuple{T}

# Evaluation
function Distributions.pdf(d::LogGamma, x::Real)
    return exp(logpdf(d, x))
end

function Distributions.logpdf(d::LogGamma, x::Real)
    α, β = Distributions.params(d)
    return β * x - exp(x - log(α)) - β * log(α) - loggamma(β)
end
