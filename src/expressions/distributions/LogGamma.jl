export LogGamma

import Distributions: ContinuousUnivariateDistribution, @distr_support

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

@distr_support LogGamma 0.0 Inf

#### Conversions
convert(::Type{LogGamma{T}}, α::S, β::S) where {T <: Real, S <: Real} = LogGamma(T(α), T(β))
Base.convert(::Type{LogGamma{T}}, d::LogGamma) where {T<:Real} = LogGamma{T}(T(d.α), T(d.β))
Base.convert(::Type{LogGamma{T}}, d::LogGamma{T}) where {T<:Real} = d

#### Parameters
params(d::LogGamma) = (d.α, d.β)
partype(::LogGamma{T}) where {T} = Tuple{T}

#### Statistics

meanlogx(d::LogGamma) = mean(Gamma(d.α, d.β))
varlogx(d::LogGamma) = var(Gamma(d.α, d.β))
stdlogx(d::LogGamma) = std(Gamma(d.α, d.β))
shapelogx(d::LogGamma) = shape(Gamma(d.α, d.β))
ratelogx(d::LogGamma) = rate(Gamma(d.α, d.β))
scalelogx(d::LogGamma) = scale(Gamma(d.α, d.β))

mean(d::LogGamma) = mean(ClosedFormExpectation(), log, Gamma(d.α, d.β))

# Evaluation
function pdf(d::LogGamma, x::Real) 
    α, β = params(d)
    return exp(β * x - exp(x) / α) / (α^β * gamma(β))
end

function logpdf(d::LogGamma, x::Real)
    α, β = params(d)
    return β * x - exp(x) / α - β * log(α) - loggamma(β)
end
