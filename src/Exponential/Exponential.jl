using StaticArrays
import Distributions: Exponential, LogNormal, scale, kldivergence, entropy

# The gradients are with respect to the scale of `q`, its mean, as a vector of length one
import Base.MathConstants: eulergamma
import LogExpFunctions: xlogx

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{Exponential{T}}, q::Exponential) where {T}
    λ1 = mean(q)
    λ2 = mean(f.x)
    return -(λ1 + xlogx(λ2))/λ2
end

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{LogNormal{T}}, q::Exponential) where {T}
    μ, σ = f.x.μ, f.x.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ))) + eulergamma - log(λ) - 0.5*log(2π) - log(σ)
end 

@supports_expectation function expectation(::ClosedFormExpectation, ::typeof(log), q::Exponential)
    return -eulergamma + log(mean(q))
end

@supports_expectation function expectation(::ClosedFormExpectation, ::ComposedFunction{typeof(log), typeof(identity)}, q::Exponential)
    return -eulergamma + log(mean(q))
end

@supports_expectation function expectation(::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Exponential) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ)))
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, ::typeof(log), q::Exponential)
    return @SVector [1/mean(q)]
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Exponential) where {T}
    μ = f.inner.μ
    σ = f.inner.σ
    λ = mean(q)
    return @SVector [1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ)]
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, f::LogpdfOf{LogNormal{T}}, q::Exponential) where {T}
    μ, σ = f.x.μ, f.x.σ
    λ = mean(q)
    return @SVector [1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ) - 1/λ]
end 

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation, f::LogpdfOf{Exponential{T}}, ::Exponential) where {T}
    λ2 = mean(f.x)
    return @SVector [-1/λ2]
end