import Distributions: Exponential, LogNormal, scale, kldivergence, entropy
import Base.MathConstants: eulergamma
import LogExpFunctions: xlogx

function mean(::ClosedFormExpectation, f::Logpdf{Exponential{T}}, q::Exponential) where {T}
    λ1 = mean(q)
    λ2 = mean(f.dist)
    return -(λ1 + xlogx(λ2))/λ2
end

function mean(::ClosedFormExpectation, f::Logpdf{LogNormal{T}}, q::Exponential) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ))) + eulergamma - log(λ) - 0.5*log(2π) - log(σ)
end 

function mean(::ClosedFormExpectation, ::typeof(log), q::Exponential)
    return -eulergamma + log(mean(q))
end

function mean(::ClosedFormExpectation, ::ComposedFunction{typeof(log), typeof(identity)}, q::Exponential)
    return -eulergamma + log(mean(q))
end

function mean(::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Exponential) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ)))
end

function mean(::ClosedWilliamsProduct, ::typeof(log), q::Exponential)
    return 1/mean(q)
end

function mean(::ClosedWilliamsProduct, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Exponential) where {T}
    μ = f.inner.μ
    σ = f.inner.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ)
end

function mean(::ClosedWilliamsProduct, f::Logpdf{LogNormal{T}}, q::Exponential) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ) - 1/λ
end 

function mean(::ClosedWilliamsProduct, f::Logpdf{Exponential{T}}, ::Exponential) where {T}
    λ2 = mean(f.dist)
    return -1/λ2
end