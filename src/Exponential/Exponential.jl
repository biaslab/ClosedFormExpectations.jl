import Distributions: Exponential, LogNormal, scale
import Base.MathConstants: eulergamma

function mean(::ClosedFormExpectation, q::Exponential, p::ComposedFunction{typeof(log), LogNormal{T}}) where {T}
    μ, σ = p.inner.μ, p.inner.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ))) + eulergamma - log(λ) - 0.5*log(2π) - log(σ)
end 

function mean(::ClosedFormExpectation, q::Exponential, p::typeof(log))
    return -eulergamma + log(mean(q))
end

function mean(::ClosedFormExpectation, q::Exponential, p::ComposedFunction{typeof(log), typeof(identity)})
    return -eulergamma + log(mean(q))
end

function mean(::ClosedFormExpectation, q::Exponential, p::ComposedFunction{typeof(log), ExpLogSquare{T}}) where {T}
    μ, σ = p.inner.μ, p.inner.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ)))
end

function mean(::ClosedWilliamsProduct, q::Exponential, p::typeof(identity))
    return 1/mean(q)
end

function mean(::ClosedWilliamsProduct, q::Exponential, p::ExpLogSquare)
    μ = p.μ
    σ = p.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ)
end

function mean(::ClosedWilliamsProduct, q::Exponential, p::LogNormal)
    μ, σ = p.μ, p.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-1/λ*(-2*(eulergamma+μ) + log(λ)) - log(λ)/λ) - 1/λ
end 