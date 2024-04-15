import Distributions: Exponential, LogNormal, scale
import Base.MathConstants: eulergamma

function meanlog(::ClosedFormExpectation, q::Exponential, p::LogNormal)
    μ, σ = p.μ, p.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ))) + eulergamma - log(λ) - 0.5*log(2π) - log(σ)
end 

function meanlog(::ClosedFormExpectation, q::Exponential, p::typeof(identity))
    λ = 1/mean(q)
    return -eulergamma - log(λ)
end

function meanlog(::ClosedFormExpectation, q::Exponential, p::ExpLogSquare)
    μ = p.μ
    σ = p.σ
    λ = mean(q)
    return 1/(2*σ^2)*(-(μ+eulergamma)^2 - π^2/6 - log(λ)*(-2*(eulergamma+μ) + log(λ)))
end