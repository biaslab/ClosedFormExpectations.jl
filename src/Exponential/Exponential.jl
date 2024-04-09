import Distributions: Exponential, LogNormal
import Base.MathConstants: eulergamma

function meanlog(::ClosedFormExpectation, q::Exponential, p::LogNormal)
    μ, σ = p.μ, p.σ
    λ = mean(q)
    return 1/12 * σ^2 * (6 * eulergamma^2 + 12 * eulergamma * μ + 6 * μ^2 + π^2 + 12 * eulergamma * log(λ) + 12 * μ * log(λ) + 6 * log(λ)^2)
end 