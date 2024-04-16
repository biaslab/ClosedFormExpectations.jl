import Distributions: Laplace, Normal, Rayleigh, params, cdf


function mean(::ClosedFormExpectation, p::ComposedFunction{typeof(log), Normal{T}}, q::Normal) where {T}
    μ_q, σ_q = q.μ, q.σ
    μ_p, σ_p = p.inner.μ, p.inner.σ
    return - 1/2 * log(2 * π * σ_p^2) - (σ_q^2 + (μ_p- μ_q)^2) / (2 * σ_p^2)
end 

function mean(::ClosedFormExpectation, p::ComposedFunction{typeof(log), Laplace{T}}, q::Normal) where {T}
    μ_q, σ_q = q.μ, q.σ
    (μ_p, θ_p) = params(p.inner)
    normal = Normal(0,σ_q)
    diff = μ_p - μ_q
    return - log(2*θ_p) - θ_p^(-1) * ( 2 * (σ_q/sqrt(2*π)) * exp(-diff^2/(2*σ_q^2))  +  diff * (2 * cdf(normal,diff) - 1) )
end 
