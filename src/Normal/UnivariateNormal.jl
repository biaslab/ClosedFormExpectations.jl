import Distributions: Laplace, Normal

function mean(::ClosedFormExpectation, q::Normal, p::ComposedFunction{typeof(log), Normal{T}}) where {T}
    μ_q, σ_q = q.μ, q.σ
    μ_p, σ_p = p.inner.μ, p.inner.σ
    return - 1/2 * log(2 * π * σ_p^2) - (σ_q^2 + (μ_p- μ_q)^2) / (2 * σ_p^2)
end 

function mean(::ClosedFormExpectation, q::Normal, p::ComposedFunction{typeof(log), Laplace{T}}) where {T}
    μ_q, σ_q = q.μ, q.σ, 
    μ_p, θ_p = p.μ, p.Θ
    return 0
end 
