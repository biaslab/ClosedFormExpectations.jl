function mean(::ClosedFormExpectation, p::Logpdf{<:GammaDistributionsFamily}, q::LogNormal)
    α_p, θ_p = shape(p.dist), scale(p.dist)
    μ_q, σ_q = q.μ, q.σ
    E_log_x = μ_q
    E_x = exp(μ_q + σ_q^2 / 2)
    return (α_p - 1) * E_log_x - E_x / θ_p - α_p * log(θ_p) - loggamma(α_p)
end

function mean(::ClosedFormExpectation, p::Logpdf{LogNormal{T}}, q::LogNormal) where {T}
    μ_p, σ_p = p.dist.μ, p.dist.σ
    μ_q, σ_q = q.μ, q.σ
    return -μ_q - log(σ_p) - 0.5 * log(2 * pi) - (σ_q^2 + (μ_q - μ_p)^2) / (2 * σ_p^2)
end

function mean(::ClosedFormExpectation, p::Logpdf{<:UnivariateNormalDistributionsFamily}, q::LogNormal)
    μ_p, v_p = mean_var(p.dist)
    μ_q, σ_q = q.μ, q.σ
    E_x = exp(μ_q + σ_q^2 / 2)
    E_x2 = exp(2 * μ_q + 2 * σ_q^2)
    return -0.5 * log(2 * pi * v_p) - (E_x2 - 2 * μ_p * E_x + μ_p^2) / (2 * v_p)
end
