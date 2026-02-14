using StaticArrays
import SpecialFunctions: trigamma, digamma, polygamma
import Distributions: Gamma, shape, rate, scale
import LogExpFunctions: xlogx
import ExponentialFamily: NormalMeanPrecision, NormalMeanVariance, mean_precision, mean_var, UnivariateNormalDistributionsFamily, GammaDistributionsFamily

function mean(::ClosedFormExpectation, ::typeof(log), q::Gamma)
    return digamma(shape(q)) + log(scale(q))
end

function mean(::ClosedWilliamsProduct, ::typeof(log), q::Gamma)
    return @SVector [
        polygamma(1, shape(q)),
        1/scale(q)
    ]
end

function mean(::ClosedFormExpectation, ::typeof(xlogx), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * (digamma(shape(q)+1) - log(rate(q))) 
end

function mean(::ClosedFormExpectation, ::typeof(xlog2x), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * ((digamma(shape(q)+1) - log(rate(q)))^2 + trigamma(shape(q)+1))
end

function mean(::ClosedFormExpectation, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return trigamma(shape(q)) + (digamma(shape(q)) - log(rate(q)))^2
end

function mean(::ClosedWilliamsProduct, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return @SVector [
        polygamma(2, shape(q)) + 2 * (digamma(shape(q)) - log(rate(q)))*trigamma(shape(q)),
        2 * (digamma(shape(q)) - log(rate(q))) * rate(q)
    ]
end

function mean(strategy::ClosedFormExpectation, ::ComposedFunction{Power{Val{3}}, typeof(log)}, q::Gamma)
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return polygamma(2, shape(q)) + 3*Elogx * Elog2x - 2*Elogx^3
end

function mean(strategy::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return -1/(2*σ^2)*(μ^2 - 2*μ*Elogx + Elog2x)
end

function mean(strategy::ClosedWilliamsProduct, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = mean(strategy, log, q)
    Elog2x = mean(strategy, Square() ∘ log, q)
    return (2*μ*Elogx - Elog2x)/(2*σ^2)
end

function mean(strategy::ClosedFormExpectation, f::Logpdf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    E_logexplogsquare = mean(strategy, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = mean(strategy, log, q)
    return E_logexplogsquare - E_logx - log(σ) - 0.5*log(2pi)
end

function mean(strategy::ClosedWilliamsProduct, f::Logpdf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.dist.μ, f.dist.σ
    E_logexplogsquare = mean(strategy, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = mean(strategy, log, q)
    return E_logexplogsquare - E_logx
end

function mean(::ClosedWilliamsProduct, p::Logpdf{<:GammaDistributionsFamily}, q::Gamma)
    α_p, θ_p = shape(p.dist), scale(p.dist)
    α_q, θ_q = shape(q), scale(q)
    # ∇_{α_q} E_q[log p]
    grad_shape = (α_p - 1) * polygamma(1, α_q) - θ_q / θ_p
    # ∇_{θ_q} E_q[log p]
    grad_scale = (α_p - 1) / θ_q - α_q / θ_p
    return @SVector [grad_shape, grad_scale]
end

function mean(::ClosedFormExpectation, p::Logpdf{<:GammaDistributionsFamily}, q::Gamma)
    α_p, θ_p = shape(p.dist), scale(p.dist)
    α_q, θ_q = shape(q), scale(q)
    E_log_x = digamma(α_q) + log(θ_q)
    E_x = α_q * θ_q
    return -loggamma(α_p) - α_p * log(θ_p) + (α_p - 1) * E_log_x - E_x / θ_p
end

function mean(::ClosedFormExpectation, p::Logpdf{<:UnivariateNormalDistributionsFamily}, q::Gamma)
    # E_q [log N(x | λ, v)]
    # We use mean_var to support any UnivariateNormalDistributionsFamily.
    # mean_var returns (μ, v) of the distribution.
    # In this context, the distribution parameter `μ` corresponds to `x` in the derivation,
    # and the random variable `λ` (from q) corresponds to the mean of the Normal.
    # log N(x | λ, v) = -0.5 log(2πv) - (λ - x)^2 / 2v
    
    x_val, v_val = mean_var(p.dist)
    α, θ = shape(q), scale(q)
    
    # E[λ]
    E_lambda = α * θ
    # E[λ^2] = Var(λ) + E[λ]^2 = α*θ^2 + (α*θ)^2 = α*θ^2(1+α)
    E_lambda2 = α * θ^2 * (1 + α)
    
    term1 = -0.5 * log(2 * pi * v_val)
    term2 = -1 / (2 * v_val) * (E_lambda2 - 2 * x_val * E_lambda + x_val^2)
    
    return term1 + term2
end

function mean(::ClosedWilliamsProduct, p::Logpdf{<:UnivariateNormalDistributionsFamily}, q::Gamma)
    x_val, v_val = mean_var(p.dist)
    α, θ = shape(q), scale(q)
    
    # Gradients derived in feature request
    # ∇_α E = -1/2v * (θ^2(1+2α) - 2xθ)
    grad_alpha = -1 / (2 * v_val) * (θ^2 * (1 + 2 * α) - 2 * x_val * θ)
    
    # ∇_θ E = -1/2v * (2αθ(1+α) - 2xα)
    grad_theta = -1 / (2 * v_val) * (2 * α * θ * (1 + α) - 2 * x_val * α)
    
    return @SVector [grad_alpha, grad_theta]
end
