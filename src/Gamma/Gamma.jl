using StaticArrays
import SpecialFunctions: trigamma, digamma, polygamma
import Distributions: Gamma, shape, rate, scale
import LogExpFunctions: xlogx
import ExponentialFamily: NormalMeanPrecision, NormalMeanVariance, mean_precision, mean_var, UnivariateNormalDistributionsFamily, GammaDistributionsFamily

@supports_expectation function expectation(::ClosedFormExpectation, ::typeof(log), q::Gamma)
    return digamma(shape(q)) + log(scale(q))
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation{Nothing}, ::typeof(log), q::Gamma)
    return @SVector [
        polygamma(1, shape(q)),
        1/scale(q)
    ]
end

@supports_expectation function expectation(::ClosedFormExpectation, ::typeof(xlogx), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * (digamma(shape(q)+1) - log(rate(q))) 
end

@supports_expectation function expectation(::ClosedFormExpectation, ::typeof(xlog2x), q::Gamma)
    scaler = shape(q)/rate(q)
    return scaler * ((digamma(shape(q)+1) - log(rate(q)))^2 + trigamma(shape(q)+1))
end

@supports_expectation function expectation(::ClosedFormExpectation, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return trigamma(shape(q)) + (digamma(shape(q)) - log(rate(q)))^2
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation{Nothing}, ::ComposedFunction{Square, typeof(log)}, q::Gamma)
    return @SVector [
        polygamma(2, shape(q)) + 2 * (digamma(shape(q)) - log(rate(q)))*trigamma(shape(q)),
        2 * (digamma(shape(q)) - log(rate(q))) * rate(q)
    ]
end

@supports_expectation function expectation(backend::ClosedFormExpectation, ::ComposedFunction{Power{Val{3}}, typeof(log)}, q::Gamma)
    Elogx = expectation(backend, log, q)
    Elog2x = expectation(backend, Square() ∘ log, q)
    return polygamma(2, shape(q)) + 3*Elogx * Elog2x - 2*Elogx^3
end

@supports_expectation function expectation(backend::ClosedFormExpectation, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = expectation(backend, log, q)
    Elog2x = expectation(backend, Square() ∘ log, q)
    return -1/(2*σ^2)*(μ^2 - 2*μ*Elogx + Elog2x)
end

@supports_expectation_gradient function expectation_gradient(backend::ClosedFormExpectation{Nothing}, f::ComposedFunction{typeof(log), ExpLogSquare{T}}, q::Gamma) where {T}
    μ, σ = f.inner.μ, f.inner.σ
    Elogx = expectation_gradient(backend, log, q)
    Elog2x = expectation_gradient(backend, Square() ∘ log, q)
    return (2*μ*Elogx - Elog2x)/(2*σ^2)
end

@supports_expectation function expectation(backend::ClosedFormExpectation, f::LogpdfOf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.x.μ, f.x.σ
    E_logexplogsquare = expectation(backend, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = expectation(backend, log, q)
    return E_logexplogsquare - E_logx - log(σ) - 0.5*log(2pi)
end

@supports_expectation_gradient function expectation_gradient(backend::ClosedFormExpectation{Nothing}, f::LogpdfOf{LogNormal{T}}, q::Gamma) where {T}
    μ, σ = f.x.μ, f.x.σ
    E_logexplogsquare = expectation_gradient(backend, log ∘ ExpLogSquare(μ, σ), q)
    E_logx = expectation_gradient(backend, log, q)
    return E_logexplogsquare - E_logx
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation{Nothing}, p::LogpdfOf{<:GammaDistributionsFamily}, q::Gamma)
    α_p, θ_p = shape(p.x), scale(p.x)
    α_q, θ_q = shape(q), scale(q)
    # ∇_{α_q} E_q[log p]
    grad_shape = (α_p - 1) * polygamma(1, α_q) - θ_q / θ_p
    # ∇_{θ_q} E_q[log p]
    grad_scale = (α_p - 1) / θ_q - α_q / θ_p
    return @SVector [grad_shape, grad_scale]
end

@supports_expectation function expectation(::ClosedFormExpectation, p::LogpdfOf{<:GammaDistributionsFamily}, q::Gamma)
    α_p, θ_p = shape(p.x), scale(p.x)
    α_q, θ_q = shape(q), scale(q)
    E_log_x = digamma(α_q) + log(θ_q)
    E_x = α_q * θ_q
    return -loggamma(α_p) - α_p * log(θ_p) + (α_p - 1) * E_log_x - E_x / θ_p
end

@supports_expectation function expectation(::ClosedFormExpectation, p::LogpdfOf{<:UnivariateNormalDistributionsFamily}, q::Gamma)
    # E_q [log N(x | λ, v)]
    # We use mean_var to support any UnivariateNormalDistributionsFamily.
    # mean_var returns (μ, v) of the distribution.
    # In this context, the distribution parameter `μ` corresponds to `x` in the derivation,
    # and the random variable `λ` (from q) corresponds to the mean of the Normal.
    # log N(x | λ, v) = -0.5 log(2πv) - (λ - x)^2 / 2v
    
    x_val, v_val = mean_var(p.x)
    α, θ = shape(q), scale(q)
    
    # E[λ]
    E_lambda = α * θ
    # E[λ^2] = Var(λ) + E[λ]^2 = α*θ^2 + (α*θ)^2 = α*θ^2(1+α)
    E_lambda2 = α * θ^2 * (1 + α)
    
    term1 = -0.5 * log(2 * pi * v_val)
    term2 = -1 / (2 * v_val) * (E_lambda2 - 2 * x_val * E_lambda + x_val^2)
    
    return term1 + term2
end

@supports_expectation_gradient function expectation_gradient(::ClosedFormExpectation{Nothing}, p::LogpdfOf{<:UnivariateNormalDistributionsFamily}, q::Gamma)
    x_val, v_val = mean_var(p.x)
    α, θ = shape(q), scale(q)
    
    # Gradients derived in feature request
    # ∇_α E = -1/2v * (θ^2(1+2α) - 2xθ)
    grad_alpha = -1 / (2 * v_val) * (θ^2 * (1 + 2 * α) - 2 * x_val * θ)
    
    # ∇_θ E = -1/2v * (2αθ(1+α) - 2xα)
    grad_theta = -1 / (2 * v_val) * (2 * α * θ * (1 + α) - 2 * x_val * α)
    
    return @SVector [grad_alpha, grad_theta]
end
