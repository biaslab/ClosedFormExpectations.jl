export LogGammaKernel

"""
    LogGammaKernel(α, β)

The unnormalised log-density of a [`LogGamma`](@ref) distribution, as an integrand:

```math
f(x) = \\beta x - e^{x} / \\alpha, \\quad \\alpha > 0, \\; \\beta \\in \\mathbb{R}.
```

For ``\\beta > 0`` it differs from `Base.Fix1(logpdf, LogGamma(α, β))` by the constant
``\\beta \\log \\alpha + \\log \\Gamma(\\beta)``. For ``\\beta \\le 0`` the density ``e^{f}`` cannot be
normalised, so there is no `LogGamma` distribution, but `f` is still a log-message: a Gamma
message ``z^{a-1} e^{-b z}`` pulled back through ``z = e^{x}`` is `LogGammaKernel(1 / b, a - 1)`,
and for a Poisson count of zero ``a = 1``. Under a univariate Normal its expectation and gradient
have closed forms, for any ``\\beta``.
"""
struct LogGammaKernel{T <: Real}
    α::T
    β::T
    function LogGammaKernel{T}(α::T, β::T) where {T <: Real}
        α > zero(α) || throw(DomainError(α, "the scale α of a `LogGammaKernel` must be positive"))
        return new{T}(α, β)
    end
end

function LogGammaKernel(α::Real, β::Real)
    T = float(promote_type(typeof(α), typeof(β)))
    return LogGammaKernel{T}(convert(T, α), convert(T, β))
end

Distributions.params(f::LogGammaKernel) = (f.α, f.β)

(f::LogGammaKernel)(x) = f.β * x - exp(x - log(f.α))
