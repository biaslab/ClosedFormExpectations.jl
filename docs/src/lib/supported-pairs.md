# [Supported Pairs](@id lib-supported-pairs)

This page lists all supported `(distribution, function)` pairs for which closed-form expectations and Williams' products are implemented.

## [Exponential Distribution](@id lib-exponential)

Distribution ``q \sim \mathrm{Exponential}(\lambda)``, where ``\lambda`` is the scale (mean).

### ClosedFormExpectation

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x]`` |
| `Logpdf(Exponential(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Exp}}(x)]`` |
| `Logpdf(LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2}\right]`` |

### ClosedWilliamsProduct

| Function `f` | Returns |
|:-------------|:--------|
| `log` | ``\mathbb{E}_q[\log x \cdot \nabla_\lambda \log q(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2} \cdot \nabla_\lambda \log q(x)\right]`` |
| `Logpdf(Exponential(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Exp}}(x) \cdot \nabla_\lambda \log q(x)]`` |
| `Logpdf(LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x) \cdot \nabla_\lambda \log q(x)]`` |

## [Gamma Distribution](@id lib-gamma)

Distribution ``q \sim \mathrm{Gamma}(\alpha, \theta)``, where ``\alpha`` is the shape and ``\theta`` is the scale.

!!! note
    Any `GammaDistributionsFamily` type is accepted, including `GammaShapeRate`. The package uses `shape(q)` and `scale(q)` internally.

### ClosedFormExpectation

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x]`` |
| `xlogx` | ``\mathbb{E}_q[x \log x]`` |
| `xlog2x` | ``\mathbb{E}_q[x (\log x)^2]`` |
| `Square() ∘ log` | ``\mathbb{E}_q[(\log x)^2]`` |
| `Power(Val(3)) ∘ log` | ``\mathbb{E}_q[(\log x)^3]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2}\right]`` |
| `Logpdf(LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `Logpdf(Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x)]`` |
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Normal}}(x)]`` |

### ClosedWilliamsProduct

Returns a 2-element `SVector` with gradients ``[\nabla_\alpha, \nabla_\theta]``.

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Square() ∘ log` | ``\mathbb{E}_q[(\log x)^2 \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q[\ldots \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Logpdf(LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Logpdf(Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Normal}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |

## [Normal Distribution](@id lib-normal)

Distribution ``q \sim \mathcal{N}(\mu, \sigma^2)``.

!!! note
    Any `GaussianDistributionsFamily` type is accepted for `ClosedFormExpectation`, including `NormalMeanVariance`, `NormalMeanPrecision`, and `NormalWeightedMeanPrecision`. For `ClosedWilliamsProduct`, the base implementation is on `Normal(μ, σ)`, with Jacobian-adjusted dispatches for `NormalMeanVariance` and `ExponentialFamilyDistribution{NormalMeanVariance}`.

### ClosedFormExpectation

| Function `f` | Expression |
|:-------------|:-----------|
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x)]`` |
| `Logpdf(Laplace(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Lap}}(x)]`` |
| `Abs()` | ``\mathbb{E}_q[\lvert x \rvert]`` |
| `Logpdf(LogGamma(α, β))` | ``\mathbb{E}_q[\log p_{\mathrm{LG}}(x)]`` |

### ClosedWilliamsProduct

Returns a 2-element `SVector` with gradients ``[\nabla_\mu, \nabla_\sigma]``.

| Function `f` | Expression |
|:-------------|:-----------|
| `Abs()` | ``\mathbb{E}_q[\lvert x \rvert \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(Laplace(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Lap}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(LogGamma(α, β))` | ``\mathbb{E}_q[\log p_{\mathrm{LG}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |

## [Multivariate Normal Distribution](@id lib-mvnormal)

Distribution ``q \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})``.

!!! note
    Any `MultivariateNormalDistributionsFamily` type is accepted.

### ClosedFormExpectation

| Function `f` | Expression |
|:-------------|:-----------|
| `Logpdf(MvNormal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(\mathbf{x})]`` |
| `Logpdf(LinearLogGamma(α, β, w))` | ``\mathbb{E}_q[\log p_{\mathrm{LLG}}(\mathbf{x})]`` |

## [ExponentialFamily Parametrizations](@id lib-ef-pairs)

For `ClosedWilliamsProduct`, the following ExponentialFamily parametrizations are supported with automatic Jacobian transformations:

| Distribution `q` | Gradient w.r.t. | Notes |
|:------------------|:----------------|:------|
| `NormalMeanVariance(μ, v)` | ``[\nabla_\mu, \nabla_v]`` | Jacobian from ``(\mu, \sigma) \to (\mu, v)`` |
| `ExponentialFamilyDistribution{NormalMeanVariance}` | ``[\nabla_{\eta_1}, \nabla_{\eta_2}]`` | Natural parameters |
| `ExponentialFamilyDistribution{Gamma}` | ``[\nabla_{\eta_1}, \nabla_{\eta_2}]`` | Natural parameters |

These work with **any** function `f` that is supported for the corresponding base distribution (`Normal` or `Gamma`).

## [ProductOf Distributions](@id lib-productof-pairs)

For any `ProductOf` distribution from ExponentialFamily.jl, the expectation decomposes additively:

```math
\mathbb{E}_q[\log(p_1 \cdot p_2)] = \mathbb{E}_q[\log p_1] + \mathbb{E}_q[\log p_2]
```

This is supported for both `ClosedFormExpectation` and `ClosedWilliamsProduct`, and works recursively for nested products. Each component must individually have a supported closed-form expression.
