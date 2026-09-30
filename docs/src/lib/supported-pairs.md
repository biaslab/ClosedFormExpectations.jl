# [Supported Pairs](@id lib-supported-pairs)

This page lists all supported `(distribution, function)` pairs for which closed-form expectations and Williams' products are implemented.

## [Categorical Distribution](@id lib-categorical)

### ClosedFormExpectation

For `q = Categorical(p)`, `ClosedFormExpectation` accepts any callable scalar score `f`
and computes the exact finite sum:

```math
\mathbb{E}_q[f(k)] = \sum_{k:p_k>0} p_k f(k).
```

Categories with zero probability are skipped without evaluating `f(k)`, so scores
need only be defined at categories with positive probability. Targets may also be
`Logpdf` wrappers (including noncategorical distributions), raw distributions, or
`Base.Fix1(logpdf, target)`. Log-density products retain their additive decomposition.

```julia
q = Categorical([0.2, 0.3, 0.5])
scores = [1, 2, 4]
mean(ClosedFormExpectation(), k -> scores[k], q) # ≈ 2.8
```

### ClosedWilliamsProduct

For `ef = convert(ExponentialFamilyDistribution, q)`, the default
`ClosedWilliamsProduct()` returns the **full length-K score gradient in softmax
logits**, with ``p = \operatorname{softmax}(\eta)`` and ``\mu = \mathbb{E}_q[f]``:

```math
g_k = p_k(f(k) - \mu).
```

Zero-probability categories are skipped without evaluating `f(k)` and receive
zero entries. The reference category's component is retained. This method is defined only for the EF parametrization, not for probability
parameters on a plain `Categorical`. It accepts arbitrary callable scores, including
`Logpdf` targets, and uses the existing raw-distribution and product bridges.

```julia
ef = convert(ExponentialFamilyDistribution, q)
mean(ClosedWilliamsProduct(), k -> scores[k], ef) # ≈ [-0.36, -0.24, 0.6]
```

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
| `Logpdf(ReLUForwardMessage(m_x, v_x))` | ``\mathbb{E}_q[\log m_{f \to y}(y)]``, ``m_{f \to y} = \Phi(-\alpha_x)\delta(y) + \mathbf{1}_{y>0}\mathcal{N}(y;\,m_x,v_x)`` |
| `Logpdf(ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

!!! note "Why the forward message collapses to a Gaussian integral here"
    The forward message ``m_{f \to y}`` is a **spike-slab** distribution, not a Gaussian.
    Its log-density is ``\log\mathcal{N}(y;\,m_x,v_x)`` for ``y > 0`` and ``-\infty`` for ``y \leq 0``.
    Because Gamma (and LogNormal) have support on ``(0,\infty)``, the atom at ``y=0`` carries
    zero mass under ``q``, so the expectation equals ``\mathbb{E}_q[\log\mathcal{N}(y;\,m_x,v_x)]``
    numerically — but this is a property of the support of ``q``, not of the message itself.
    For ``q = \mathcal{N}(\mu,\sigma^2)`` the atom contributes ``-\infty`` and the expectation
    is undefined; see the Normal section for the backward message instead.

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
| `Logpdf(ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

### ClosedWilliamsProduct

Returns a 2-element `SVector` with gradients ``[\nabla_\mu, \nabla_\sigma]``.

| Function `f` | Expression |
|:-------------|:-----------|
| `Abs()` | ``\mathbb{E}_q[\lvert x \rvert \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(Laplace(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Lap}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Logpdf(LogGamma(α, β))` | ``\mathbb{E}_q[\log p_{\mathrm{LG}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |

## [LogNormal Distribution](@id lib-lognormal)

Distribution ``q \sim \mathrm{LogNormal}(\mu, \sigma)``, where ``\mu`` is the log-mean and ``\sigma`` is the log-standard-deviation.

### ClosedFormExpectation

| Function `f` | Expression |
|:-------------|:-----------|
| `Logpdf(Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x)]`` |
| `Logpdf(Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x)]`` |
| `Logpdf(LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `Logpdf(ReLUForwardMessage(m_x, v_x))` | ``\mathbb{E}_q[\log m_{f \to y}(y)]``, ``m_{f \to y} = \Phi(-\alpha_x)\delta(y) + \mathbf{1}_{y>0}\mathcal{N}(y;\,m_x,v_x)`` |
| `Logpdf(ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

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

For `ClosedWilliamsProduct`, the following ExponentialFamily parametrizations are supported:

| Distribution `q` | Gradient w.r.t. | Notes |
|:------------------|:----------------|:------|
| `NormalMeanVariance(μ, v)` | ``[\nabla_\mu, \nabla_v]`` | Jacobian from ``(\mu, \sigma) \to (\mu, v)`` |
| `ExponentialFamilyDistribution{NormalMeanVariance}` | ``[\nabla_{\eta_1}, \nabla_{\eta_2}]`` | Natural parameters |
| `ExponentialFamilyDistribution{Gamma}` | ``[\nabla_{\eta_1}, \nabla_{\eta_2}]`` | Natural parameters |
| `ExponentialFamilyDistribution{Categorical}` | ``[\nabla_{\eta_1}, \ldots, \nabla_{\eta_K}]`` | Full softmax-logit score gradient |

The Normal and Gamma methods work with **any** function `f` supported for the corresponding base distribution. The Categorical method enumerates arbitrary callable scores over categories with positive probability.

## [ProductOf Distributions](@id lib-productof-pairs)

For any `ProductOf` distribution from ExponentialFamily.jl, the expectation decomposes additively:

```math
\mathbb{E}_q[\log(p_1 \cdot p_2)] = \mathbb{E}_q[\log p_1] + \mathbb{E}_q[\log p_2]
```

This is supported for both `ClosedFormExpectation` and `ClosedWilliamsProduct`, and works recursively for nested products. Each component must individually have a supported closed-form expression.
