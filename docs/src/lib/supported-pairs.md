# [Supported Pairs](@id lib-supported-pairs)

This page lists the `(distribution, function)` pairs for which `ClosedFormExpectation()` computes
the expectation, with `BayesBase.expectation`, and its gradient with respect to the parameters of
`q`, with `BayesBase.expectation_gradient`. The gradient equals the Williams product
``\mathbb{E}_q[f(x) \nabla_\theta \log q(x; \theta)]``.

## [Categorical Distribution](@id lib-categorical)

### Expectations

For `q = Categorical(p)`, the expectation of any callable scalar score `f` is the exact finite
sum:

```math
\mathbb{E}_q[f(k)] = \sum_{k:p_k>0} p_k f(k).
```

Categories with zero probability are skipped without evaluating `f(k)`, so scores
need only be defined at categories with positive probability. The integrand can be the
log-density `Base.Fix1(logpdf, p)` of any distribution, and the log-density of a product is
split into its parts.

```julia
q = Categorical([0.2, 0.3, 0.5])
scores = [1, 2, 4]
BayesBase.expectation(ClosedFormExpectation(), k -> scores[k], q) # ≈ 2.8
```

### Gradients

For `ef = convert(ExponentialFamilyDistribution, q)`, the gradient is the full vector of
length K in the softmax logits, with ``p = \operatorname{softmax}(\eta)`` and ``\mu = \mathbb{E}_q[f]``:

```math
g_k = p_k(f(k) - \mu).
```

Zero-probability categories are skipped without evaluating `f(k)` and receive
zero entries. The reference category's component is retained. The gradient is defined only for
the exponential family form, not for the probability parameters of a plain `Categorical`. It
accepts any callable score, including `Base.Fix1(logpdf, p)`.

```julia
ef = convert(ExponentialFamilyDistribution, q)
BayesBase.expectation_gradient(ClosedFormExpectation(), k -> scores[k], ef) # ≈ [-0.36, -0.24, 0.6]
```

## [Exponential Distribution](@id lib-exponential)

Distribution ``q \sim \mathrm{Exponential}(\lambda)``, where ``\lambda`` is the scale (mean).
The gradients are vectors of length one, with respect to ``\lambda``.

### Expectations

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x]`` |
| `Base.Fix1(logpdf, Exponential(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Exp}}(x)]`` |
| `Base.Fix1(logpdf, LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2}\right]`` |

### Gradients

| Function `f` | Returns |
|:-------------|:--------|
| `log` | ``\mathbb{E}_q[\log x \cdot \nabla_\lambda \log q(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2} \cdot \nabla_\lambda \log q(x)\right]`` |
| `Base.Fix1(logpdf, Exponential(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Exp}}(x) \cdot \nabla_\lambda \log q(x)]`` |
| `Base.Fix1(logpdf, LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x) \cdot \nabla_\lambda \log q(x)]`` |

## [Gamma Distribution](@id lib-gamma)

Distribution ``q \sim \mathrm{Gamma}(\alpha, \theta)``, where ``\alpha`` is the shape and ``\theta`` is the scale.

!!! note
    For an expectation, a `GammaShapeRate` is converted to a `Gamma`. The gradients are defined under a `Gamma` and its exponential family form.

### Expectations

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x]`` |
| `xlogx` | ``\mathbb{E}_q[x \log x]`` |
| `xlog2x` | ``\mathbb{E}_q[x (\log x)^2]`` |
| `Square() ∘ log` | ``\mathbb{E}_q[(\log x)^2]`` |
| `Power(Val(3)) ∘ log` | ``\mathbb{E}_q[(\log x)^3]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q\left[-\frac{(\log x - \mu)^2}{2\sigma^2}\right]`` |
| `Base.Fix1(logpdf, LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `Base.Fix1(logpdf, Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x)]`` |
| `Base.Fix1(logpdf, Normal(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Normal}}(x)]`` |
| `Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x))` | ``\mathbb{E}_q[\log m_{f \to y}(y)]``, ``m_{f \to y} = \Phi(-\alpha_x)\delta(y) + \mathbf{1}_{y>0}\mathcal{N}(y;\,m_x,v_x)`` |
| `Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

!!! note "Why the forward message collapses to a Gaussian integral here"
    The forward message ``m_{f \to y}`` is a **spike-slab** distribution, not a Gaussian.
    Its log-density is ``\log\mathcal{N}(y;\,m_x,v_x)`` for ``y > 0`` and ``-\infty`` for ``y \leq 0``.
    Because Gamma (and LogNormal) have support on ``(0,\infty)``, the atom at ``y=0`` carries
    zero mass under ``q``, so the expectation equals ``\mathbb{E}_q[\log\mathcal{N}(y;\,m_x,v_x)]``
    numerically — but this is a property of the support of ``q``, not of the message itself.
    For ``q = \mathcal{N}(\mu,\sigma^2)`` the atom contributes ``-\infty`` and the expectation
    is undefined; see the Normal section for the backward message instead.

### Gradients

A 2-element `SVector` with the derivatives ``[\nabla_\alpha, \nabla_\theta]``.

| Function `f` | Expression |
|:-------------|:-----------|
| `log` | ``\mathbb{E}_q[\log x \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Square() ∘ log` | ``\mathbb{E}_q[(\log x)^2 \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `log ∘ ExpLogSquare(μ, σ)` | ``\mathbb{E}_q[\ldots \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Base.Fix1(logpdf, LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Base.Fix1(logpdf, Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |
| `Base.Fix1(logpdf, Normal(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Normal}}(x) \cdot \nabla_{(\alpha,\theta)} \log q(x)]`` |

## [Normal Distribution](@id lib-normal)

Distribution ``q \sim \mathcal{N}(\mu, \sigma^2)``.

!!! note
    Any `UnivariateNormalDistributionsFamily` type is accepted for expectations, including `Normal`, `NormalMeanVariance`, `NormalMeanPrecision`, and `NormalWeightedMeanPrecision`. The gradients are defined under `Normal(μ, σ)`, and through the Jacobian under `NormalMeanVariance` and `ExponentialFamilyDistribution{NormalMeanVariance}`.

### Expectations

| Function `f` | Expression |
|:-------------|:-----------|
| `Base.Fix1(logpdf, Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x)]`` |
| `Base.Fix1(logpdf, Laplace(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Lap}}(x)]`` |
| `Abs()` | ``\mathbb{E}_q[\lvert x \rvert]`` |
| `Base.Fix1(logpdf, LogGamma(α, β))` | ``\mathbb{E}_q[\log p_{\mathrm{LG}}(x)]`` |
| `LogGammaKernel(α, β)` | ``\mathbb{E}_q[\beta x - e^x / \alpha]``, for any real ``\beta`` |
| `Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

### Gradients

A 2-element `SVector` with the derivatives ``[\nabla_\mu, \nabla_\sigma]``.

| Function `f` | Expression |
|:-------------|:-----------|
| `Abs()` | ``\mathbb{E}_q[\lvert x \rvert \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Base.Fix1(logpdf, Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Base.Fix1(logpdf, Laplace(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Lap}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `Base.Fix1(logpdf, LogGamma(α, β))` | ``\mathbb{E}_q[\log p_{\mathrm{LG}}(x) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |
| `LogGammaKernel(α, β)` | ``\mathbb{E}_q[(\beta x - e^x / \alpha) \cdot \nabla_{(\mu,\sigma)} \log q(x)]`` |

## [LogNormal Distribution](@id lib-lognormal)

Distribution ``q \sim \mathrm{LogNormal}(\mu, \sigma)``, where ``\mu`` is the log-mean and ``\sigma`` is the log-standard-deviation.

### Expectations

| Function `f` | Expression |
|:-------------|:-----------|
| `Base.Fix1(logpdf, Gamma(...))` | ``\mathbb{E}_q[\log p_{\mathrm{Gamma}}(x)]`` |
| `Base.Fix1(logpdf, Normal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(x)]`` |
| `Base.Fix1(logpdf, LogNormal(μ, σ))` | ``\mathbb{E}_q[\log p_{\mathrm{LogN}}(x)]`` |
| `Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x))` | ``\mathbb{E}_q[\log m_{f \to y}(y)]``, ``m_{f \to y} = \Phi(-\alpha_x)\delta(y) + \mathbf{1}_{y>0}\mathcal{N}(y;\,m_x,v_x)`` |
| `Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y))` | ``\mathbb{E}_q[\log m_{f \to x}(x)]``, ``m_{f \to x}(x) \propto \mathcal{N}(\max(0,x);\,m_y,v_y)`` |

## [Multivariate Normal Distribution](@id lib-mvnormal)

Distribution ``q \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})``.

!!! note
    Any `MultivariateNormalDistributionsFamily` type is accepted.

### Expectations

| Function `f` | Expression |
|:-------------|:-----------|
| `Base.Fix1(logpdf, MvNormal(...))` | ``\mathbb{E}_q[\log p_{\mathcal{N}}(\mathbf{x})]`` |
| `Base.Fix1(logpdf, LinearLogGamma(α, β, w))` | ``\mathbb{E}_q[\log p_{\mathrm{LLG}}(\mathbf{x})]`` |

## [ExponentialFamily Parametrizations](@id lib-ef-pairs)

The gradients are also defined under these ExponentialFamily parametrizations:

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

This holds for the expectation and for its gradient, and for nested products. Each part must have a closed form.
