# [Custom Distributions](@id lib-custom-distributions)

The package defines additional distribution types that are not part of [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

## [LogGamma](@id lib-loggamma)

```@docs
LogGamma
```

The LogGamma distribution is a continuous distribution on the real line with probability density function:

```math
\mathcal{LG}(x \mid \alpha, \beta) = \frac{e^{\beta x} \, e^{-e^x / \alpha}}{\alpha^\beta \, \Gamma(\beta)}, \quad -\infty < x < \infty, \; \alpha > 0, \; \beta > 0
```

where ``\alpha`` is the scale parameter and ``\beta`` is the shape parameter.

**Reference:** [Log-Gamma distribution (William & Mary)](https://www.math.wm.edu/~leemis/chart/UDR/PDFs/Loggamma.pdf)

### Usage

```julia
using ClosedFormExpectations, Distributions

d = LogGamma(2.0, 3.0)
logpdf(d, 0.5)

# Used as a target in expectations
mean(ClosedFormExpectation(), Logpdf(LogGamma(2.0, 3.0)), Normal(0, 1))
```

## [ReLU Factor Messages](@id lib-relu-messages)

These types represent the exact factor-to-variable messages of a hard ReLU factor node
``y = \max(0, x)``, given Gaussian incoming messages on either side.

```@docs
ReLUForwardMessage
ReLUBackwardMessage
```

### Usage

```julia
using ClosedFormExpectations, Distributions, BayesBase

# Forward message: given Normal incoming on x, evaluate log density at y > 0
d_fwd = ReLUForwardMessage(1.0, 2.0)   # m_x = 1.0, v_x = 2.0
logpdf(d_fwd, 0.5)                     # log 𝒩(0.5; 1.0, 2.0)
logpdf(d_fwd, -0.1)                    # -Inf (no support for y ≤ 0)

# Backward message: given Normal incoming on y, evaluate log density at any x
d_bwd = ReLUBackwardMessage(1.0, 2.0)  # m_y = 1.0, v_y = 2.0
logpdf(d_bwd, 0.5)                     # log 𝒩(0.5; 1.0, 2.0)
logpdf(d_bwd, -0.5)                    # log 𝒩(0.0; 1.0, 2.0)  (constant for x ≤ 0)

# Closed-form expectations
mean(ClosedFormExpectation(), Logpdf(ReLUForwardMessage(1.0, 2.0)), Gamma(3.0, 1.0))
mean(ClosedFormExpectation(), Logpdf(ReLUBackwardMessage(1.0, 2.0)), Normal(0.5, 1.0))
```

## [LinearLogGamma](@id lib-linearloggamma)

```@docs
LinearLogGamma
```

The LinearLogGamma distribution is a multivariate distribution derived from the LogGamma distribution via a linear projection:

```math
\mathrm{LLG}(\mathbf{x} \mid \alpha, \beta, \mathbf{w}) = \mathcal{LG}(\mathbf{w}^\top \mathbf{x} \mid \alpha, \beta)
```

where ``\mathbf{w}`` is a fixed vector of weights/covariates.

### Usage

```julia
using ClosedFormExpectations, Distributions

d = LinearLogGamma(2.0, 3.0, [0.5, 0.3, 0.2])
logpdf(d, [1.0, 2.0, 3.0])

# Used as a target in expectations with multivariate Normal
mean(ClosedFormExpectation(), Logpdf(LinearLogGamma(2.0, 3.0, [0.5, 0.3])), MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]))
```
