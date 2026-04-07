# [API Reference](@id lib-api)

## [Core Types](@id lib-core-types)

The package provides two main computation strategies, represented as singleton types.

```@docs
ClosedFormExpectation
```

```@docs
ClosedWilliamsProduct
```

```@docs
mean(::ClosedFormExpectation, ::Nothing, ::Nothing)
mean(::ClosedWilliamsProduct, ::Nothing, ::Nothing)
```

### [Computing Expectations](@id lib-computing-expectations)

All expectations are computed via the `mean` function:

```julia
# Closed-form expectation: E_q[f(x)]
mean(ClosedFormExpectation(), f, q)

# Williams' product: E_q[f(x) ∇_θ log q(x; θ)]
mean(ClosedWilliamsProduct(), f, q)
```

**Arguments:**
- The first argument selects the computation strategy (`ClosedFormExpectation` or `ClosedWilliamsProduct`).
- `f` is the function whose expectation is computed. It can be:
  - A built-in function like `log`
  - A `Logpdf(dist)` wrapper
  - A `Base.Fix1(logpdf, dist)` partial application
  - A raw `Distribution` (automatically wrapped as `Logpdf`)
  - A composed expression like `Square() ∘ log` or `log ∘ ExpLogSquare(μ, σ)`
- `q` is the distribution to take the expectation with respect to.

**Return values:**
- `ClosedFormExpectation` returns a scalar (or vector for multivariate distributions).
- `ClosedWilliamsProduct` returns a static vector (gradient with respect to distribution parameters).

## [Logpdf Wrapper](@id lib-logpdf)

```@docs
Logpdf
```

The `Logpdf` wrapper allows computing expectations of log-probability density functions. Several convenience dispatches are provided:

```julia
# These are all equivalent:
mean(ClosedFormExpectation(), Logpdf(Normal(0, 1)), q)
mean(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(0, 1)), q)
mean(ClosedFormExpectation(), Normal(0, 1), q)
```

## [ExponentialFamily Support](@id lib-ef-support)

The package integrates with [ExponentialFamily.jl](https://github.com/ReactiveBayes/ExponentialFamily.jl) to support `ExponentialFamilyDistribution` objects. When an `ExponentialFamilyDistribution` is passed as the distribution `q`, it is automatically converted to its standard `Distributions.jl` representation for `ClosedFormExpectation`, and the appropriate Jacobian transformation is applied for `ClosedWilliamsProduct`.

Supported ExponentialFamily parametrizations with hand-coded Jacobians:
- `NormalMeanVariance` (with Jacobian adjustment for Williams' product)
- `ExponentialFamilyDistribution{NormalMeanVariance}` (with natural parameter Jacobian)
- `ExponentialFamilyDistribution{Gamma}` (with natural parameter Jacobian)

For any other `ExponentialFamilyDistribution{T}` family where a `ClosedFormExpectation` exists
but no hand-coded `ClosedWilliamsProduct` is available, load Enzyme.jl and use the
[Enzyme extension](@ref ext-enzyme) to compute the Williams' product automatically via AD.

## [ProductOf Support](@id lib-productof)

The package supports `ProductOf` distributions from [ExponentialFamily.jl](https://github.com/ReactiveBayes/ExponentialFamily.jl). When computing the expectation of a `Logpdf{ProductOf}`, the product is decomposed additively:

```math
\mathbb{E}_q[\log p_1(x) p_2(x)] = \mathbb{E}_q[\log p_1(x)] + \mathbb{E}_q[\log p_2(x)]
```

This works recursively for nested products and is supported for both `ClosedFormExpectation` and `ClosedWilliamsProduct`.
