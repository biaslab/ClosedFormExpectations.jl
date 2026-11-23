# [API Reference](@id lib-api)

## [The Backend](@id lib-core-types)

```@docs
ClosedFormExpectation
```

## [Computing Expectations and Gradients](@id lib-computing-expectations)

The package computes through the expectation protocol of BayesBase:

```julia
# The expectation E_q[f(x)]
BayesBase.expectation(ClosedFormExpectation(), f, q)

# Its gradient with respect to the parameters θ of q, ∇_θ E_q[f(x)] = E_q[f(x) ∇_θ log q(x; θ)]
BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)

# Whether a pair has a closed form
BayesBase.supports_expectation(ClosedFormExpectation(), f, q)
BayesBase.supports_expectation_gradient(ClosedFormExpectation(), f, q)
```

`f` is the integrand. It can be a function such as `log`, the log-density `Base.Fix1(logpdf, p)`
of a distribution `p`, or an expression such as `Square() ∘ log` or `log ∘ ExpLogSquare(μ, σ)`.
`q` is the distribution the expectation is taken under. A pair without a closed form throws a
`BayesBase.UnsupportedExpectationError`.

The expectation is a number. The gradient is a vector, in the parametrization of `q`: with respect
to `params(q)` for a distribution of Distributions.jl, with respect to the mean and the variance
for a `NormalMeanVariance`, and with respect to the natural parameters for an
`ExponentialFamilyDistribution`. For the exponential family form of a `Categorical` it is the full
vector of length K in the softmax logits.

```jldoctest
julia> using ClosedFormExpectations, BayesBase, Distributions

julia> BayesBase.expectation(ClosedFormExpectation(), log, Gamma(2.0, 1.0)) ≈ 1 - Base.MathConstants.eulergamma
true

julia> BayesBase.expectation_gradient(ClosedFormExpectation(), log, Gamma(2.0, 1.0)) ≈ [π^2 / 6 - 1, 1.0]
true

julia> BayesBase.supports_expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Beta(2.0, 2.0)), Normal(0.0, 1.0))
false
```

## [ExponentialFamily Support](@id lib-ef-support)

The package works with the `ExponentialFamilyDistribution` objects of
[ExponentialFamily.jl](https://github.com/ReactiveBayes/ExponentialFamily.jl). For an expectation,
`q` is converted to its standard Distributions.jl form. The gradient is with respect to the natural
parameters, and the package has formulas for:
- `NormalMeanVariance`, through the Jacobian from ``(\mu, \sigma)`` to ``(\mu, v)``;
- `ExponentialFamilyDistribution{NormalMeanVariance}`, through the Jacobian to the natural parameters;
- `ExponentialFamilyDistribution{Gamma}`, likewise;
- `ExponentialFamilyDistribution{Categorical}`, the full softmax-logit score gradient, see
  [Categorical Distribution](@ref lib-categorical).

For any other `ExponentialFamilyDistribution{T}` whose expectation has a closed form, load Enzyme.jl
and use `ClosedFormExpectation(EnzymeBackend())`, see the [Enzyme extension](@ref ext-enzyme).

## [ProductOf Support](@id lib-productof)

The log-density of a `ProductOf` distribution from ExponentialFamily.jl is the sum of the
log-densities of its parts:

```math
\mathbb{E}_q[\log p_1(x) p_2(x)] = \mathbb{E}_q[\log p_1(x)] + \mathbb{E}_q[\log p_2(x)]
```

So `Base.Fix1(logpdf, prod(GenericProd(), p₁, p₂))` has a closed form, and its gradient too,
wherever both parts have one. Nested products are split the same way.

## [Migrating from 0.4](@id lib-migrating)

Version 1.0 computes everything through BayesBase's protocol. The `mean` methods, the
`ClosedWilliamsProduct` strategy and the `Logpdf` wrapper are removed.

| 0.4 | 1.0 |
|:----|:----|
| `mean(ClosedFormExpectation(), f, q)` | `BayesBase.expectation(ClosedFormExpectation(), f, q)` |
| `mean(ClosedWilliamsProduct(), f, q)` | `BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)` |
| `mean(ClosedWilliamsProduct(EnzymeBackend()), f, q)` | `BayesBase.expectation_gradient(ClosedFormExpectation(EnzymeBackend()), f, q)` |
| `Logpdf(p)`, or the distribution `p` as the integrand | `Base.Fix1(logpdf, p)` |
| a closed form under an `Exponential`, a number | a vector of length one, with respect to the scale |
