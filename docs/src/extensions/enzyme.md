# [Enzyme Extension](@id ext-enzyme)

The `ClosedFormExpectationsEnzymeExt` extension is loaded when
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is loaded. It computes the gradient of an
expectation by differentiating the closed-form expectation with Enzyme, so a new distribution
family does not need a hand-written Jacobian.

## Motivation

The gradient of an expectation with respect to the natural parameters ``\eta`` of ``q`` satisfies

```math
\mathbb{E}_q\!\left[f(x)\,\nabla_\eta \log q(x;\eta)\right]
= \nabla_\eta\,\mathbb{E}_q[f(x)]
```

The right-hand side is the gradient of an expectation the package already computes, so Enzyme can
compute it without a hand-derived formula.

## Backends

The AD mode is the `gradient` field of [`ClosedFormExpectation`](@ref lib-core-types):

| Constructor | Mode | Notes |
|:------------|:-----|:------|
| `ClosedFormExpectation(EnzymeBackend())` | Reverse (default) | One backward pass; efficient when there are few outputs |
| `ClosedFormExpectation(EnzymeBackend(EnzymeReverse()))` | Reverse (explicit) | Same as above |
| `ClosedFormExpectation(EnzymeBackend(EnzymeForward()))` | Forward | One forward pass per parameter; good for low-dimensional ``\eta`` |

Expectations are computed by these backends as by `ClosedFormExpectation()`. Only gradients differ.

```@docs
EnzymeBackend
EnzymeReverse
EnzymeForward
```

## Usage

```julia
using ClosedFormExpectations, BayesBase, Distributions, ExponentialFamily, Enzyme

ef = convert(ExponentialFamilyDistribution, LogNormal(1.0, 0.5))

# Reverse mode (default)
BayesBase.expectation_gradient(ClosedFormExpectation(EnzymeBackend()), Base.Fix1(logpdf, Gamma(2.0, 3.0)), ef)

# Forward mode
BayesBase.expectation_gradient(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, Gamma(2.0, 3.0)), ef)
```

`ClosedFormExpectation()` uses the formulas written in the package. See
[The Backend](@ref lib-core-types).

## When to use it

Use `EnzymeBackend` when:

- you have added a closed-form expectation for a new distribution family but no formula for its
  gradient;
- you want to check a new formula against automatic differentiation;
- the distribution is an `ExponentialFamilyDistribution{T}`, whose natural parameters the
  extension differentiates.

!!! note
    The hand-written formulas of `ClosedFormExpectation()` avoid the cost of automatic
    differentiation, so use them where they exist.

## Supported scope

The extension computes the gradient under any `ExponentialFamilyDistribution{T}` for which
`BayesBase.expectation(ClosedFormExpectation(), f, q)` has a closed form, including through the
conversion of `q` to its standard Distributions.jl form. Under other distributions it supports
nothing, and `BayesBase.supports_expectation_gradient` says so.
