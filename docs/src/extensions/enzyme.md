# [Enzyme Extension](@id ext-enzyme)

The `ClosedFormExpectationsEnzymeExt` extension is loaded when
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is loaded. With it, the
[`ClosedFormAD`](@ref) backend computes the gradient of an expectation by differentiating the
closed-form expectation with Enzyme, so a gradient without a hand-written formula can still be
computed exactly.

## Motivation

The gradient of an expectation with respect to the natural parameters ``\eta`` of ``q`` satisfies

```math
\mathbb{E}_q\!\left[f(x)\,\nabla_\eta \log q(x;\eta)\right]
= \nabla_\eta\,\mathbb{E}_q[f(x)]
```

The right-hand side is the gradient of an expectation the package already computes, so Enzyme can
compute it without a hand-derived formula.

## The backend

`ClosedFormAD(ad)` takes an automatic differentiation backend of
[ADTypes.jl](https://github.com/SciML/ADTypes.jl). With this extension, `AutoEnzyme` is supported:

| Constructor | Mode |
|:------------|:-----|
| `ClosedFormAD(AutoEnzyme())` | Reverse, the default |
| `ClosedFormAD(AutoEnzyme(mode = Enzyme.Reverse))` | Reverse |
| `ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward))` | Forward, one pass per parameter |

The backend computes only gradients, and only where `ClosedFormExpectation()` has the
expectation. It is meant to follow `ClosedFormExpectation()` in a router of BayesBase, so that the
hand-written formulas are used where they exist and automatic differentiation for the rest:

```julia
using ClosedFormExpectations, BayesBase, Distributions, ExponentialFamily, Enzyme
import ADTypes: AutoEnzyme

backend = BayesBase.ExpectationFirstOf(ClosedFormExpectation(), ClosedFormAD(AutoEnzyme()))
q = convert(ExponentialFamilyDistribution, LogNormal(1.0, 0.5))

# No formula for this gradient: Enzyme differentiates the closed-form expectation
BayesBase.expectation_gradient(backend, Base.Fix1(logpdf, Gamma(2.0, 3.0)), q)

# A formula: the router uses it
BayesBase.expectation_gradient(backend, log, convert(ExponentialFamilyDistribution, Gamma(2.0, 3.0)))
```

A numerical backend, such as a quadrature rule of ExpectationApproximations.jl, can follow in the
same router for the pairs without a closed form.

## Supported scope

The extension computes the gradient under a univariate `ExponentialFamilyDistribution{T}`, with
respect to its natural parameters, wherever
`BayesBase.expectation(ClosedFormExpectation(), f, q)` has a closed form, including through the
conversion of `q` to its standard Distributions.jl form.
`BayesBase.supports_expectation_gradient(ClosedFormAD(AutoEnzyme()), f, q)` is `false` everywhere
else: under other distributions, and under multivariate exponential family forms, whose closed
forms go through matrix factorisations that Enzyme does not differentiate.

!!! note
    The hand-written formulas of `ClosedFormExpectation()` avoid the cost of automatic
    differentiation and are exact in any parametrization they are written in, so put
    `ClosedFormExpectation()` first.
