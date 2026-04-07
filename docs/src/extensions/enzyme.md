# [Enzyme Extension](@id ext-enzyme)

The `ClosedFormExpectationsEnzymeExt` extension is automatically loaded when
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) is present in the environment.
It provides an automatic differentiation backend for
[`ClosedWilliamsProduct`](@ref lib-core-types)
that eliminates the need to hand-code Jacobians for new distribution families.

## Motivation

The Williams' product satisfies the identity

```math
\mathbb{E}_q\!\left[f(x)\,\nabla_\eta \log q(x;\eta)\right]
= \nabla_\eta\,\mathbb{E}_q[f(x)]
```

where ``\eta`` are the natural parameters of ``q``.  
Because the right-hand side is just the gradient of the *already-implemented*
[`ClosedFormExpectation`](@ref lib-core-types), Enzyme can compute it automatically —
no manual Jacobian derivation required.

## Backends

Two AD modes are available, selected via the `backend` field of
[`ClosedWilliamsProduct`](@ref lib-core-types):

| Constructor | Mode | Notes |
|:------------|:-----|:------|
| `ClosedWilliamsProduct(EnzymeBackend())` | Reverse (default) | One backward pass; efficient when there are few outputs |
| `ClosedWilliamsProduct(EnzymeBackend(EnzymeReverse()))` | Reverse (explicit) | Same as above |
| `ClosedWilliamsProduct(EnzymeBackend(EnzymeForward()))` | Forward | One forward pass per parameter; good for low-dimensional ``\eta`` |

```@docs
EnzymeBackend
EnzymeReverse
EnzymeForward
```

## Usage

```julia
using ClosedFormExpectations, Distributions, ExponentialFamily, Enzyme

# Any ExponentialFamilyDistribution{T} for which ClosedFormExpectation is defined
# automatically gets a Williams' product via Enzyme — no extra code needed.

ef = convert(ExponentialFamilyDistribution, LogNormal(1.0, 0.5))

# Reverse-mode (default)
mean(ClosedWilliamsProduct(EnzymeBackend()), Logpdf(Gamma(2.0, 3.0)), ef)

# Forward-mode
mean(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(Gamma(2.0, 3.0)), ef)
```

The default `ClosedWilliamsProduct()` (no backend) is unchanged and still routes to
hand-coded implementations where available. See [Core Types](@ref lib-core-types).

## When to use it

Use `EnzymeBackend` when:

- You have added a new distribution family with a `ClosedFormExpectation` but have **not**
  yet written a hand-coded `ClosedWilliamsProduct`.
- You want to **prototype** quickly and verify correctness before committing to a manual
  derivation.
- The distribution is wrapped as an `ExponentialFamilyDistribution{T}` — the extension
  handles the natural-parameter Jacobian automatically.

!!! note
    Hand-coded implementations (registered via `mean_ef_impl` for `ClosedWilliamsProduct{Nothing}`)
    are still preferred when available, as they avoid AD overhead. The Enzyme backend acts
    as a generic fallback for any `ExponentialFamilyDistribution{T}` not yet covered.

## Supported scope

The extension works for **any** `ExponentialFamilyDistribution{T}` where
`mean(ClosedFormExpectation(), f, q)` is defined (including via automatic conversion
from `ExponentialFamilyDistribution` to a standard `Distributions.jl` type).

This covers all distributions listed in the [Supported Pairs](@ref lib-supported-pairs)
table when passed as `ExponentialFamilyDistribution` wrappers.
