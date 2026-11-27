ClosedFormExpectations.jl
=========================

*Julia package for computing closed-form expectations of functions with respect to probability distributions.*

## Overview

`ClosedFormExpectations.jl` provides closed-form expressions for computing expectations of the form

```math
\mathbb{E}_q[f(x)]
```

where ``q`` is a probability distribution and ``f`` is a function (e.g., `log`, `logpdf`). This is particularly useful in variational inference, where such expectations appear frequently in the evidence lower bound (ELBO) and its gradients.

The package also computes the gradient of such an expectation with respect to the parameters
``\theta`` of ``q``, which equals the Williams product, or score function gradient:

```math
\nabla_\theta \mathbb{E}_q[f(x)] = \mathbb{E}_q[f(x) \nabla_\theta \log q(x; \theta)]
```

Natural-gradient and stochastic variational inference use it. Both are computed through the
expectation protocol of [BayesBase](https://github.com/ReactiveBayes/BayesBase.jl), with the backend
`ClosedFormExpectation()`.

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager:

```julia
pkg> add ClosedFormExpectations
```

## Quick Start

```julia
using ClosedFormExpectations, BayesBase, Distributions

# E_q[log(x)] where q = Exponential(10)
BayesBase.expectation(ClosedFormExpectation(), log, Exponential(10))

# E_q[log p(x)] where p = LogNormal(1, 10), q = Exponential(10)
BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(1, 10)), Exponential(10))

# The gradient of E_q[log(x)] with respect to the shape and the scale of q = Gamma(2, 3)
BayesBase.expectation_gradient(ClosedFormExpectation(), log, Gamma(2, 3))
```

## Table of Contents

```@contents
Pages = [
  "lib/api.md",
  "lib/expressions.md",
  "lib/supported-pairs.md",
  "lib/distributions.md",
  "lib/protocol.md",
  "extensions/enzyme.md",
  "extra/contributing.md",
]
Depth = 2
```

## Index

```@index
```
