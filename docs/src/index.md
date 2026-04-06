ClosedFormExpectations.jl
=========================

*Julia package for computing closed-form expectations of functions with respect to probability distributions.*

## Overview

`ClosedFormExpectations.jl` provides closed-form expressions for computing expectations of the form

```math
\mathbb{E}_q[f(x)]
```

where ``q`` is a probability distribution and ``f`` is a function (e.g., `log`, `logpdf`). This is particularly useful in variational inference, where such expectations appear frequently in the evidence lower bound (ELBO) and its gradients.

The package also supports computing the **Williams' product** (score function gradient):

```math
\mathbb{E}_q[f(x) \nabla_\theta \log q(x; \theta)]
```

which is used for natural gradient and stochastic variational inference methods.

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager:

```julia
pkg> add ClosedFormExpectations
```

## Quick Start

```julia
using ClosedFormExpectations, Distributions

# E_q[log(x)] where q = Exponential(10)
mean(ClosedFormExpectation(), log, Exponential(10))

# E_q[log p(x)] where p = LogNormal(1, 10), q = Exponential(10)
mean(ClosedFormExpectation(), Logpdf(LogNormal(1, 10)), Exponential(10))

# Williams' product: E_q[log(x) * ∇_θ log q(x; θ)] where q = Gamma(2, 3)
mean(ClosedWilliamsProduct(), log, Gamma(2, 3))
```

## Table of Contents

```@contents
Pages = [
  "lib/api.md",
  "lib/expressions.md",
  "lib/supported-pairs.md",
  "lib/distributions.md",
  "extra/contributing.md",
]
Depth = 2
```

## Index

```@index
```
