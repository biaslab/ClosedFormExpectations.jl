# ClosedFormExpectations.jl

[![Build Status](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/biaslab/ClosedFormExpectations.jl/graph/badge.svg?token=2syiPm7b6L)](https://codecov.io/gh/biaslab/ClosedFormExpectations.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://biaslab.github.io/ClosedFormExpectations.jl/dev/)


`ClosedFormExpectations.jl` is a Julia package that provides closed-form expressions for computing the expectation of a function (e.g pdfs) with respect to another distribution, i.e., $E_q[f(x)].$

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add ClosedFormExpectations
```

## Usage

The package defines `ClosedFormExpectation()`, a backend of the expectation protocol of
[BayesBase](https://github.com/ReactiveBayes/BayesBase.jl). `BayesBase.expectation` computes the
expectation of `f` under `q`, and `BayesBase.expectation_gradient` its gradient with respect to the
parameters of `q`. The expectation of a log-density, `E_q[log p]`, takes the integrand
`Base.Fix1(logpdf, p)`.

```julia
using ClosedFormExpectations, BayesBase, Distributions

BayesBase.expectation(ClosedFormExpectation(), log, Exponential(10))
BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(1, 10)), Exponential(10))

# The gradient with respect to the shape and the scale of the Gamma
BayesBase.expectation_gradient(ClosedFormExpectation(), log, Gamma(2, 3))

# Whether a pair has a closed form
BayesBase.supports_expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Beta(2, 2)), Normal(0, 1)) # false
```

The backend labels its results exact and is registered when the package is loaded, so inference
packages that choose among the loaded backends, such as RxInfer, use the closed forms where they
exist. With Enzyme.jl loaded, `ClosedFormAD(AutoEnzyme())` differentiates a closed-form
expectation where the package has no formula for its gradient. Version 1.0 replaced
`mean(ClosedFormExpectation(), f, q)` and `mean(ClosedWilliamsProduct(), f, q)` with these
functions; the documentation has a table for migrating.
