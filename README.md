# ClosedFormExpectations.jl

[![Build Status](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/biaslab/ClosedFormExpectations.jl/graph/badge.svg?token=2syiPm7b6L)](https://codecov.io/gh/biaslab/ClosedFormExpectations.jl)

`ClosedFormExpectations.jl` is a Julia package that provides closed-form expressions for computing the expectation of a function (e.g pdfs) with respect to another distribution, i.e., $E_q[f(x)].$

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add ClosedFormExpectations
```

The package exports the following:

`ClosedFormExpectation`: A struct representing the closed-form expectation

`ClosedWilliamsProduct`: A struct representing the closed-form expectation of the product of score function and a target function (the gradient of the `ClosedFormExpectation`)

`Logpdf`: A struct to represent the logpdf function of a distribution

## Usage

```julia
using ClosedFormExpectations
using Distributions
mean(ClosedFormExpectation(), log, Exponential(10))
mean(ClosedFormExpectation(), Logpdf(LogNormal(1, 10)), Exponential(10))
mean(ClosedFormExpectation(), Base.Fix1(logpdf, Exponential(1)), Exponential(10))
```
