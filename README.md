# ClosedFormExpectations.jl

[![Build Status](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)

`ClosedFormExpectations.jl` is a Julia package that provides closed-form expressions for computing the expectation of a function (e.g pdfs) with respect to another distribution, i.e., $E_q[f(x)].$

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add ClosedFormExpectations
```

The package exports the following:

`ClosedFormExpectation`: A struct representing the closed-form expectation

`ClosedWilliamsProduct`: A struct representing the closed-form expectation of the product of score function and a target function (the gradient of the `ClosedFormExpectation`)

## Usage

```julia
using ClosedFormExpectations
using Distributions
mean(ClosedFormExpectation(), log, Exponetial(10))
```
