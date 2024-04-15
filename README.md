# ClosedFormExpectations.jl

[![Build Status](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/ClosedFormExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)

Here's a sample README for your package:

`ClosedFormExpectations.jl` is a Julia package that provides closed-form expressions for computing the expectation of a function (e.g pdfs) with respect to another distribution, i.e., $E_q[f(x)].$

## Installation

You can install `ClosedFormExpectations.jl` using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add ClosedFormExpectations
```

The package exports the following:

ClosedFormExpectation: A struct representing the closed-form expectation.

meanlog: A function to compute the expectation E_q[log p(x)], where q is a distribution and p is a function.

## Usage

```julia
using ClosedFormExpectations
using Distributions
mean(ClosedFormExpectation(), Exponetial(10), log)
```