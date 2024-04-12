# LogExpectations.jl

[![Build Status](https://github.com/biaslab/LogExpectations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/LogExpectations.jl/actions/workflows/CI.yml?query=branch%3Amain)

Here's a sample README for your package:

`LogExpectations.jl` is a Julia package that provides closed-form expressions for computing the expectation of the logarithm of a function (e.g pdfs) with respect to another distribution, i.e., $E_q[log p(x)].$

## Installation

You can install `LogExpectations.jl` using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add LogExpectations
```

The package exports the following:

ClosedFormExpectation: A struct representing the closed-form expectation.

meanlog: A function to compute the expectation E_q[log p(x)], where q is a distribution and p is a function.

## Usage

```julia
using LogExpectations
using Distributions
meanlog(ClosedFormExpectation(), q, p)
```