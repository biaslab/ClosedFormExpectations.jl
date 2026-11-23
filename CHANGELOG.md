# Changelog

All notable changes to ClosedFormExpectations.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

The package now computes through the expectation protocol of BayesBase 1.7, with
`ClosedFormExpectation()` as a backend. The table "Migrating from 0.4" in the API reference lists
the replacements.

### Changed

- **Breaking:** `mean(ClosedFormExpectation(), f, q)` is replaced by
  `BayesBase.expectation(ClosedFormExpectation(), f, q)`, and `mean` is no longer exported.
- **Breaking:** the Williams product `mean(ClosedWilliamsProduct(), f, q)` is replaced by the
  gradient of the expectation, `BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)`, in
  the same parametrization. `ClosedWilliamsProduct` is removed, and
  `ClosedWilliamsProduct(EnzymeBackend())` is replaced by `ClosedFormExpectation(EnzymeBackend())`.
  `ClosedFormExpectation` has a type parameter and a field, `gradient`, for this.
- **Breaking:** the `Logpdf` wrapper is removed. The integrand of `E_q[log p]` is
  `Base.Fix1(logpdf, p)`, and a distribution is no longer accepted as an integrand.
- **Breaking:** the gradients under an `Exponential` are vectors of length one instead of numbers.
- Every closed form is defined with BayesBase's `@supports_expectation` or
  `@supports_expectation_gradient`, so `BayesBase.supports_expectation` and
  `BayesBase.supports_expectation_gradient` say exactly which pairs have one. A test checks that
  every closed form declares its support.

### Added

- `ClosedFormExpectation()` is a backend of BayesBase's expectation protocol, labelled exact and
  registered when the package is loaded, so `BayesBase.ExpectationDynamicRouter()` chooses it for
  its pairs. A `GammaShapeRate` and an `ExponentialFamilyDistribution` are converted for the
  expectation, as before.
- Routers of the protocol compute a gradient in closed form where the package has one, and with
  a numerical backend otherwise.
- BayesBase 1.7 is required.
