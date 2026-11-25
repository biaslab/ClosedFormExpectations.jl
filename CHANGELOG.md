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
  the same parametrization. `ClosedWilliamsProduct` is removed.
- **Breaking:** `ClosedWilliamsProduct(EnzymeBackend())` is replaced by a separate backend,
  `ClosedFormAD(AutoEnzyme())`, with `AutoEnzyme` from ADTypes.jl and Enzyme.jl loaded. The
  `EnzymeBackend`, `EnzymeForward` and `EnzymeReverse` types are removed: the mode is
  `AutoEnzyme(mode = Enzyme.Forward)`. `ClosedFormAD` computes gradients only under univariate
  exponential family forms, and is meant to follow `ClosedFormExpectation()` in a router, so that
  the hand-written formulas are used where they exist.
- **Breaking:** the `Logpdf` wrapper is removed. The integrand of `E_q[log p]` is
  `Base.Fix1(logpdf, p)`, and a distribution is no longer accepted as an integrand.
- **Breaking:** the gradients under an `Exponential` are vectors of length one instead of numbers.
- **Breaking:** the closed forms for a univariate Normal `q` are declared for the univariate normal
  family only. Before, they were also reached under multivariate normals, where they failed.
- Every closed form is defined with BayesBase's `@supports_expectation` or
  `@supports_expectation_gradient`, so `BayesBase.supports_expectation` and
  `BayesBase.supports_expectation_gradient` say exactly which pairs have one. A test checks that
  every closed form declares its support.

### Added

- `ClosedFormExpectation()` is a backend of BayesBase's expectation protocol, labelled exact and
  registered when the package is loaded, so `BayesBase.ExpectationDynamicRouter()` chooses it for
  its pairs. An `ExponentialFamilyDistribution` is converted to its standard form for the
  expectation, as before. A `GammaShapeRate` is now converted to a `Gamma`, so the closed forms
  under a `Gamma` also hold under it; in 0.4 these calls failed.
- The gradients take `Base.Fix1(logpdf, p)` integrands, like the expectations.
- `ClosedFormAD(ad)`, a backend that differentiates a closed-form expectation with automatic
  differentiation. With Enzyme.jl loaded, `ClosedFormAD(AutoEnzyme())` replaces
  `ClosedWilliamsProduct(EnzymeBackend())`.
- `LogGammaKernel(α, β)`, the unnormalised log-density `βx − eˣ/α` of a `LogGamma` for any real
  `β`, with a closed-form expectation and gradient under a univariate Normal. A Gamma message
  pulled back through `exp`, such as the message of a Poisson count of zero, is one, while a
  `LogGamma` with `β ≤ 0` is not a distribution.
- `log ∘ Product(...)` can be evaluated at a point, so numerical backends can compute it where
  there is no closed form.
- ADTypes.jl is a new dependency.
- Routers of the protocol compute a gradient in closed form where the package has one, and with
  a numerical backend otherwise.
- BayesBase 1.7 is required.

### Fixed

- The gradients under the exponential family forms of a `NormalMeanVariance` and a `Gamma`, and
  under a `NormalMeanVariance`, keep the element type of `q`: a `Float32` `q` no longer gives a
  `Float64` gradient.
