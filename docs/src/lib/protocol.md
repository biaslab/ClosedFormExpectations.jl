# [The Expectation Protocol of BayesBase](@id lib-protocol)

BayesBase defines a protocol for computing expectations, `BayesBase.expectation(backend, f, q)`.
The `backend` argument selects how the expectation is computed, for example in closed form, by
quadrature or by sampling. Backends are defined in other packages. Inference packages such as
RxInfer compute expectations through this protocol and choose among the backends that are loaded.

`ClosedFormExpectation()` is one of these backends. For the pairs with a closed form,
`BayesBase.expectation(ClosedFormExpectation(), f, q)` returns `mean(ClosedFormExpectation(), f, q)`:

```julia
using ClosedFormExpectations, BayesBase, Distributions

f = Base.Fix1(logpdf, LogNormal(0.3, 0.7))   # x -> logpdf(LogNormal(0.3, 0.7), x)
q = Gamma(2.0, 1.5)

BayesBase.supports_expectation(ClosedFormExpectation(), f, q)  # true
BayesBase.is_expectation_exact(ClosedFormExpectation(), f, q)  # true
BayesBase.expectation(ClosedFormExpectation(), f, q)           # E_q[log LogNormal(x | 0.3, 0.7)]
```

For `E_q[log p]`, the integrand is written as `Base.Fix1(logpdf, p)`, which is
`x -> logpdf(p, x)`. `Logpdf(p)` and the distribution `p` itself are accepted as well. The
distribution `q` can also be a `GammaShapeRate`, or an `ExponentialFamilyDistribution` whose
standard form has a closed form.

The backend supports the pairs for which [`ClosedFormExpectations.has_closed_form`](@ref) is `true`,
listed in [Supported Pairs](@ref lib-supported-pairs), and labels every result
`BayesBase.ExpectationExact()`. For any other pair it throws a `BayesBase.UnsupportedExpectationError`.

The backend is registered when the package is loaded. A `BayesBase.ExpectationDynamicRouter()` then
chooses it for the pairs it supports, ahead of the numerical backends, because its results are exact.

Under a `Categorical`, every integrand has a closed form, an exact sum. Still, the backend does not
claim support for an arbitrary integrand (`F = Any`) under it, because an expectation under a
product of distributions needs the points of each component, and a numerical backend provides
them.

```@docs
ClosedFormExpectations.has_closed_form
```

The protocol does not cover gradients yet, so the Williams product is not part of it. Compute it
with `mean(ClosedWilliamsProduct(), f, q)`, which also accepts `Base.Fix1(logpdf, p)`, and check
where it has a closed form with `has_closed_form(ClosedWilliamsProduct(), f, q)`.
