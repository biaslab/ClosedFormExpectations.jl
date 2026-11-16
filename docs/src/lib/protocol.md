# [The Expectation Protocol of BayesBase](@id lib-protocol)

BayesBase defines one way to ask for an expectation, `BayesBase.expectation(backend, f, q)`,
answered by backends from other packages: closed forms, quadrature, sampling. Inference packages
such as RxInfer ask through it, and choose among the backends that are loaded.

`ClosedFormExpectation()` is such a backend. For the pairs with a closed form,
`BayesBase.expectation(ClosedFormExpectation(), f, q)` is `mean(ClosedFormExpectation(), f, q)`:

```julia
using ClosedFormExpectations, BayesBase, Distributions

f = Base.Fix1(logpdf, LogNormal(0.3, 0.7))   # x -> logpdf(LogNormal(0.3, 0.7), x)
q = Gamma(2.0, 1.5)

BayesBase.supports_expectation(ClosedFormExpectation(), f, q)  # true
BayesBase.is_expectation_exact(ClosedFormExpectation(), f, q)  # true
BayesBase.expectation(ClosedFormExpectation(), f, q)           # E_q[log LogNormal(x | 0.3, 0.7)]
```

- **The integrand** of `E_q[log p]` is `Base.Fix1(logpdf, p)`, Julia's own callable for
  `x -> logpdf(p, x)`. `Logpdf(p)` and the distribution `p` itself are accepted as well.
- **The distribution** can also be a `GammaShapeRate`, or an `ExponentialFamilyDistribution` whose
  standard form has a closed form.
- **Support.** The backend supports the pairs of [`ClosedFormExpectations.has_closed_form`](@ref), the
  table of [Supported Pairs](@ref lib-supported-pairs), and labels every result
  `BayesBase.ExpectationExact()`. Any other pair throws a `BayesBase.UnsupportedExpectationError`.
- **Routing.** The backend is registered when the package is loaded, so a
  `BayesBase.ExpectationDynamicRouter()` chooses it for the pairs it supports, before any numerical
  backend, since its results are exact.
- **Every integrand under a `Categorical`** has a closed form, an exact sum, but the backend does not
  claim every integrand (`F = Any`) for it. An expectation under a product of distributions needs the
  points of each component, which a numerical backend provides.

```@docs
ClosedFormExpectations.has_closed_form
```

The Williams product is not part of the protocol, which has no gradients yet. It is asked for with
`mean(ClosedWilliamsProduct(), f, q)`, which accepts `Base.Fix1(logpdf, p)` too, and
`has_closed_form(ClosedWilliamsProduct(), f, q)` says where it has a closed form.
