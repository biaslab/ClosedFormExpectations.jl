# [The Expectation Protocol of BayesBase](@id lib-protocol)

BayesBase defines a protocol for computing expectations, `BayesBase.expectation(backend, f, q)`,
and their gradients with respect to the parameters of `q`,
`BayesBase.expectation_gradient(backend, f, q)`. The `backend` argument selects how they are
computed, for example in closed form, by quadrature or by sampling. Backends are defined in other
packages. Inference packages such as RxInfer compute expectations through this protocol and choose
among the backends that are loaded.

`ClosedFormExpectation()` is the backend of this package:

```julia
using ClosedFormExpectations, BayesBase, Distributions

f = Base.Fix1(logpdf, LogNormal(0.3, 0.7))   # x -> logpdf(LogNormal(0.3, 0.7), x)
q = Gamma(2.0, 1.5)

BayesBase.supports_expectation(ClosedFormExpectation(), f, q)  # true
BayesBase.is_expectation_exact(ClosedFormExpectation(), f, q)  # true
BayesBase.expectation(ClosedFormExpectation(), f, q)           # E_q[log LogNormal(x | 0.3, 0.7)]
BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)  # with respect to the shape and the scale
```

For `E_q[log p]`, the integrand is `Base.Fix1(logpdf, p)`, which is `x -> logpdf(p, x)`. The
distribution `q` can also be a `GammaShapeRate`, or an `ExponentialFamilyDistribution` whose
standard form has a closed form.

The backend supports the pairs listed in [Supported Pairs](@ref lib-supported-pairs), and
`BayesBase.supports_expectation` and `BayesBase.supports_expectation_gradient` say whether a pair
is one of them. Each closed form is defined with BayesBase's `@supports_expectation` or
`@supports_expectation_gradient`, which declare its support together with the method. The backend
labels every result `BayesBase.ExpectationExact()`, and for any other pair it throws a
`BayesBase.UnsupportedExpectationError`.

The backend is registered when the package is loaded. A `BayesBase.ExpectationDynamicRouter()`
then chooses it for the pairs it supports ahead of the numerical backends, because its results
are exact. A router such as `BayesBase.ExpectationFirstOf(ClosedFormExpectation(), quadrature)`
computes a gradient in closed form where there is one, and by the quadrature of `f` times the
score of `q` otherwise.

Under a `Categorical`, every integrand has a closed form, an exact sum. Still, the backend does not
claim support for an arbitrary integrand (`F = Any`) under it, because an expectation under a
product of distributions needs the points of each component, and a numerical backend provides
them.
