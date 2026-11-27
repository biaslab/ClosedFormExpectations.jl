module ClosedFormExpectations

import Base: log
import ADTypes
import BayesBase
import BayesBase: @supports_expectation, @supports_expectation_gradient, expectation, expectation_gradient
import Distributions: mean, logpdf

export ClosedFormExpectation
export ExpLogSquare

"""
    ClosedFormExpectation()

A backend of BayesBase's expectation protocol that computes expectations and their gradients in
closed form. `BayesBase.expectation(ClosedFormExpectation(), f, q)` computes ``\\mathbb{E}_q[f]``,
and `BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)` its gradient with respect to the
parameters of `q`, for the pairs listed in [Supported Pairs](@ref lib-supported-pairs).
`BayesBase.supports_expectation(ClosedFormExpectation(), f, q)` and
`BayesBase.supports_expectation_gradient(ClosedFormExpectation(), f, q)` say whether a pair is
supported. The backend labels its results exact and is registered when the package is loaded. See
[The Expectation Protocol of BayesBase](@ref lib-protocol).

The gradients are the formulas written in the package. For a gradient that has no formula, but
whose expectation has a closed form, see [`ClosedFormAD`](@ref).
"""
struct ClosedFormExpectation end

"""
    ClosedFormAD(ad)

A backend of BayesBase's expectation protocol that computes the gradient of an expectation by
differentiating its closed form with automatic differentiation, using the identity

```math
\\mathbb{E}_q[f(x)\\,\\nabla_\\eta \\log q(x;\\eta)] = \\nabla_\\eta \\mathbb{E}_q[f(x)]
```

`ad` is an automatic differentiation backend from ADTypes.jl. `AutoEnzyme()` is supported once
Enzyme.jl is loaded, see [Enzyme Extension](@ref ext-enzyme). The gradient is with respect to the
natural parameters `η` of a univariate `ExponentialFamilyDistribution{T}`, wherever
`ClosedFormExpectation()` has the expectation under `T`. The backend computes nothing else, so it
is meant to follow `ClosedFormExpectation()` in a router, which then uses a formula where the
package has one:

```julia
BayesBase.ExpectationFirstOf(ClosedFormExpectation(), ClosedFormAD(AutoEnzyme()), quadrature)
```

Its results are labelled exact. It is not registered with BayesBase's dynamic router.
"""
struct ClosedFormAD{A <: ADTypes.AbstractADType}
    ad::A
end

export ClosedFormAD

# The integrand of the expected log-density `E_q[log p]`
const LogpdfOf{D} = Base.Fix1{typeof(logpdf), D}

abstract type Expression end

function (f::ComposedFunction{typeof(log), T})(x) where {T <: Expression}
    return log(f.inner, x)
end

# expressions
include("expressions/ExpLogSquare.jl")
include("expressions/Product.jl")
include("expressions/Square.jl")
include("expressions/Power.jl")
include("expressions/xlog2x.jl")
include("expressions/Abs.jl")

# extra distributions
include("expressions/distributions/LogGamma.jl")
include("expressions/LogGammaKernel.jl")
include("expressions/distributions/ReLUForwardMessage.jl")
include("expressions/distributions/ReLUBackwardMessage.jl")

# extra multivariate distributions
include("expressions/LinearLogGamma.jl")

# rules for computing expectations
include("Categorical/expectation.jl")
include("Categorical/williams_ef.jl")
include("Exponential/Exponential.jl")
# normal
include("Normal/expectation.jl")
include("Normal/relu_messages.jl")
include("Normal/williams/normal.jl")
include("Normal/williams/normal_mean_variance.jl")
include("Normal/williams/ef_parametrization.jl")
# mv normal
include("MvNormal/expectation.jl")

# gamma
include("Gamma/Gamma.jl")
include("Gamma/relu_messages.jl")

# lognormal
include("LogNormal/expectation.jl")
include("LogNormal/relu_messages.jl")
include("Gamma/williams_gamma_ef.jl")

# the expectation protocol of BayesBase
include("protocol.jl")

end
