module ClosedFormExpectations

import Base: log
import BayesBase
import BayesBase: @supports_expectation, @supports_expectation_gradient, expectation, expectation_gradient
import Distributions: mean, logpdf

export ClosedFormExpectation
export ExpLogSquare

"""
    ClosedFormExpectation()
    ClosedFormExpectation(gradient)

A backend of BayesBase's expectation protocol that computes expectations and their gradients in
closed form. `BayesBase.expectation(ClosedFormExpectation(), f, q)` computes ``\\mathbb{E}_q[f]``,
and `BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)` its gradient with respect to the
parameters of `q`, for the pairs listed in [Supported Pairs](@ref lib-supported-pairs).
`BayesBase.supports_expectation(ClosedFormExpectation(), f, q)` and
`BayesBase.supports_expectation_gradient(ClosedFormExpectation(), f, q)` say whether a pair is
supported. The backend labels its results exact and is registered when the package is loaded. See
[The Expectation Protocol of BayesBase](@ref lib-protocol).

`gradient` selects how gradients are computed: `nothing`, the default, uses the formulas written
in the package, and an [`EnzymeBackend`](@ref) differentiates the closed-form expectation with
Enzyme.jl, see [Enzyme Extension](@ref ext-enzyme).
"""
struct ClosedFormExpectation{G}
    gradient::G
end

ClosedFormExpectation() = ClosedFormExpectation(nothing)

"""
    EnzymeForward

Selects Enzyme.jl forward-mode AD for `EnzymeBackend`.
"""
struct EnzymeForward end

"""
    EnzymeReverse

Selects Enzyme.jl reverse-mode AD for `EnzymeBackend`. This is the default mode.
"""
struct EnzymeReverse end

export EnzymeForward, EnzymeReverse

"""
    EnzymeBackend{Mode}

Computes the gradient of an expectation for `ClosedFormExpectation(EnzymeBackend())` by
differentiating the closed-form expectation with Enzyme.jl, using the identity

```math
\\mathbb{E}_q[f(x)\\,\\nabla_\\eta \\log q(x;\\eta)] = \\nabla_\\eta \\mathbb{E}_q[f(x)]
```

It requires Enzyme.jl to be loaded. It works for an `ExponentialFamilyDistribution{T}`, in its
natural parameters `η`, wherever the expectation has a closed form under `T`.

The `Mode` type parameter selects the AD mode:
- `EnzymeBackend()` or `EnzymeBackend(EnzymeReverse())`: reverse mode, the default.
- `EnzymeBackend(EnzymeForward())`: forward mode.
"""
struct EnzymeBackend{Mode}
    mode::Mode
end
EnzymeBackend() = EnzymeBackend(EnzymeReverse())

export EnzymeBackend

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
