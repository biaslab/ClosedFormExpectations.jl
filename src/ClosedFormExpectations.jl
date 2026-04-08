module ClosedFormExpectations

export mean

import Base: log
import BayesBase
import Distributions: mean

export ClosedFormExpectation
export ExpLogSquare
export ClosedWilliamsProduct

"""
    ClosedFormExpectation
"""
struct ClosedFormExpectation end

"""
    mean(::ClosedFormExpectation, f, q)

Compute the E_q[f(x)] where q is a distribution and f is a function.
"""
function mean(::ClosedFormExpectation, ::Nothing, ::Nothing) end

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

Backend for `ClosedWilliamsProduct` that uses Enzyme.jl automatic differentiation
to compute ``\\nabla_\\eta \\mathbb{E}_q[f(x)]``, exploiting the identity

```math
\\mathbb{E}_q[f(x)\\,\\nabla_\\eta \\log q(x;\\eta)] = \\nabla_\\eta \\mathbb{E}_q[f(x)]
```

Requires Enzyme.jl to be loaded. Works for any `ExponentialFamilyDistribution{T}` where
a `ClosedFormExpectation` is already defined.

The `Mode` type parameter selects the AD mode:
- `EnzymeBackend()` or `EnzymeBackend(EnzymeReverse())` — reverse-mode (default, efficient for many outputs).
- `EnzymeBackend(EnzymeForward())` — forward-mode (efficient for few inputs).
"""
struct EnzymeBackend{Mode}
    mode::Mode
end
EnzymeBackend() = EnzymeBackend(EnzymeReverse())

export EnzymeBackend

"""
    ClosedWilliamsProduct{B}

A strategy for computing the Williams' product (score function estimator gradient):
``\\mathbb{E}_q[f(x) \\nabla_\\theta \\log q(x; \\theta)]``.

The optional `backend` field selects the differentiation backend:
- `ClosedWilliamsProduct()` — default, uses hand-coded implementations.
- `ClosedWilliamsProduct(EnzymeBackend())` — Enzyme reverse-mode AD (requires `using Enzyme`).
- `ClosedWilliamsProduct(EnzymeBackend(EnzymeForward()))` — Enzyme forward-mode AD.
"""
struct ClosedWilliamsProduct{B}
    backend::B
end
ClosedWilliamsProduct() = ClosedWilliamsProduct(nothing)

"""
    mean(::ClosedWilliamsProduct, f, q)

Suppose q is a distribution with density parameterized by θ and f is a function.

Compute the E_q[f(x) ∇_θ log q(x; θ)] where q is a distribution and f is a function.
"""
function mean(::ClosedWilliamsProduct, ::Nothing, ::Nothing) end

abstract type Expression end

function (f::ComposedFunction{typeof(log), T})(x) where {T <: Expression}
    return log(f.inner, x)
end

# Logpdf structure
include("logpdf.jl")

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

# exponetial family distribution interface
include("exponential_family_interface.jl")

# support BayesBase ProductOf
include("productof.jl")

end
