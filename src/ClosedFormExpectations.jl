module ClosedFormExpectations

export mean

import Base: log
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

struct ClosedWilliamsProduct end 

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

# extra multivariate distributions
include("expressions/LinearLogGamma.jl")

# rules for computing expectations
include("Exponential/Exponential.jl")
# normal 
include("Normal/expectation.jl")
include("Normal/williams/normal.jl")
include("Normal/williams/normal_mean_variance.jl")
include("Normal/williams/ef_parametrization.jl")
# mv normal
include("MvNormal/expectation.jl")

# gamma
include("Gamma/Gamma.jl")
include("Gamma/williams_gamma_ef.jl")

# exponetial family distribution interface
include("exponential_family_interface.jl")

# support BayesBase ProductOf
include("productof.jl")

end
