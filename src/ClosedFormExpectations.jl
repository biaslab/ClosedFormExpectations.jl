module ClosedFormExpectations

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
    mean(::ClosedFormExpectation, q, f)

Compute the E_q[f(x)] where q is a distribution and f is a function.
"""
function mean(::ClosedFormExpectation, ::Nothing, ::Nothing) end

struct ClosedWilliamsProduct end 

"""
    mean(::ClosedWilliamsProduct, q, f)

Suppose q is a distribution with density parameterized by θ and f is a function.

Compute the E_q[f(x) ∇_θ log q(x; θ)] where q is a distribution and f is a function.
"""
function mean(::ClosedWilliamsProduct, ::Nothing, ::Nothing) end

abstract type Expression end

function (f::ComposedFunction{typeof(log), T})(x) where {T <: Expression}
    return log(f.inner, x)
end

# expressions
include("expressions/ExpLogSquare.jl")
include("expressions/Product.jl")

# rules for computing expectation of log
include("Exponential/Exponential.jl")
include("Normal/UnivariateNormal.jl")

end
