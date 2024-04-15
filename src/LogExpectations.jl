module LogExpectations

import Base: log
import Distributions: mean

export ClosedFormExpectation, meanlog
export ExpLogSquare
export ClosedWilliamsProduct

"""
    ClosedFormExpectation
"""
struct ClosedFormExpectation end

"""
    meanlog(::ClosedFormExpectation, q, f)

Compute the E_q[log f(x)] where q is a distribution and f is a function.
"""
function meanlog(::ClosedFormExpectation, ::Nothing, ::Nothing) end

struct ClosedWilliamsProduct end 

"""
    meanlog(::ClosedWilliamsProduct, q, f)

Suppose q is a distribution with density parameterized by θ and f is a function.

Compute the E_q[log f(x) ∇_θ log q(x; θ)] where q is a distribution and f is a function.
"""
function meanlog(::ClosedWilliamsProduct, ::Nothing, ::Nothing) end

abstract type LogExpression end

function (f::ComposedFunction{typeof(log), T})(x) where {T <: LogExpression}
    return log(f.inner, x)
end

# expressions
include("expressions/ExpLogSquare.jl")

# rules for computing expectation of log
include("Exponential/Exponential.jl")

end
