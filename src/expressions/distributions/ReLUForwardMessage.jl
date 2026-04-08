export ReLUForwardMessage

import Distributions: ContinuousUnivariateDistribution

"""
    ReLUForwardMessage{T}

Represents the exact forward message from a ReLU factor node to its output variable ``y``,
given an incoming Gaussian message on ``x`` with mean ``m_x`` and variance ``v_x``.

The full message is a mixed distribution:

```math
m_{f \\to y}(y) = \\Phi\\!\\left(\\frac{-m_x}{\\sqrt{v_x}}\\right)\\delta(y)
               + \\mathbf{1}_{y>0}\\,\\mathcal{N}(y;\\,m_x,\\,v_x)
```

The `logpdf` method returns the log density of the **continuous component** (defined for ``y > 0``).
For ``y \\leq 0`` it returns ``-\\infty``.  The atom weight at ``y = 0`` is
``\\Phi(-m_x / \\sqrt{v_x})``.
"""
struct ReLUForwardMessage{T<:Real} <: ContinuousUnivariateDistribution
    m_x::T
    v_x::T
end

ReLUForwardMessage(m_x::Real, v_x::Real) = ReLUForwardMessage(promote(m_x, v_x)...)

function BayesBase.logpdf(d::ReLUForwardMessage, y::Real)  # extend BayesBase.logpdf
    if y > zero(y)
        return -1 / 2 * log(2π * d.v_x) - (y - d.m_x)^2 / (2 * d.v_x)
    end
    return convert(float(typeof(y)), -Inf)
end
