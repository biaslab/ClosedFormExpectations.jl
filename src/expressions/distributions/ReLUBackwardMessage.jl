export ReLUBackwardMessage

"""
    ReLUBackwardMessage{T}

Represents the exact backward message from a ReLU factor node to its input variable ``x``,
given an incoming Gaussian message on ``y`` with mean ``m_y`` and variance ``v_y``.

The message is proportional to ``\\mathcal{N}(\\operatorname{ReLU}(x);\\,m_y,\\,v_y)``:

```math
m_{f \\to x}(x) \\propto
\\begin{cases}
  \\mathcal{N}(0;\\,m_y,\\,v_y), & x \\leq 0,\\\\
  \\mathcal{N}(x;\\,m_y,\\,v_y), & x > 0.
\\end{cases}
```

This is an unnormalized density on ``\\mathbb{R}``: constant on the negative half-line and
Gaussian-shaped on the positive half-line.  The `logpdf` method returns
``\\log\\mathcal{N}(\\max(0,x);\\,m_y,\\,v_y)``.
"""
struct ReLUBackwardMessage{T<:Real}
    m_y::T
    v_y::T
end

ReLUBackwardMessage(m_y::Real, v_y::Real) = ReLUBackwardMessage(promote(m_y, v_y)...)

function BayesBase.logpdf(d::ReLUBackwardMessage, x::Real)
    relu_x = max(zero(x), x)
    return -1 / 2 * log(2π * d.v_y) - (relu_x - d.m_y)^2 / (2 * d.v_y)
end
