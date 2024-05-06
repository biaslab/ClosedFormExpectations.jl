export Abs

"""
    Abs{T}

The absolute value of a number: |x|.
"""

struct Abs <: Expression end

function (f::Abs)(x)
    return abs(x)
end
