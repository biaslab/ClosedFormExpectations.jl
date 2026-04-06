export Abs

"""
    Abs

Expression type representing the absolute value function ``|x|``.
"""
struct Abs <: Expression end

function (f::Abs)(x)
    return abs(x)
end
