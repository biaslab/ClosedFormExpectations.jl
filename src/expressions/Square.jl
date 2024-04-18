export Square

"""
    Square

Square is a type that represents the x^2 function.
"""
struct Square <: Expression end

function (::Square)(x)
    return x^2
end

function Base.log(::Square, x) 
    return 2 * log(x)
end