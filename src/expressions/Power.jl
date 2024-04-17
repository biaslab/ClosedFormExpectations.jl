export Power

"""
Power

Power is a type that represents the x^N function.
"""
struct Power{T} <: Expression
    n::T
end

function (f::Power{Val{N}})(x) where {N}
    return x^N
end

function Base.log(::Power{Val{N}}, x) where {N}
    return N * log(x)
end