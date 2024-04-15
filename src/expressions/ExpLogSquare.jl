"""
    ExpLogSquare(μ, σ)

ExpLogSquare is a type that represents the exp(-(log(x) - μ)^2/(2σ^2)) function.
"""
struct ExpLogSquare{T} <: LogExpression
    μ::T
    σ::T
end

function (p::ExpLogSquare)(x)
    return exp(-(log(x) - p.μ)^2/(2*p.σ^2))
end

function Base.log(p::ExpLogSquare, x) 
    return -(log(x) - p.μ)^2/(2*p.σ^2)
end