export Product

"""
    Product(multipliers)

Expression type representing a product of multiple functions. When composed with `log`, the product decomposes into a sum of logarithms.
"""
struct Product{T} <: Expression
    multipliers::T
end

function mean(::ClosedFormExpectation, p::ComposedFunction{typeof(log), Product{T}}, q) where {T}
    return sum(mean(ClosedFormExpectation(), log ∘ p_i, q) for p_i in p.inner.multipliers)
end