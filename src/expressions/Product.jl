export Product

"""
    Product(multipliers)

Expression type representing a product of multiple functions. When composed with `log`, the product decomposes into a sum of logarithms.
"""
struct Product{T} <: Expression
    multipliers::T
end
