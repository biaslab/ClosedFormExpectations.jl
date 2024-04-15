export Product

struct Product{T} <: Expression
    multipliers::T
end

function mean(::ClosedFormExpectation, q, p::ComposedFunction{typeof(log), Product{T}}) where {T}
    return sum(mean(ClosedFormExpectation(), q, log ∘ p_i) for p_i in p.inner.multipliers)
end