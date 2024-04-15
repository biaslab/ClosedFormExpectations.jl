struct Product <: LogExpression
    multipliers::Tuple{<:LogExpression}
end

function mean(::ClosedFormExpectation, q, p::ComposedFunction{typeof(log), Product})
    return sum(mean(ClosedFormExpectation(), q, log ∘ p_i) for p_i in p.multipliers)
end