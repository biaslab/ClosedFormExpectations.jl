struct Product <: LogExpression
    multipliers::Tuple{LogExpression}
end

function meanlog(::ClosedFormExpectation, q, p::Product)
    return sum(meanlog(ClosedFormExpectation(), q, p_i) for p_i in p.multipliers)
end