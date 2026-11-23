import Distributions: Gamma, mean, var

# For Gamma q the support is (0,∞), so ReLU(x) = x on the full support and both
# messages reduce to ∫ log 𝒩(x; m, v) Gamma(x) dx, which equals
#
#   -½ log(2π v) − (Var_q + (E_q[x] − m)²) / (2v)

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{<:ReLUForwardMessage}, q::Gamma)
    m_x, v_x = f.x.m_x, f.x.v_x
    μ_q, v_q = mean(q), var(q)
    return -1 / 2 * log(2π * v_x) - (v_q + (μ_q - m_x)^2) / (2 * v_x)
end

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{<:ReLUBackwardMessage}, q::Gamma)
    m_y, v_y = f.x.m_y, f.x.v_y
    μ_q, v_q = mean(q), var(q)
    return -1 / 2 * log(2π * v_y) - (v_q + (μ_q - m_y)^2) / (2 * v_y)
end
