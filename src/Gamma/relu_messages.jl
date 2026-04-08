import Distributions: Gamma, mean, var

# For Gamma q the support is (0,∞), so ReLU(x) = x on the full support and both
# messages reduce to ∫ log 𝒩(x; m, v) Gamma(x) dx, which equals
#
#   -½ log(2π v) − (Var_q + (E_q[x] − m)²) / (2v)

function mean(::ClosedFormExpectation, f::Logpdf{<:ReLUForwardMessage}, q::Gamma)
    m_x, v_x = f.dist.m_x, f.dist.v_x
    μ_q, v_q = mean(q), var(q)
    return -1 / 2 * log(2π * v_x) - (v_q + (μ_q - m_x)^2) / (2 * v_x)
end

function mean(::ClosedFormExpectation, f::Logpdf{<:ReLUBackwardMessage}, q::Gamma)
    m_y, v_y = f.dist.m_y, f.dist.v_y
    μ_q, v_q = mean(q), var(q)
    return -1 / 2 * log(2π * v_y) - (v_q + (μ_q - m_y)^2) / (2 * v_y)
end
