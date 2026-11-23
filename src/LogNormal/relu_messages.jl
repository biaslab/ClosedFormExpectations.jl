import Distributions: LogNormal

# For LogNormal q the support is (0,∞), so both messages reduce to
#
#   ∫ log 𝒩(x; m, v) LogNormal(x) dx
#   = -½ log(2π v) − (E[x²] − 2m E[x] + m²) / (2v)
#
# where for LogNormal(μ, σ):  E[x] = exp(μ + σ²/2),  E[x²] = exp(2μ + 2σ²).

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{<:ReLUForwardMessage}, q::LogNormal)
    m_x, v_x = f.x.m_x, f.x.v_x
    μ_q, σ_q = q.μ, q.σ
    E_x  = exp(μ_q + σ_q^2 / 2)
    E_x2 = exp(2 * μ_q + 2 * σ_q^2)
    return -1 / 2 * log(2π * v_x) - (E_x2 - 2 * m_x * E_x + m_x^2) / (2 * v_x)
end

@supports_expectation function expectation(::ClosedFormExpectation, f::LogpdfOf{<:ReLUBackwardMessage}, q::LogNormal)
    m_y, v_y = f.x.m_y, f.x.v_y
    μ_q, σ_q = q.μ, q.σ
    E_x  = exp(μ_q + σ_q^2 / 2)
    E_x2 = exp(2 * μ_q + 2 * σ_q^2)
    return -1 / 2 * log(2π * v_y) - (E_x2 - 2 * m_y * E_x + m_y^2) / (2 * v_y)
end
