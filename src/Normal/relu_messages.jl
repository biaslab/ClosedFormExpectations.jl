import Distributions: Normal, cdf, pdf, std, var, mean
import ExponentialFamily: GaussianDistributionsFamily

# Note: Logpdf(ReLUForwardMessage) is NOT implemented for GaussianDistributionsFamily.
# ReLU output y ≥ 0 always, so logpdf(ReLUForwardMessage, y) = −∞ for y ≤ 0.
# A Normal q assigns positive mass to y ≤ 0, making E_Normal[logpdf(ReLUFM, y)] = −∞.
# Use Gamma or LogNormal (support on (0,∞)) for the forward-message expectation.

# ---------------------------------------------------------------------------
# ReLUBackwardMessage — exact expectation over all of ℝ
#
#   ∫ log 𝒩(max(0,x); m_y, v_y) 𝒩(x; μ, σ²) dx
#
# Split into x ≤ 0 (constant = log 𝒩(0; m_y, v_y)) and x > 0 (Gaussian):
#
#   = -½ log(2π v_y)
#     − m_y² (1-Φ) / (2 v_y)
#     − [(μ-m_y)²+σ²] Φ / (2 v_y)
#     − (μ-2m_y) σ φ / (2 v_y)
# ---------------------------------------------------------------------------
function mean(::ClosedFormExpectation, f::Logpdf{<:ReLUBackwardMessage}, q::GaussianDistributionsFamily)
    m_y, v_y = f.dist.m_y, f.dist.v_y
    μ, σ = mean(q), std(q)
    α  = μ / σ
    Φα = cdf(Normal(), α)
    φα = pdf(Normal(), α)
    return -1 / 2 * log(2π * v_y) -
           (m_y^2 * (1 - Φα) + ((μ - m_y)^2 + σ^2) * Φα + (μ - 2 * m_y) * σ * φα) / (2 * v_y)
end
