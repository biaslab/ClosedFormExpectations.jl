using StaticArrays
import Distributions: Gamma, shape, scale
import ExponentialFamily
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters

# Gamma natural parameters in ExponentialFamily.jl are usually:
# η₁ = α - 1
# η₂ = -1/θ (where θ is scale)
# sufficient statistics: [log(x), x]
#
# Mapping back:
# α = η₁ + 1
# θ = -1/η₂
#
# Jacobian J = ∂(α, θ) / ∂(η₁, η₂)
# ∂α/∂η₁ = 1, ∂α/∂η₂ = 0
# ∂θ/∂η₁ = 0, ∂θ/∂η₂ = 1/η₂²
#
# J = [1      0
#      0  1/η₂²]

function mean_ef_impl(expectation::ClosedWilliamsProduct, f, q::ExponentialFamilyDistribution{T}) where {T <: ExponentialFamily.Gamma}
    η = getnaturalparameters(q)
    # η[1] is shape-related, η[2] is scale-related
    
    # Construct Jacobian diagonal
    d_alpha_d_eta1 = 1.0
    d_scale_d_eta2 = 1.0 / (η[2]^2)
    
    jacobian_diag = @SVector [d_alpha_d_eta1, d_scale_d_eta2]
    
    # Convert to distribution to use the standard `mean` method
    dist = convert(Distribution, q)
    
    # Get gradient w.r.t standard parameters [α, θ]
    grad_standard = mean(expectation, f, dist)
    
    # Apply Jacobian (chain rule)
    # ∇_η E = J^T * ∇_θ E
    # Since J is diagonal:
    return grad_standard .* jacobian_diag
end
