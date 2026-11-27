using StaticArrays
import Distributions: Distribution, Gamma
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters

# The gradient under the exponential family form of a `Gamma`, with respect to its natural
# parameters η₁ = α - 1 and η₂ = -1/θ, from the gradient under a `Gamma` with respect to its shape
# α and scale θ. The Jacobian ∂(α, θ)/∂(η₁, η₂) is diagonal, with entries 1 and 1/η₂².
function gamma_ef_gradient(backend, f, q::ExponentialFamilyDistribution{Gamma})
    η = getnaturalparameters(q)
    jacobian_diagonal = @SVector [one(η[2]), inv(η[2]^2)]
    return expectation_gradient(backend, f, convert(Distribution, q)) .* jacobian_diagonal
end
