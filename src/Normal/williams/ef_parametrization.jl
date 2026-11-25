using StaticArrays
import Distributions: var, mean, std
import ExponentialFamily: ExponentialFamilyDistribution, NormalMeanVariance, getnaturalparameters

# The gradient under the exponential family form of a `NormalMeanVariance`, with respect to its
# natural parameters, from the gradient under a `Normal` with respect to its mean and standard
# deviation
function normalmeanvariance_ef_gradient(backend, f, q::ExponentialFamilyDistribution{NormalMeanVariance})
    η = getnaturalparameters(q)
    η₁, η₂ = η[1], η[2]
    v = -inv(η₂) # twice the variance
    jacobian = @SMatrix [-inv(2 * η₂) η₁ / (2 * η₂^2); zero(η₂) v * sqrt(v) / (2 * sqrt(2 * one(η₂)))]
    normal = Normal(mean(q), std(q))
    return vec(jacobian' * expectation_gradient(backend, f, normal))
end
