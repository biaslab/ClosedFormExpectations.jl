using StaticArrays
import Distributions: var, mean, std
import ExponentialFamily: ExponentialFamilyDistribution, NormalMeanVariance, getnaturalparameters

# The gradient under the exponential family form of a `NormalMeanVariance`, with respect to its
# natural parameters, from the gradient under a `Normal` with respect to its mean and standard
# deviation
function normalmeanvariance_ef_gradient(backend, f, q::ExponentialFamilyDistribution{NormalMeanVariance})
    η = getnaturalparameters(q)
    jacobian = @SMatrix [-inv(2*η[2]) η[1]/(2*η[2]^2); 0.0 (-1/η[2])^(3/2)/(2*sqrt(2))]
    normal = Normal(mean(q), std(q))
    return vec(jacobian' * expectation_gradient(backend, f, normal))
end
