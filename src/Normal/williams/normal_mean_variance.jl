import Distributions: Normal, std
import ExponentialFamily: NormalMeanVariance
using StaticArrays

# The gradient under a `NormalMeanVariance`, with respect to its mean and variance, from the
# gradient under a `Normal` with respect to its mean and standard deviation
function normalmeanvariance_gradient(backend, f, q::NormalMeanVariance)
    normal = Normal(mean(q), std(q))
    σ = std(q)
    jacobian_diagonal = @SVector [one(σ), inv(2 * σ)]
    return jacobian_diagonal .* expectation_gradient(backend, f, normal)
end
