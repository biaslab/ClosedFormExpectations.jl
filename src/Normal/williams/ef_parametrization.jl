using StaticArrays
import Distributions: var, mean, std
import ExponentialFamily: ExponentialFamilyDistribution, NormalMeanVariance, getnaturalparameters

function mean_ef_impl(expectation::ClosedWilliamsProduct, f, q::ExponentialFamilyDistribution{T}) where {T <: NormalMeanVariance}
    η = getnaturalparameters(q)
    jacobian = @SMatrix [-inv(2*η[2]) η[1]/(2*η[2]^2); 0.0 (-1/η[2])^(3/2)/(2*sqrt(2))]
    normal = Normal(mean(q), std(q))
    return vec(jacobian' * mean(expectation, f, normal))
end
