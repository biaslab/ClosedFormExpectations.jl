import Distributions: Normal, std
import ExponentialFamily: GaussianDistributionsFamily, NormalMeanVariance
using StaticArrays
using SpecialFunctions: erfc, erf

function mean(::ClosedWilliamsProduct, p, q::NormalMeanVariance{T}) where {T}
    normal = Normal(mean(q), std(q))
    jacobian_diagonal = @SVector [1.0, 0.5/std(q)]
    return jacobian_diagonal .* mean(ClosedWilliamsProduct(), p, normal)
end