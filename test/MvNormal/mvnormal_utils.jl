include("../test_utils.jl")

using Distributions
using ExponentialFamily

function prepare_samples(::MultivariateNormalDistributionsFamily{T}, samples) where {T}
    return eachcol(samples)
end

function generate_random_posdef_matrix(rng, d::Int)
    A = randn(rng, d, d) 
    A = A'*A
    return (A + A')/2
end