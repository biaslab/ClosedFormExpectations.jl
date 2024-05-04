include("../test_utils.jl")

using StableRNGs
using ClosedFormExpectations
using Distributions

score(q::Normal, x) = [-(x - q.μ)/q.σ^2, -1/q.σ + (x - q.μ)^2/q.σ^3]

