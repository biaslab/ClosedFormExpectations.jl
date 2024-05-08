include("../normal_utils.jl")

using ExponentialFamily

score(q::ExponentialFamilyDistribution{NormalMeanVariance}, x) = sufficientstatistics(q, x) .- gradlogpartition(q)