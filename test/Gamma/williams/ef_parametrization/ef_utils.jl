include("../gamma_utils.jl")

using ExponentialFamily

score(q::ExponentialFamilyDistribution{ExponentialFamily.Gamma}, x) = sufficientstatistics(q, x) .- gradlogpartition(q)
