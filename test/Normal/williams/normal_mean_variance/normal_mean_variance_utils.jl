include("../normal_utils.jl")

using ExponentialFamily
import Distributions: var

score(q::NormalMeanVariance, x) = [(x - mean(q))/var(q), -0.5/var(q) + 0.5*(x - mean(q))^2/var(q)^2]