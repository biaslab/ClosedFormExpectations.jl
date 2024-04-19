include("../test_utils.jl")

using ClosedFormExpectations
using StableRNGs
using SpecialFunctions
using Distributions

score(q::Gamma, x) = [log(x) -  log(scale(q)) - polygamma(0, shape(q)), x/scale(q)^2 - shape(q)/scale(q)]

