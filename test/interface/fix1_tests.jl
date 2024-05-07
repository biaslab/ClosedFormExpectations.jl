@testitem "Support ::Base.Fix1{typeof(logpdf), D}" begin
    include("../test_utils.jl")
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    rng = StableRNG(123)
    dist = Exponential(1.0)
    fixed_logpdf = Base.Fix1(logpdf, dist)
    @test mean(ClosedFormExpectation(), fixed_logpdf, Exponential(10.0)) ≈ mean(ClosedFormExpectation(), Logpdf(Exponential(1.0)), Exponential(10.0))
end