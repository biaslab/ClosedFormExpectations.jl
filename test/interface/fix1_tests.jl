@testitem "Support ::Base.Fix1{typeof(logpdf), D}" begin
    import BayesBase
    using Distributions
    include("../test_utils.jl")
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    rng = StableRNG(123)
    dist = Exponential(1.0)
    fixed_logpdf = Base.Fix1(logpdf, dist)
    @test BayesBase.expectation(ClosedFormExpectation(), fixed_logpdf, Exponential(10.0)) ≈ BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Exponential(1.0)), Exponential(10.0))
end