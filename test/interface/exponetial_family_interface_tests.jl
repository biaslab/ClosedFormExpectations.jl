@testitem "Support ExponetialFamilyDistribution for ClosedFormExpectation" begin
    import BayesBase
    using Distributions
    include("../test_utils.jl")
    using ExponentialFamily
    @testset "expectation(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{Exponential})" begin
        dist = Exponential(1.0)
        ef = convert(ExponentialFamilyDistribution, Exponential(1.0))
        @test BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Exponential(1.0)), ef) ≈ BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Exponential(1.0)), dist)
    end
    @testset "expectation(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{NormalMeanVariance})" begin
        dist = Normal(1.0, 1.0)
        ef = convert(ExponentialFamilyDistribution, Normal(1.0, 1.0))
        @test BayesBase.expectation(ClosedFormExpectation(), Abs(), ef) ≈ BayesBase.expectation(ClosedFormExpectation(), Abs(), dist)
    end
    @testset "expectation(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{Gamma})" begin
        import LogExpFunctions: xlogx
        dist = Gamma(1.0, 1.0)
        ef = convert(ExponentialFamilyDistribution, Gamma(1.0, 1.0))
        @test BayesBase.expectation(ClosedFormExpectation(), xlogx, ef) ≈ BayesBase.expectation(ClosedFormExpectation(), xlogx, dist)
    end
end