@testitem "Support ExponetialFamilyDistribution for ClosedFormExpectation" begin
    include("../test_utils.jl")
    using ExponentialFamily
    @testset "mean(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{Exponential})" begin
        dist = Exponential(1.0)
        ef = convert(ExponentialFamilyDistribution, Exponential(1.0))
        @test mean(ClosedFormExpectation(), Logpdf(Exponential(1.0)), ef) ≈ mean(ClosedFormExpectation(), Logpdf(Exponential(1.0)), dist)
    end
    @testset "mean(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{NormalMeanVariance})" begin
        dist = Normal(1.0, 1.0)
        ef = convert(ExponentialFamilyDistribution, Normal(1.0, 1.0))
        @test mean(ClosedFormExpectation(), Abs(), ef) ≈ mean(ClosedFormExpectation(), Abs(), dist)
    end
    @testset "mean(::ClosedFormExpectation, f, q::ExponentialFamilyDistribution{Gamma})" begin
        import LogExpFunctions: xlogx
        dist = Gamma(1.0, 1.0)
        ef = convert(ExponentialFamilyDistribution, Gamma(1.0, 1.0))
        @test mean(ClosedFormExpectation(), xlogx, ef) ≈ mean(ClosedFormExpectation(), xlogx, dist)
    end
end