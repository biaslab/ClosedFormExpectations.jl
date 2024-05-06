@testitem "Support Exponetial Family Types" begin
    include("../test_utils.jl")
    @testset "GaussianDistributionsFamily support" begin
        using ExponentialFamily
        using ClosedFormExpectations
        using StableRNGs
        
        nmv = NormalMeanVariance(0.0, 1.0)
        nmp = NormalMeanPrecision(0.0, 1.0)

        @test mean(ClosedFormExpectation(), Logpdf(nmv), nmv) isa Number
        @test mean(ClosedFormExpectation(), Logpdf(nmp), nmv) isa Number
    end
    
    @testset "Values for Normal and GaussianDistributionsFamily are equal" begin
        using ExponentialFamily
        using Distributions
        using ClosedFormExpectations
        using StableRNGs
        
        nmv = NormalMeanVariance(0.0, 1.0)
        normal = Normal(0.0, 1.0)

        @test mean(ClosedFormExpectation(), Logpdf(nmv), nmv) ≈ mean(ClosedFormExpectation(), Logpdf(normal), normal)
    end
end