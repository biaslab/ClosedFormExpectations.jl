@testitem "mean(::ClosedFormExpectation, SMT, ::NormalMeanVariance)" begin
    include("../test_utils.jl")
    using ClosedFormExpectations
    using StableRNGs
    using ExponentialFamily
    using Distributions

    nmv = NormalMeanVariance(0.0, 1.0)
    normal = Normal(0.0, 1.0)
    laplace = Laplace(0.0, 1.0)
    @test mean(ClosedFormExpectation(), Logpdf(nmv), nmv) ≈ mean(ClosedFormExpectation(), Logpdf(normal), normal)
    @test mean(ClosedFormExpectation(), Logpdf(laplace), nmv) ≈ mean(ClosedFormExpectation(), Logpdf(laplace), normal)
end

@testitem "Support GaussianDistributionsFamily" begin
    using ExponentialFamily
    using ClosedFormExpectations
    using StableRNGs

    @testset "ClosedFormExpectation interface" begin
        nmv = NormalMeanVariance(0.0, 1.0)
        nmp = NormalMeanPrecision(0.0, 1.0)
    
        @test mean(ClosedFormExpectation(), Logpdf(nmv), nmv) isa Number
        @test mean(ClosedFormExpectation(), Logpdf(nmp), nmv) isa Number 
    end

    @testset "ClosedWilliamsProduct interface" begin
        using Distributions

        nmv = NormalMeanVariance(0.0, 1.0)
        nmp = NormalMeanPrecision(0.0, 1.0)
    
        @test mean(ClosedWilliamsProduct(), Logpdf(nmv), Normal(0, 1)) isa AbstractArray
        @test mean(ClosedWilliamsProduct(), Logpdf(nmp), Normal(0, 1)) isa AbstractArray
        @test mean(ClosedWilliamsProduct(), Logpdf(nmv), Normal(0, 1)) ≈ mean(ClosedWilliamsProduct(), Logpdf(nmp), Normal(0, 1))
        @test mean(ClosedWilliamsProduct(), Logpdf(nmv), nmv) ≈ mean(ClosedWilliamsProduct(), Logpdf(nmp), nmv)
    end
end