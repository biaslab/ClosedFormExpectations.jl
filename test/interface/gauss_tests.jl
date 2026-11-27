@testitem "expectation(::ClosedFormExpectation, SMT, ::NormalMeanVariance)" begin
    import BayesBase
    using Distributions
    include("../test_utils.jl")
    using ClosedFormExpectations
    using StableRNGs
    using ExponentialFamily
    using Distributions

    nmv = NormalMeanVariance(0.0, 1.0)
    normal = Normal(0.0, 1.0)
    laplace = Laplace(0.0, 1.0)
    @test BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), nmv) ≈ BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, normal), normal)
    @test BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, laplace), nmv) ≈ BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, laplace), normal)
end

@testitem "Support GaussianDistributionsFamily" begin
    import BayesBase
    using Distributions
    using ExponentialFamily
    using ClosedFormExpectations
    using StableRNGs

    @testset "Expectations" begin
        nmv = NormalMeanVariance(0.0, 1.0)
        nmp = NormalMeanPrecision(0.0, 1.0)
    
        @test BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), nmv) isa Number
        @test BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, nmp), nmv) isa Number 
    end

    @testset "Gradients" begin
        using Distributions
        using ExponentialFamily

        nmv = NormalMeanVariance(0.0, 1.0)
        nmp = NormalMeanPrecision(0.0, 1.0)
        ef = convert(ExponentialFamilyDistribution, nmv)
    
        @test BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), Normal(0, 1)) isa AbstractArray
        @test BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmp), Normal(0, 1)) isa AbstractArray
        @test BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), Normal(0, 1)) ≈ BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmp), Normal(0, 1))
        @test BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), nmv) ≈ BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmp), nmv)
        @test BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, nmv), ef) isa AbstractArray
    end
end