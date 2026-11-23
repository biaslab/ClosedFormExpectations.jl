@testitem "No method of the package is ambiguous" begin
    using Test
    @test isempty(Test.detect_ambiguities(ClosedFormExpectations))
end

@testitem "Products of log-densities under the exponential family form" begin
    import BayesBase
    using Distributions
    using ExponentialFamily
    using Distributions
    import ExponentialFamily: ProductOf

    @testset "Under a NormalMeanVariance" begin
        p = Base.Fix1(logpdf, ProductOf(Normal(0, 1), Normal(1, 1)))
        q_ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(0.0, 1.0))
        grad = BayesBase.expectation_gradient(ClosedFormExpectation(), p, q_ef)
        @test grad isa AbstractVector
        @test length(grad) == 2

        # The gradient of a product is the sum of the gradients of its parts
        g1 = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(0, 1)), q_ef)
        g2 = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(1, 1)), q_ef)
        @test grad ≈ g1 + g2
    end

    @testset "Under a Gamma" begin
        p = Base.Fix1(logpdf, ProductOf(Gamma(2, 2), Gamma(3, 3)))
        q_ef = convert(ExponentialFamilyDistribution, Gamma(2.0, 2.0))
        grad = BayesBase.expectation_gradient(ClosedFormExpectation(), p, q_ef)
        @test grad isa AbstractVector
        @test length(grad) == 2
    end
end
