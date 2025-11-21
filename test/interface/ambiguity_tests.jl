
@testitem "Ambiguity Tests" begin
    using ClosedFormExpectations
    using ExponentialFamily
    using Distributions
    using Test
    using StaticArrays
    
    # Import ProductOf explicitly if not exported by ExponentialFamily in this environment
    import ExponentialFamily: ProductOf

    @testset "ProductOf with ExponentialFamilyDistribution" begin
        # Create a ProductOf
        p = Logpdf(ProductOf(Normal(0,1), Normal(1,1)))
        
        # Test with NormalMeanVariance (EF)
        q_ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(0.0, 1.0))
        
        # This should trigger ambiguity if resolution methods are missing
        # because we have mean(::ProductOf, q) and mean(f, ::EFDist)
        grad = mean(ClosedWilliamsProduct(), p, q_ef)
        
        @test grad isa AbstractVector
        @test length(grad) == 2
        
        # Value check: should be sum of individual gradients
        g1 = mean(ClosedWilliamsProduct(), Logpdf(Normal(0,1)), q_ef)
        g2 = mean(ClosedWilliamsProduct(), Logpdf(Normal(1,1)), q_ef)
        @test grad ≈ g1 + g2
    end

    @testset "Distribution with ExponentialFamilyDistribution" begin
        d = Normal(0, 1)
        q_ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(0.0, 1.0))
        
        # This should trigger ambiguity: mean(d::Distribution, q) vs mean(f, q::EFDist)
        grad = mean(ClosedWilliamsProduct(), d, q_ef)
        
        @test grad isa AbstractVector
        @test length(grad) == 2
        
        # Should be same as Logpdf
        g_logpdf = mean(ClosedWilliamsProduct(), Logpdf(d), q_ef)
        @test grad ≈ g_logpdf
    end
    
    @testset "Gamma with ProductOf" begin
        p = Logpdf(ProductOf(Gamma(2,2), Gamma(3,3)))
        q_ef = convert(ExponentialFamilyDistribution, Gamma(2.0, 2.0))
        
        grad = mean(ClosedWilliamsProduct(), p, q_ef)
        @test grad isa AbstractVector
        @test length(grad) == 2
    end
end
