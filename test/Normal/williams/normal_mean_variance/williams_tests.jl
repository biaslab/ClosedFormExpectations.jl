@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, Normal}, q::NormalMeanVariance)" begin
    import BayesBase
    using Distributions
    include("normal_mean_variance_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        μ2, σ2 = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(μ1, σ1)), NormalMeanVariance(μ2, σ2^2), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p, q::NormalMeanVariance)" begin
    import BayesBase
    using Distributions
    include("normal_mean_variance_utils.jl")

    @testset "equal to Normal under jacobian" begin
        rng = StableRNG(123)        
        μ, σ = rand(rng)*10, rand(rng)*10
        loc, θ = rand(rng)*10, rand(rng)*10 
            
        laplace = Laplace(loc, θ)
        normal = Normal(μ, σ)
        nmv = NormalMeanVariance(μ, σ^2)
        
        williams_result = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, laplace), nmv)
        williams_result_mv = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, laplace), normal)

        jacobian = [1 0; 0 0.5/std(normal)]

        @test williams_result ≈ jacobian * williams_result_mv
    end
end