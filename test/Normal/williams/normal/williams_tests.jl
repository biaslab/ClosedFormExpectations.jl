@testitem "expectation_gradient(::ClosedFormExpectation, p::Abs, q::Normal)" begin
    import BayesBase
    using Distributions
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Abs(), Normal(μ, σ), score, 10^5)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, Normal}, q::Normal)" begin
    import BayesBase
    using Distributions
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        μ2, σ2 = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(μ1, σ1)), Normal(μ2, σ2), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, Laplace}, q::Normal)" begin
    import BayesBase
    using Distributions
    include("normal_utils.jl")
    
    rng = StableRNG(123)

    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        loc, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Laplace(loc, θ)), Normal(μ, σ), score)
    end

    @testset "compare Base.Fix1(logpdf, Laplace) gradient with Abs gradient" begin
        for _ in 1:10
            μ, σ = rand(rng)*10, rand(rng)*5
            θ = rand(rng)*10
            williams_result_abs = -1/θ*BayesBase.expectation_gradient(ClosedFormExpectation(), Abs(), Normal(μ, σ))
            williams_result_laplace = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, Laplace(0, θ)), Normal(μ, σ))
            @test williams_result_abs ≈ williams_result_laplace
        end
    end   
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, LogGamma}, q::Normal)" begin
    import BayesBase
    using Distributions
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        α, β = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogGamma(α, β)), Normal(μ, σ), score)
    end   
end
