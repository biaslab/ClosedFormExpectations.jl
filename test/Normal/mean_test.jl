@testitem "expectation(::ClosedFormExpectation, ::Fix1{logpdf, Normal}, ::Normal)" begin
    import BayesBase
    using Distributions
    using Distributions

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ_1  = rand(rng)*10, rand(rng)*5
        μ_2, σ_2  = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(μ_2, σ_2)), Normal(μ_1, σ_1))
    end
end

@testitem "expectation(::ClosedFormExpectation, ::Fix1{logpdf, Laplace}, ::Normal)" begin
    import BayesBase
    using Distributions
    using Distributions

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ = rand(rng)*10, rand(rng)*5
        μ_2, θ = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Laplace(μ_2, θ)), Normal(μ_1, σ))
    end
end

@testitem "expectation(::ClosedFormExpectation, ::Abs, ::Normal)" begin
    import BayesBase
    using Distributions
    using Distributions

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        value = BayesBase.expectation(ClosedFormExpectation(), Abs(), Normal(μ, σ))
        central_limit_theorem_test(ClosedFormExpectation(), Abs(), Normal(μ, σ))
    end

    @testset "compare with laplace" begin
        μ, σ = rand(rng)*10, rand(rng)*5
        θ = rand(rng)*10
        value = BayesBase.expectation(ClosedFormExpectation(), Abs(), Normal(μ, σ))
        value_laplace = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, Laplace(0, θ)), Normal(μ, σ))
        @test log(0.5*1/θ) - 1/θ * value ≈ value_laplace
    end
    
end

@testitem "expectation(::ClosedFormExpectation, ::Fix1{logpdf, LogGamma}, ::Normal)" begin
    import BayesBase
    using Distributions
    using Distributions

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, β = rand(rng)*10, 1+rand(rng)
        μ, σ = rand(rng)*10, 1+rand(rng)
        value = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, LogGamma(α, β)), Normal(μ, σ))
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogGamma(α, β)), Normal(μ, σ))
    end
end