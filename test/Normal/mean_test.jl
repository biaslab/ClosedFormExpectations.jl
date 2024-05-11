@testitem "mean(::ClosedFormExpectation, ::Logpdf{Normal}, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ_1  = rand(rng)*10, rand(rng)*5
        μ_2, σ_2  = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(Normal(μ_2, σ_2)), Normal(μ_1, σ_1))
    end
end

@testitem "mean(::ClosedFormExpectation, ::Logpdf{Laplace}, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ = rand(rng)*10, rand(rng)*5
        μ_2, θ = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, σ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::Abs, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        value = mean(ClosedFormExpectation(), Abs(), Normal(μ, σ))
        central_limit_theorem_test(ClosedFormExpectation(), Abs(), Normal(μ, σ))
    end

    @testset "compare with laplace" begin
        μ, σ = rand(rng)*10, rand(rng)*5
        θ = rand(rng)*10
        value = mean(ClosedFormExpectation(), Abs(), Normal(μ, σ))
        value_laplace = mean(ClosedFormExpectation(), Logpdf(Laplace(0, θ)), Normal(μ, σ))
        @test log(0.5*1/θ) - 1/θ * value ≈ value_laplace
    end
    
end

@testitem "mean(::ClosedFormExpectation, ::LogBesselk, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        ν = rand(rng)*10
        value = mean(ClosedFormExpectation(), LogBesselk(1), Normal(μ, σ))
        central_limit_theorem_test(ClosedFormExpectation(), LogBesselk(1), Normal(μ, σ))
    end
end