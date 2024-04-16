@testitem "meanlog(::Normal, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ_1  = rand(rng)*10, rand(rng)*5
        μ_2, σ_2  = rand(rng)*10, rand(rng)*5
        N = 10^5
        samples = rand(rng, Normal(μ_1, σ_1), N)
        log_samples = map(x -> logpdf(Normal(μ_2, σ_2),x),samples)
        @test sigma_rule(mean(ClosedFormExpectation(), Normal(μ_1, σ_1), log ∘ Normal(μ_2, σ_2)), mean(log_samples), std(log_samples), N)
    end
end

@testitem "meanlog(::Normal, ::Laplace)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    N = 10^5
    for _ in 1:10
        μ_1, σ  = rand(rng)*10, rand(rng)*5
        μ_2, θ   = rand(rng)*10, rand(rng)*5
        normal = Normal(μ_1, σ)
        laplace = Laplace(μ_2, θ)
        samples = rand(rng, normal, N)
        log_samples = map(x -> logpdf(laplace,x),samples)
        @test sigma_rule(mean(ClosedFormExpectation(), normal, log ∘ laplace), mean(log_samples), std(log_samples), N)
    end
end