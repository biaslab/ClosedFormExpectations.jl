@testset "Support ::Base.Fix1{typeof(logpdf), D}" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ = rand(rng)*10
        σ = rand(rng)*10
        λ = rand(rng)*10
        N = 10^5
        samples = rand(rng, Exponential(λ), N)
        log_samples = log.(ExpLogSquare(μ, σ).(samples))
        @test sigma_rule(mean(ClosedFormExpectation(), Exponential(λ), log ∘ ExpLogSquare(μ, σ)), mean(log_samples), std(log_samples), N)
    end
end