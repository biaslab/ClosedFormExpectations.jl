@testitem "mean(::ClosedFormExpectation, ::typeof{log}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)
        log_samples = log.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), log, q), mean(log_samples), std(log_samples), N)
    end
end