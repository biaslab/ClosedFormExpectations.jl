@testitem "meanlog(::Normal, ::Normal)" begin
    using Distributions
    using LogExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ_1  = rand(rng)*10, rand(rng)*5
        μ_2, σ_2  = rand(rng)*10, rand(rng)*5
        N = 10^5
        samples = rand(rng, Normal(μ_1, σ_1), N)
        log_samples = map(x -> logpdf(Normal(μ_2, σ_2),x),samples)
        @test sigma_rule(meanlog(ClosedFormExpectation(), Normal(μ_1, σ_1), Normal(μ_2, σ_2)), mean(log_samples), std(log_samples), N)
    end
end

@testitem "meanlog(::Normal, ::Laplace)" begin

end