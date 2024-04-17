@testitem "mean(::ClosedWilliamsProduct, p::log, q::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma
    using SpecialFunctions

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Gamma, x) = [log(x) -  log(scale(q)) - polygamma(0, shape(q)), x/scale(q)^2 - shape(q)/scale(q)]

    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        q = Gamma(α, θ)
        samples = rand(rng, q, 10^6)
        williams_product = map(x -> score(q, x)*log(x), samples)
        for (expectation, mean, std) in zip(mean(ClosedWilliamsProduct(), log, q), mean(williams_product), std(williams_product))
            @test sigma_rule(expectation, mean, std, N)
        end
    end
end