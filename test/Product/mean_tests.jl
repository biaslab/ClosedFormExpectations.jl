@testitem "mean(::ClosedFormExpectation, ::Product, ::Exponential)" begin
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
        product = ClosedFormExpectations.Product((ExpLogSquare(μ, σ), identity))
        sum_mean = mean(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), Exponential(λ)) + mean(ClosedFormExpectation(), log, Exponential(λ))
        @test mean(ClosedFormExpectation(), log ∘ product, Exponential(λ)) ≈ sum_mean
    end
end