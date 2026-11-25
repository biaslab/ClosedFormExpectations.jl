@testitem "expectation(::ClosedFormExpectation, ::Product, ::Exponential)" begin
    import BayesBase
    using Distributions
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
        sum_mean = BayesBase.expectation(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), Exponential(λ)) + BayesBase.expectation(ClosedFormExpectation(), log, Exponential(λ))
        @test BayesBase.expectation(ClosedFormExpectation(), log ∘ product, Exponential(λ)) ≈ sum_mean
    end
end
@testitem "log ∘ Product can be evaluated at a point" begin
    import ClosedFormExpectations: Product, Square
    f = Product((ExpLogSquare(0.1, 0.9), Square()))
    x = 1.7
    @test f(x) ≈ ExpLogSquare(0.1, 0.9)(x) * x^2
    @test (log ∘ f)(x) ≈ log(f(x))
end
