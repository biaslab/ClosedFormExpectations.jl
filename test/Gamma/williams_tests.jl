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
        central_limit_theorem_test(ClosedWilliamsProduct(), log, Gamma(α, θ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::ComposedFunction{Type{Square}, typeof(log)}, ::Gamma)" begin
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma
    using SpecialFunctions
    using Distributions

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Gamma, x) = [log(x) -  log(scale(q)) - polygamma(0, shape(q)), x/scale(q)^2 - shape(q)/scale(q)]

    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        central_limit_theorem_test(ClosedWilliamsProduct(), Square() ∘ log, Gamma(α, θ), score)
    end
end
