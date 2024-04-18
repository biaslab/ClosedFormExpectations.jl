@testitem "mean(::ClosedWilliamsProduct, ::log, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Exponential, x) = -1/mean(q) + x/(mean(q)^2)

    for _ in 1:10
        λ = rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), log, Exponential(λ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::log ∘ ExpLogSquare, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Exponential, x) = -1/mean(q) + x/(mean(q)^2)

    for _ in 1:10
        μ = rand(rng)*10
        σ = rand(rng)*10
        λ = rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), log ∘ ExpLogSquare(μ, σ), Exponential(λ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::Logpdf{LogNormal}, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Exponential, x) = -1/mean(q) + x/(mean(q)^2)

    for _ in 1:10
        μ = rand(rng)*10
        σ = rand(rng)*10
        λ = rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(LogNormal(μ, σ)), Exponential(λ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::Logpdf{Exponential}, ::Exponential}" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)
    score(q::Exponential, x) = -1/mean(q) + x/(mean(q)^2)

    for _ in 1:10
        λ1 = rand(rng)*10
        λ2 = rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(Exponential(λ2)), Exponential(λ1), score)
    end
end