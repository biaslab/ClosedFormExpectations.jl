@testitem "mean(::ClosedFormExpectation, ::Logpdf{LogNormal}, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    
    for _ in 1:10   
        μ, σ  = rand(rng)*10, rand(rng)*10
        λ = rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(LogNormal(μ, σ)), Exponential(λ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::typeof{log}, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        λ = rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), log, Exponential(λ))
    end
end

@testitem "mean(::ClosedFormExpectations, ::ComposedFunction{typeof(log), ExpLogSquare}, ::Exponential)" begin
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
        central_limit_theorem_test(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), Exponential(λ), N)
    end
end

@testitem "mean(::ClosedFormExpectation, ::Logpdf{Exponential}, ::Exponential)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        λ1 = rand(rng)*10
        λ2 = rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(Exponential(λ2)), Exponential(λ1))
    end
end