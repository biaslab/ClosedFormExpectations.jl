@testitem "mean(::Exponential, ::ComposedFunction{typeof(log), LogNormal})" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)

    for _ in 1:10   
        μ, σ  = rand(rng)*10, rand(rng)*10
        λ = rand(rng)*10
        N = 10^6
        samples = rand(rng, Exponential(λ), N)
        logpdf_samples = logpdf(LogNormal(μ, σ), samples)        
        expectation = mean(ClosedFormExpectation(), Exponential(λ), log ∘ LogNormal(μ, σ))
        @test sigma_rule(expectation, mean(logpdf_samples), std(logpdf_samples), 10^6)
    end
end

@testitem "mean(::Exponential, ::typeof{log})" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        λ = rand(rng)*10
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        log_samples = log.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), Exponential(λ), log), mean(log_samples), std(log_samples), N)
    end
end

@testitem "mean(::Exponential, ::ComposedFunction{typeof(log), ExpLogSquare})" begin
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

@testitem "mean(::Exponential, ::ComposedFunction{typeof(log), ExpLogSquare x identity}" begin
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
        sum_mean = mean(ClosedFormExpectation(), Exponential(λ), log ∘ ExpLogSquare(μ, σ)) + mean(ClosedFormExpectation(), Exponential(λ), log)
        @test mean(ClosedFormExpectation(), Exponential(λ), log ∘ product) ≈ sum_mean
    end
end