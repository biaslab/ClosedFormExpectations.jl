@testitem "meanlog(::Exponential, ::LogNormal)" begin
    using Distributions
    using LogExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)

    for _ in 1:10   
        μ, σ  = rand(rng)*10, rand(rng)*10
        λ = rand(rng)*10
        N = 10^6
        samples = rand(rng, Exponential(λ), N)
        logpdf_samples = logpdf(LogNormal(μ, σ), samples)        
        expectation = meanlog(ClosedFormExpectation(), Exponential(λ), LogNormal(μ, σ))
        # @show expectation
        # @show mean(logpdf_samples)
        @test sigma_rule(expectation, mean(logpdf_samples), std(logpdf_samples), 10^6)
    end
end

@testitem "meanlog(::Exponential, ::identity)" begin
    using Distributions
    using LogExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        λ = rand(rng)*10
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        log_samples = log.(samples)
        @test sigma_rule(meanlog(ClosedFormExpectation(), Exponential(λ), identity), mean(log_samples), std(log_samples), N)
    end
end

@testitem "meanlog(::Exponential, ::ExpLogSquare)" begin
    using Distributions
    using LogExpectations
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
        @test sigma_rule(meanlog(ClosedFormExpectation(), Exponential(λ), ExpLogSquare(μ, σ)), mean(log_samples), std(log_samples), N)
    end
end