@testitem "mean(::ClosedWilliamsProduct, q::Exponential, p::typeof(identity))" begin
    using Distributions
    using LogExpectations
    using StableRNGs
    using Base.MathConstants: eulergamma

    include("../test_utils.jl")
    rng = StableRNG(123)

    wprod = ClosedWilliamsProduct()

    score(q::Exponential, x) = -1/mean(q) + x/(mean(q)^2)

    for _ in 1:10
        λ = rand(rng)*10
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        williams_product = map(x -> score(Exponential(λ), x)*log(x), samples)
        expectation = mean(ClosedWilliamsProduct(), Exponential(λ), identity)
        @test sigma_rule(expectation, mean(williams_product), std(williams_product), N)
    end
end

@testitem "meanlog(::ClosedWilliamsProduct, q::Exponential, p::ExpLogSquare)" begin
    using Distributions
    using LogExpectations
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
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        fn(x)  = (log ∘ ExpLogSquare(μ, σ))(x)
        williams_product = map(x -> score(Exponential(λ), x)*fn(x), samples)
        expectation = mean(ClosedWilliamsProduct(), Exponential(λ), ExpLogSquare(μ, σ))
        @test sigma_rule(expectation, mean(williams_product), std(williams_product), N)
    end
end

@testitem "meanlog(::ClosedWilliamsProduct, q::Exponential, p::LogNormal)" begin
    using Distributions
    using LogExpectations
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
        N = 10^7
        samples = rand(rng, Exponential(λ), 10^6)
        fn(x)  = logpdf(LogNormal(μ, σ), x)
        williams_product = map(x -> score(Exponential(λ), x)*fn(x), samples)
        expectation = mean(ClosedWilliamsProduct(), Exponential(λ), ExpLogSquare(μ, σ))
        @show mean(williams_product)
        @show expectation
    end
end