@testitem "mean(::ClosedWilliamsProduct, q::Exponential, p::typeof(identity))" begin
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
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        williams_product = map(x -> score(Exponential(λ), x)*log(x), samples)
        expectation = mean(ClosedWilliamsProduct(), identity, Exponential(λ))
        @test sigma_rule(expectation, mean(williams_product), std(williams_product), N)
    end
end

@testitem "mean(::ClosedWilliamsProduct, q::Exponential, p::ExpLogSquare)" begin
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
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        fn(x)  = (log ∘ ExpLogSquare(μ, σ))(x)
        williams_product = map(x -> score(Exponential(λ), x)*fn(x), samples)
        expectation = mean(ClosedWilliamsProduct(), ExpLogSquare(μ, σ), Exponential(λ))
        @test sigma_rule(expectation, mean(williams_product), std(williams_product), N)
    end
end

@testitem "mean(::ClosedWilliamsProduct, q::Exponential, p::LogNormal)" begin
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
        N = 10^6
        samples = rand(rng, Exponential(λ), 10^6)
        fn(x)  = logpdf(LogNormal(μ, σ), x)
        williams_product = map(x -> score(Exponential(λ), x)*fn(x), samples)
        expectation = mean(ClosedWilliamsProduct(), LogNormal(μ, σ), Exponential(λ))
        @test sigma_rule(expectation, mean(williams_product), std(williams_product), N)
    end
end

@testitem "mean(::ClosedWilliamsProduct, Exponential, Exponential}" begin
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
        N = 10^6
        samples = rand(rng, Exponential(λ1), N)
        fn(x) = logpdf(Exponential(λ2), x)
        williams_product = map(x -> score(Exponential(λ1), x)*fn(x), samples)
        @test sigma_rule(mean(ClosedWilliamsProduct(), Exponential(λ2), Exponential(λ1)), mean(williams_product), std(williams_product), N)
    end
end