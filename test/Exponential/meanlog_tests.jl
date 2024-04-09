@testitem "meanlog(::Exponential, ::LogNormal)" begin
    using Distributions
    using LogpdfExpectation
    using StableRNGs

    for _ in 1:10
        rng = StableRNG(123)
        μ, σ  = rand(rng)*10, rand(rng)*5
        λ = rand(rng)*10

        samples = rand(rng, Exponential(λ), 10^3)

        @test meanlog(ClosedFormExpectation(), Exponential(λ), LogNormal(μ, σ)) ≈ mean(log, pdf(LogNormal(μ, σ), samples))
    end
end