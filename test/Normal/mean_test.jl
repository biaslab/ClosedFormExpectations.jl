@testitem "mean(::ClosedFormExpectation, ::Logpdf{Normal}, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ_1  = rand(rng)*10, rand(rng)*5
        μ_2, σ_2  = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(Normal(μ_2, σ_2)), Normal(μ_1, σ_1))
    end
end

@testitem "mean(::ClosedFormExpectation, ::Logpdf{Laplace}, ::Normal)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs

    include("../test_utils.jl")
    rng = StableRNG(123)
    N = 10^5
    for _ in 1:10
        μ_1, σ  = rand(rng)*10, rand(rng)*5
        μ_2, θ   = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, σ))
    end
end