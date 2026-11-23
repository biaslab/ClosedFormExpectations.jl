@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, Normal}, q::ExponentialFamilyDistribution{NormalMeanVariance})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        μ2, σ2 = rand(rng)*10, rand(rng)*5
        ef = convert(ExponentialFamilyDistribution, Normal(μ1, σ1))
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Normal(μ2, σ2)), ef, score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Abs, q::ExponentialFamilyDistribution{NormalMeanVariance})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        ef = convert(ExponentialFamilyDistribution, Normal(μ1, σ1))
        central_limit_theorem_test(ClosedFormExpectation(), Abs(), ef, score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, Laplace}, q::ExponentialFamilyDistribution{NormalMeanVariance})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        loc, θ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Normal(μ1, σ1))
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Laplace(loc, θ)), ef, score)
    end
end