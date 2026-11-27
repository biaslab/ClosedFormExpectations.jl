@testitem "expectation_gradient(::ClosedFormExpectation, p::log, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(), log, ef, score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::ComposedFunction{Type{Square}, typeof(log)}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(), Square() ∘ log, ef, score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::ComposedFunction{typeof(log), ExpLogSquare}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), ef, score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, p::Fix1{logpdf, LogNormal}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(μ, σ)), ef, score)
    end
end
