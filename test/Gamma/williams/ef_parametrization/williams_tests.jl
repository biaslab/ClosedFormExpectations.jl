@testitem "mean(::ClosedWilliamsProduct, p::log, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(), log, ef, score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, p::ComposedFunction{Type{Square}, typeof(log)}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(), Square() ∘ log, ef, score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, p::ComposedFunction{typeof(log), ExpLogSquare}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(), log ∘ ExpLogSquare(μ, σ), ef, score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, p::Logpdf{LogNormal}, q::ExponentialFamilyDistribution{Gamma})" begin
    include("ef_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand(rng)*10, rand(rng)*10
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(LogNormal(μ, σ)), ef, score)
    end
end
