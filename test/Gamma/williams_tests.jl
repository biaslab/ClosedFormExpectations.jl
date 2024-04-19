@testitem "mean(::ClosedWilliamsProduct, p::log, q::Gamma)" begin
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), log, Gamma(α, θ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::ComposedFunction{Type{Square}, typeof(log)}, ::Gamma)" begin
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        central_limit_theorem_test(ClosedWilliamsProduct(), Square() ∘ log, Gamma(α, θ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::ComposedFunction{typeof(log), ExpLogSquare{T}}, ::Gamma)" begin
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand()*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), log ∘ ExpLogSquare(μ, σ), Gamma(α, θ), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, ::Logpdf(LogNormal), ::Gamma)" begin
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand()*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(LogNormal(μ, σ)), Gamma(α, θ), score)
    end
end