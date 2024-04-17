@testitem "mean(::ClosedFormExpectation, ::typeof{log}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), log, Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::typeof{xlogx}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using LogExpFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), xlogx, Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::ComposedFunction{Type{Square}, typeof(log)}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Square() ∘ log, Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::ComposedFunction{Power{3}, typeof(log)}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Power(Val(3)) ∘ log, Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, ::xlog2x, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), xlog2x, Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, log ∘ ExpLogSquare, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), Gamma(α, θ))
    end
end

@testitem "mean(::ClosedFormExpectation, Logpdf{Lognormal}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(LogNormal(μ, σ)), Gamma(α, θ))
    end
end