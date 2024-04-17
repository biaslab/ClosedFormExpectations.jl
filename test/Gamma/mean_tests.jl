@testitem "mean(::ClosedFormExpectation, ::typeof{log}, ::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)
        log_samples = log.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), log, q), mean(log_samples), std(log_samples), N)
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
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)
        xlog_samples = xlogx.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), xlogx, q), mean(xlog_samples), std(xlog_samples), N)
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
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)

        fn = Square() ∘ log
        transformed_samples = fn.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), fn, q), mean(transformed_samples), std(transformed_samples), N)
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
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)

        fn = Power(Val(3)) ∘ log
        transformed_samples = fn.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), fn, q), mean(transformed_samples), std(transformed_samples), N)
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
        N = 10^6
        
        q = Gamma(α, θ)

        samples = rand(rng, q, 10^6)

        transformed_samples = xlog2x.(samples)
        @test sigma_rule(mean(ClosedFormExpectation(), xlog2x, q), mean(transformed_samples), std(transformed_samples), N)
    end
end