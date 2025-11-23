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

@testitem "mean(::ClosedFormExpectation, p::Logpdf{<:UnivariateNormalDistributionsFamily}, q::Gamma)" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    using ExponentialFamily
    include("../test_utils.jl")
    rng = StableRNG(123)
    @testset "Logpdf{Gamma} on Gamma" begin
        for _ in 1:10
            p_dist = Normal(10rand(rng), 10rand(rng))
            q = Gamma(10rand(rng), 10rand(rng))
            central_limit_theorem_test(ClosedFormExpectation(), Logpdf(p_dist), q)
            for parametrization in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                p_params = convert(parametrization, p_dist)
                central_limit_theorem_test(ClosedFormExpectation(), Logpdf(p_params), q)
            end
        end 
    end
end


@testitem "Gamma Cross Entropy" begin
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    include("../test_utils.jl")
    rng = StableRNG(123)
    @testset "Logpdf{Gamma} on Gamma" begin
        for _ in 1:10
            p_dist = Gamma(10rand(rng), 10rand(rng))
            q = Gamma(10rand(rng), 10rand(rng))
            central_limit_theorem_test(ClosedFormExpectation(), Logpdf(p_dist), q)
        end 
    end
end
