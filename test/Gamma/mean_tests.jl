@testitem "expectation(::ClosedFormExpectation, ::typeof{log}, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, ::typeof{xlogx}, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, ::ComposedFunction{Type{Square}, typeof(log)}, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, ::ComposedFunction{Power{3}, typeof(log)}, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, ::xlog2x, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, log ∘ ExpLogSquare, ::Gamma)" begin
    import BayesBase
    using Distributions
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

@testitem "expectation(::ClosedFormExpectation, Fix1{logpdf, Lognormal}, ::Gamma)" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(μ, σ)), Gamma(α, θ))
    end
end

@testitem "expectation(::ClosedFormExpectation, p::Fix1{logpdf, <:UnivariateNormalDistributionsFamily}, q::Gamma)" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    using ExponentialFamily
    include("../test_utils.jl")
    rng = StableRNG(123)
    @testset "Fix1{logpdf, Gamma} on Gamma" begin
        for _ in 1:10
            p_dist = Normal(10rand(rng), 10rand(rng))
            q = Gamma(10rand(rng), 10rand(rng))
            central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_dist), q)
            for parametrization in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                p_params = convert(parametrization, p_dist)
                central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_params), q)
            end
        end 
    end
end


@testitem "Gamma Cross Entropy" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using SpecialFunctions
    include("../test_utils.jl")
    rng = StableRNG(123)
    @testset "Fix1{logpdf, Gamma} on Gamma" begin
        for _ in 1:10
            p_dist = Gamma(10rand(rng), 10rand(rng))
            q = Gamma(10rand(rng), 10rand(rng))
            central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_dist), q)
        end 
    end
end

@testitem "Gamma Cross Entropy supports GammaDistributionsFamily" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using ExponentialFamily: GammaShapeRate

    rng = StableRNG(123)
    for _ in 1:10
        p_dist = Gamma(10rand(rng), 10rand(rng))
        p_ef = convert(GammaShapeRate, p_dist)
        q = Gamma(10rand(rng), 10rand(rng))

        expected = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_dist), q)
        actual = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_ef), q)
        @test isapprox(actual, expected; rtol = 1e-10, atol = 1e-12)
    end
end
