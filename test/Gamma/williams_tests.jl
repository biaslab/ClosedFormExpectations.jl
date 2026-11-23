@testitem "expectation_gradient(::ClosedFormExpectation, p::log, q::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), log, Gamma(α, θ), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, ::ComposedFunction{Type{Square}, typeof(log)}, ::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        N = 10^6
        central_limit_theorem_test(ClosedFormExpectation(), Square() ∘ log, Gamma(α, θ), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, ::ComposedFunction{typeof(log), ExpLogSquare{T}}, ::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand()*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), Gamma(α, θ), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, ::Base.Fix1(logpdf, LogNormal), ::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α, θ = rand(rng)*10, rand(rng)*10
        μ, σ = rand()*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(μ, σ)), Gamma(α, θ), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, ::Base.Fix1(logpdf, Gamma), ::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α_p, θ_p = rand(rng)*10, rand(rng)*10
        α_q, θ_q = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, Gamma(α_p, θ_p)), Gamma(α_q, θ_q), score)
    end
end

@testitem "expectation_gradient(::ClosedFormExpectation, ::Base.Fix1(logpdf, GammaDistributionsFamily), ::Gamma)" begin
    import BayesBase
    using Distributions
    include("gamma_utils.jl")
    using ExponentialFamily: GammaShapeRate

    rng = StableRNG(123)
    for _ in 1:10
        p_dist = Gamma(10rand(rng), 10rand(rng))
        p_ef = convert(GammaShapeRate, p_dist)
        q = Gamma(10rand(rng), 10rand(rng))

        expected = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, p_dist), q)
        actual = BayesBase.expectation_gradient(ClosedFormExpectation(), Base.Fix1(logpdf, p_ef), q)
        @test isapprox.(actual, expected; rtol = 1e-10, atol = 1e-12) |> all
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
    include("gamma_utils.jl")
    rng = StableRNG(123)
    @testset "Fix1{logpdf, Gamma} on Gamma" begin
        for _ in 1:10
            p_dist = Normal(10rand(rng), 10rand(rng))
            q = Gamma(10rand(rng), 10rand(rng))
            central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_dist), q, score)
            for parametrization in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                p_params = convert(parametrization, p_dist)
                central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_params), q, score)
            end
        end 
    end
end