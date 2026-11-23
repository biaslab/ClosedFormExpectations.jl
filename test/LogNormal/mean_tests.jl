@testitem "expectation(::ClosedFormExpectation, Fix1{logpdf, <:GammaDistributionsFamily}, ::LogNormal)" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using ExponentialFamily: GammaShapeRate, GammaShapeScale
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        α_p, θ_p = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        μ_q, σ_q = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        q = LogNormal(μ_q, σ_q)
        p_base = Gamma(α_p, θ_p)
        p_rate = convert(GammaShapeRate, p_base)
        p_scale = convert(GammaShapeScale, p_base)
        expected = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_base), q)
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_base), q)
        @test isapprox(BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_rate), q), expected; rtol = 1e-10)
        @test isapprox(BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_scale), q), expected; rtol = 1e-10)
    end
end

@testitem "expectation(::ClosedFormExpectation, Fix1{logpdf, LogNormal}, ::LogNormal)" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_p, σ_p = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        μ_q, σ_q = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(μ_p, σ_p)), LogNormal(μ_q, σ_q))
    end
end

@testitem "expectation(::ClosedFormExpectation, Fix1{logpdf, <:UnivariateNormalDistributionsFamily}, ::LogNormal)" begin
    import BayesBase
    using Distributions
    using Distributions
    using ClosedFormExpectations
    using StableRNGs
    using ExponentialFamily: NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ_p, σ_p = rand(rng) * 10, rand(rng) * 5 + 0.1
        μ_q, σ_q = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        q = LogNormal(μ_q, σ_q)
        p_base = Normal(μ_p, σ_p)
        expected = BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_base), q)
        central_limit_theorem_test(ClosedFormExpectation(), Base.Fix1(logpdf, p_base), q)
        for parametrization in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
            p_param = convert(parametrization, p_base)
            @test isapprox(BayesBase.expectation(ClosedFormExpectation(), Base.Fix1(logpdf, p_param), q), expected; rtol = 1e-10)
        end
    end
end
