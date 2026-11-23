@testitem "EnzymeBackend: Reverse vs manual on ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:5
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        for f in (log, Square() ∘ log)
            manual  = BayesBase.expectation_gradient(ClosedFormExpectation(),                            f, ef)
            via_rev = BayesBase.expectation_gradient(ClosedFormExpectation(EnzymeBackend()),             f, ef)
            via_fwd = BayesBase.expectation_gradient(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), f, ef)
            @test isapprox(via_rev, manual;  rtol = 1e-8)
            @test isapprox(via_fwd, manual;  rtol = 1e-8)
        end
    end
end

@testitem "EnzymeBackend: Reverse CLT on ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")
    rng = StableRNG(42)
    for _ in 1:3
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),              log, ef, score)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), log, ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{NormalMeanVariance}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = randn(rng) * 3
        v_y = 0.5 + rand(rng) * 4
        μ, σ = randn(rng) * 3, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(μ, σ^2))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),               Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),               Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score_ef)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedFormExpectation(EnzymeBackend(EnzymeForward())), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: Reverse CLT on ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_lognormal(q::ExponentialFamilyDistribution, x) =
        sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        μ_q, σ_q = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))

        α_p, θ_p = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        central_limit_theorem_test(
            ClosedFormExpectation(EnzymeBackend()),
            Base.Fix1(logpdf, Gamma(α_p, θ_p)), ef, score_lognormal,
        )
        central_limit_theorem_test(
            ClosedFormExpectation(EnzymeBackend(EnzymeForward())),
            Base.Fix1(logpdf, Gamma(α_p, θ_p)), ef, score_lognormal,
        )
    end
end

@testitem "EnzymeBackend: through the protocol's routers and support" begin
    import BayesBase
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme

    backend = ClosedFormExpectation(EnzymeBackend())
    ef = convert(ExponentialFamilyDistribution, Gamma(2.5, 1.5))
    manual = BayesBase.expectation_gradient(ClosedFormExpectation(), log, ef)

    @test BayesBase.supports_expectation_gradient(backend, log, ef)
    @test !BayesBase.supports_expectation_gradient(backend, sin, ef)
    @test !BayesBase.supports_expectation_gradient(backend, log, Gamma(2.5, 1.5)) # only exponential family forms
    @test BayesBase.expectation_gradient(BayesBase.ExpectationFirstOf(backend), log, ef) ≈ manual rtol = 1e-8
    @test BayesBase.expectation(backend, log, ef) == BayesBase.expectation(ClosedFormExpectation(), log, ef)
    @test_throws BayesBase.UnsupportedExpectationError BayesBase.expectation_gradient(backend, sin, ef)
end
