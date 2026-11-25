@testitem "ClosedFormAD with Enzyme: Reverse vs manual on ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:5
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        for f in (log, Square() ∘ log)
            manual  = BayesBase.expectation_gradient(ClosedFormExpectation(),                            f, ef)
            via_rev = BayesBase.expectation_gradient(ClosedFormAD(AutoEnzyme()),             f, ef)
            via_fwd = BayesBase.expectation_gradient(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), f, ef)
            @test isapprox(via_rev, manual;  rtol = 1e-8)
            @test isapprox(via_fwd, manual;  rtol = 1e-8)
        end
    end
end

@testitem "ClosedFormAD with Enzyme: Reverse CLT on ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")
    rng = StableRNG(42)
    for _ in 1:3
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),              log, ef, score)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), log, ef, score)
    end
end

@testitem "ClosedFormAD with Enzyme: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{NormalMeanVariance}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = randn(rng) * 3
        v_y = 0.5 + rand(rng) * 4
        μ, σ = randn(rng) * 3, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(μ, σ^2))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "ClosedFormAD with Enzyme: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),               Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score)
    end
end

@testitem "ClosedFormAD with Enzyme: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score)
    end
end

@testitem "ClosedFormAD with Enzyme: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),               Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score_ef)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)), ef, score_ef)
    end
end

@testitem "ClosedFormAD with Enzyme: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme()),               Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)), Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "ClosedFormAD with Enzyme: Reverse CLT on ExponentialFamilyDistribution{LogNormal}" begin
    import BayesBase
    using Distributions
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme, StableRNGs
    include("../test_utils.jl")

    score_lognormal(q::ExponentialFamilyDistribution, x) =
        sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        μ_q, σ_q = rand(rng) * 2, rand(rng) * 0.5 + 0.1
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))

        α_p, θ_p = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        central_limit_theorem_test(
            ClosedFormAD(AutoEnzyme()),
            Base.Fix1(logpdf, Gamma(α_p, θ_p)), ef, score_lognormal,
        )
        central_limit_theorem_test(
            ClosedFormAD(AutoEnzyme(mode = Enzyme.Forward)),
            Base.Fix1(logpdf, Gamma(α_p, θ_p)), ef, score_lognormal,
        )
    end
end

@testitem "ClosedFormAD with Enzyme: through the protocol's routers and support" begin
    import BayesBase
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme

    cfe = ClosedFormExpectation()
    backend = ClosedFormAD(AutoEnzyme())
    ef = convert(ExponentialFamilyDistribution, Gamma(2.5, 1.5))
    manual = BayesBase.expectation_gradient(cfe, log, ef)

    @test BayesBase.supports_expectation_gradient(backend, log, ef)
    @test BayesBase.is_expectation_exact(backend, BayesBase.ExpectationScoreProduct(log, ef), ef)
    @test !BayesBase.supports_expectation_gradient(backend, sin, ef)
    @test !BayesBase.supports_expectation_gradient(backend, log, Gamma(2.5, 1.5)) # only exponential family forms
    @test !BayesBase.supports_expectation(backend, log, ef) # only gradients
    @test BayesBase.expectation_gradient(BayesBase.ExpectationFirstOf(backend), log, ef) ≈ manual rtol = 1e-8
    @test_throws BayesBase.UnsupportedExpectationError BayesBase.expectation_gradient(backend, sin, ef)

    # Multivariate forms are not claimed: Enzyme cannot differentiate their closed forms
    mvnormal = MvNormalMeanCovariance([0.1, -0.2], [1.0 0.3; 0.3 2.0])
    efm = convert(ExponentialFamilyDistribution, mvnormal)
    @test BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, mvnormal), efm)
    @test !BayesBase.supports_expectation_gradient(backend, Base.Fix1(logpdf, mvnormal), efm)
    @test !BayesBase.supports_expectation_gradient(backend, Base.Fix1(logpdf, LinearLogGamma(2.0, 3.0, [0.5, -1.0])), efm)

    # After the formulas of the package in a router, automatic differentiation computes the rest
    router = BayesBase.ExpectationFirstOf(cfe, backend)
    lognormal = convert(ExponentialFamilyDistribution, LogNormal(0.2, 0.5))
    f = Base.Fix1(logpdf, Gamma(2.0, 3.0))
    @test !BayesBase.supports_expectation_gradient(cfe, f, lognormal)
    @test BayesBase.which_expectation_backend(router, BayesBase.ExpectationScoreProduct(f, lognormal), lognormal) === backend
    @test BayesBase.which_expectation_backend(router, BayesBase.ExpectationScoreProduct(log, ef), ef) === cfe
    @test all(isfinite, BayesBase.expectation_gradient(router, f, lognormal))
end

@testitem "ClosedFormAD with Enzyme: support implies success" setup = [SupportCases] begin
    import BayesBase
    using Distributions, ExponentialFamily, Enzyme
    import ADTypes: AutoEnzyme

    cfe = ClosedFormExpectation()
    for ad in (AutoEnzyme(), AutoEnzyme(mode = Enzyme.Forward))
        backend = ClosedFormAD(ad)
        supported = 0
        for (f, q) in cases
            BayesBase.supports_expectation_gradient(backend, f, q) || continue
            supported += 1
            @testset let f = f, q = q
                gradient = BayesBase.expectation_gradient(backend, f, q)
                @test all(isfinite, gradient)
                if BayesBase.supports_expectation_gradient(cfe, f, q)
                    @test gradient ≈ BayesBase.expectation_gradient(cfe, f, q) rtol = 1e-8
                end
            end
        end
        @test supported > 20
    end
end
