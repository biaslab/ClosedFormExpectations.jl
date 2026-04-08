@testitem "EnzymeBackend: Reverse vs manual on ExponentialFamilyDistribution{Gamma}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")
    rng = StableRNG(123)
    for _ in 1:5
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        for f in (log, Square() ∘ log)
            manual  = mean(ClosedWilliamsProduct(),                            f, ef)
            via_rev = mean(ClosedWilliamsProduct(EnzymeBackend()),             f, ef)
            via_fwd = mean(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), f, ef)
            @test isapprox(via_rev, manual;  rtol = 1e-8)
            @test isapprox(via_fwd, manual;  rtol = 1e-8)
        end
    end
end

@testitem "EnzymeBackend: Reverse CLT on ExponentialFamilyDistribution{Gamma}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")
    rng = StableRNG(42)
    for _ in 1:3
        α, θ = rand(rng) * 5 + 0.5, rand(rng) * 5 + 0.5
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),              log, ef, score)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), log, ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{NormalMeanVariance}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = randn(rng) * 3
        v_y = 0.5 + rand(rng) * 4
        μ, σ = randn(rng) * 3, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, NormalMeanVariance(μ, σ^2))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),               Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),               Logpdf(ReLUForwardMessage(m_x, v_x)), ef, score)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(ReLUForwardMessage(m_x, v_x)), ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{Gamma}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../Gamma/williams/ef_parametrization/ef_utils.jl")

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        α, θ = 1.0 + rand(rng) * 9, 0.5 + rand(rng) * 4
        ef = convert(ExponentialFamilyDistribution, Gamma(α, θ))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),               Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score)
    end
end

@testitem "EnzymeBackend: CLT on ReLUForwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),               Logpdf(ReLUForwardMessage(m_x, v_x)), ef, score_ef)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(ReLUForwardMessage(m_x, v_x)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: CLT on ReLUBackwardMessage × ExponentialFamilyDistribution{LogNormal}" begin
    using Distributions, ClosedFormExpectations, ExponentialFamily, Enzyme, StableRNGs
    include("../test_utils.jl")

    score_ef(q, x) = sufficientstatistics(q, x) .- gradlogpartition(q)

    rng = StableRNG(42)
    for _ in 1:3
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        μ_q, σ_q = rand(rng) * 2, 0.1 + rand(rng) * 1.5
        ef = convert(ExponentialFamilyDistribution, LogNormal(μ_q, σ_q))
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend()),               Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
        central_limit_theorem_test(ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())), Logpdf(ReLUBackwardMessage(m_y, v_y)), ef, score_ef)
    end
end

@testitem "EnzymeBackend: Reverse CLT on ExponentialFamilyDistribution{LogNormal}" begin
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
            ClosedWilliamsProduct(EnzymeBackend()),
            Logpdf(Gamma(α_p, θ_p)), ef, score_lognormal,
        )
        central_limit_theorem_test(
            ClosedWilliamsProduct(EnzymeBackend(EnzymeForward())),
            Logpdf(Gamma(α_p, θ_p)), ef, score_lognormal,
        )
    end
end
