@testitem "mean(::ClosedFormExpectation, ::Logpdf{ReLUForwardMessage}, ::LogNormal)" begin
    using Distributions, StableRNGs, ClosedFormExpectations, BayesBase

    include("../test_utils.jl")
    rng = StableRNG(42)
    for _ in 1:10
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        μ_q = rand(rng) * 2
        σ_q = 0.1 + rand(rng) * 1.5
        central_limit_theorem_test(
            ClosedFormExpectation(),
            Logpdf(ReLUForwardMessage(m_x, v_x)),
            LogNormal(μ_q, σ_q)
        )
    end
end

@testitem "mean(::ClosedFormExpectation, ::Logpdf{ReLUBackwardMessage}, ::LogNormal)" begin
    using Distributions, StableRNGs, ClosedFormExpectations, BayesBase

    include("../test_utils.jl")
    rng = StableRNG(42)
    for _ in 1:10
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        μ_q = rand(rng) * 2
        σ_q = 0.1 + rand(rng) * 1.5
        central_limit_theorem_test(
            ClosedFormExpectation(),
            Logpdf(ReLUBackwardMessage(m_y, v_y)),
            LogNormal(μ_q, σ_q)
        )
    end
end
