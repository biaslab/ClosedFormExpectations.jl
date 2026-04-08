@testitem "mean(::ClosedFormExpectation, ::Logpdf{ReLUBackwardMessage}, ::Normal)" begin
    using Distributions, StableRNGs, ClosedFormExpectations, BayesBase

    include("../test_utils.jl")
    rng = StableRNG(42)
    for _ in 1:10
        m_y = randn(rng) * 3
        v_y = 0.5 + rand(rng) * 4
        μ   = randn(rng) * 3
        σ   = 0.5 + rand(rng) * 4
        central_limit_theorem_test(
            ClosedFormExpectation(),
            Logpdf(ReLUBackwardMessage(m_y, v_y)),
            Normal(μ, σ)
        )
    end
end
