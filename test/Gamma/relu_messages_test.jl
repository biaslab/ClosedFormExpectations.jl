@testitem "expectation(::ClosedFormExpectation, ::Fix1{logpdf, ReLUForwardMessage}, ::Gamma)" begin
    using Distributions, StableRNGs, ClosedFormExpectations, BayesBase

    include("gamma_utils.jl")
    rng = StableRNG(42)
    for _ in 1:10
        m_x = rand(rng) * 5
        v_x = 0.5 + rand(rng) * 4
        α   = 1.0 + rand(rng) * 9
        θ   = 0.5 + rand(rng) * 4
        central_limit_theorem_test(
            ClosedFormExpectation(),
            Base.Fix1(logpdf, ReLUForwardMessage(m_x, v_x)),
            Gamma(α, θ)
        )
    end
end

@testitem "expectation(::ClosedFormExpectation, ::Fix1{logpdf, ReLUBackwardMessage}, ::Gamma)" begin
    using Distributions, StableRNGs, ClosedFormExpectations, BayesBase

    include("gamma_utils.jl")
    rng = StableRNG(42)
    for _ in 1:10
        m_y = rand(rng) * 5
        v_y = 0.5 + rand(rng) * 4
        α   = 1.0 + rand(rng) * 9
        θ   = 0.5 + rand(rng) * 4
        central_limit_theorem_test(
            ClosedFormExpectation(),
            Base.Fix1(logpdf, ReLUBackwardMessage(m_y, v_y)),
            Gamma(α, θ)
        )
    end
end
