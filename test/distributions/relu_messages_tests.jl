@testitem "ReLUForwardMessage logpdf" begin
    using ClosedFormExpectations, BayesBase, Distributions, StableRNGs

    rng = StableRNG(42)
    for _ in 1:20
        m_x = randn(rng) * 3
        v_x = 0.1 + rand(rng) * 5

        d = ReLUForwardMessage(m_x, v_x)

        # For y > 0: continuous component equals N(y; m_x, v_x)
        y_pos = 0.01 + rand(rng) * 10
        @test logpdf(d, y_pos) ≈ logpdf(Normal(m_x, sqrt(v_x)), y_pos)

        # For y <= 0: -Inf
        @test logpdf(d, 0.0) == -Inf
        @test logpdf(d, -rand(rng)) == -Inf
    end
end

@testitem "ReLUBackwardMessage logpdf" begin
    using ClosedFormExpectations, BayesBase, Distributions, StableRNGs

    rng = StableRNG(42)
    for _ in 1:20
        m_y = randn(rng) * 3
        v_y = 0.1 + rand(rng) * 5

        d = ReLUBackwardMessage(m_y, v_y)

        # For x > 0: equals N(x; m_y, v_y)
        x_pos = 0.01 + rand(rng) * 10
        @test logpdf(d, x_pos) ≈ logpdf(Normal(m_y, sqrt(v_y)), x_pos)

        # For x <= 0: constant, equals N(0; m_y, v_y)
        x_neg = -rand(rng) * 5
        @test logpdf(d, x_neg) ≈ logpdf(Normal(m_y, sqrt(v_y)), 0.0)
        @test logpdf(d, 0.0) ≈ logpdf(Normal(m_y, sqrt(v_y)), 0.0)

        # The backward message is constant on the negative half-line
        x_neg2 = -rand(rng) * 10
        @test logpdf(d, x_neg) ≈ logpdf(d, x_neg2)
    end
end

@testitem "ReLUForwardMessage type promotion" begin
    using ClosedFormExpectations, BayesBase

    d_ff = ReLUForwardMessage(1.0f0, 2.0f0)
    @test d_ff isa ReLUForwardMessage{Float32}

    d_fd = ReLUForwardMessage(1.0, 2.0f0)
    @test d_fd isa ReLUForwardMessage{Float64}
end

@testitem "ReLUBackwardMessage type promotion" begin
    using ClosedFormExpectations, BayesBase

    d_ff = ReLUBackwardMessage(1.0f0, 2.0f0)
    @test d_ff isa ReLUBackwardMessage{Float32}

    d_fd = ReLUBackwardMessage(1.0, 2.0f0)
    @test d_fd isa ReLUBackwardMessage{Float64}
end
