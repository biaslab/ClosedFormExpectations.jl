@testitem "LogGammaKernel" begin
    import BayesBase
    using Distributions, SpecialFunctions, StableRNGs
    include("../test_utils.jl")

    f = LogGammaKernel(2.0, 3.0)
    @test f(0.5) ≈ logpdf(LogGamma(2.0, 3.0), 0.5) + 3.0 * log(2.0) + loggamma(3.0)
    @test LogGammaKernel(2, -1) isa LogGammaKernel{Float64}
    @test LogGammaKernel(2.0f0, 0) isa LogGammaKernel{Float32}
    @test Distributions.params(LogGammaKernel(2.0, -0.5)) == (2.0, -0.5)
    @test_throws DomainError LogGammaKernel(0.0, 1.0)
    @test_throws DomainError LogGammaKernel(-1.0, 1.0)

    cfe = ClosedFormExpectation()
    rng = StableRNG(42)
    for β in (-0.5, 0.0, 3.0), _ in 1:2
        f = LogGammaKernel(0.5 + 2 * rand(rng), β)
        q = Normal(randn(rng), 0.2 + rand(rng))
        central_limit_theorem_test(cfe, f, q)
        score(q, x) = [(x - mean(q)) / var(q), ((x - mean(q))^2 - var(q)) / std(q)^3]
        central_limit_theorem_test(cfe, f, q, score)
    end

    # It differs from the log-density of a `LogGamma` by a constant, so the gradients agree
    q = Normal(0.3, 0.8)
    @test BayesBase.expectation_gradient(cfe, LogGammaKernel(2.0, 3.0), q) ≈
        BayesBase.expectation_gradient(cfe, Base.Fix1(logpdf, LogGamma(2.0, 3.0)), q)
end

@testitem "Gradients through the Jacobians keep Float32" begin
    import BayesBase
    using Distributions, ExponentialFamily

    cfe = ClosedFormExpectation()
    f = Base.Fix1(logpdf, NormalMeanVariance(1.0f0, 2.0f0))
    for q in (
            NormalMeanVariance(0.3f0, 0.64f0),
            convert(ExponentialFamilyDistribution, NormalMeanVariance(0.3f0, 0.64f0)),
        )
        @test eltype(BayesBase.expectation_gradient(cfe, f, q)) === Float32
    end
    q = convert(ExponentialFamilyDistribution, Gamma(2.0f0, 1.5f0))
    @test eltype(BayesBase.expectation_gradient(cfe, log, q)) === Float32
end
