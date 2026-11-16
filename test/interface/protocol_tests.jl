@testitem "The expectation protocol: every pair of the table has a closed form" begin
    using BayesBase, Distributions, ExponentialFamily, LogExpFunctions
    import ClosedFormExpectations: has_closed_form, Product, Square, Power, Abs, xlog2x

    cfe = ClosedFormExpectation()
    gamma = Gamma(2.0, 1.5)
    normal = NormalMeanVariance(0.3, 0.8)
    lognormal = LogNormal(0.2, 0.6)
    exponential = Exponential(2.0)
    mvnormal = MvNormalMeanCovariance([0.1, -0.2], [1.0 0.3; 0.3 2.0])

    pairs = [
        (Base.Fix1(logpdf, Exponential(1.5)), exponential),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), exponential),
        (log, exponential),
        (log ∘ ExpLogSquare(0.1, 0.9), exponential),
        (log, gamma),
        (xlogx, gamma),
        (xlog2x, gamma),
        (Square() ∘ log, gamma),
        (Power(Val(3)) ∘ log, gamma),
        (log ∘ ExpLogSquare(0.1, 0.9), gamma),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), gamma),
        (Base.Fix1(logpdf, GammaShapeRate(3.0, 2.0)), gamma),
        (Base.Fix1(logpdf, NormalMeanPrecision(1.0, 2.0)), gamma),
        (Base.Fix1(logpdf, ReLUForwardMessage(1.0, 2.0)), gamma),
        (Base.Fix1(logpdf, ReLUBackwardMessage(1.0, 2.0)), gamma),
        (log ∘ Product((ExpLogSquare(0.1, 0.9), ExpLogSquare(1.0, 2.0))), gamma),
        (Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), normal),
        (Base.Fix1(logpdf, Laplace(0.5, 2.0)), normal),
        (Abs(), normal),
        (Base.Fix1(logpdf, LogGamma(2.0, 3.0)), normal),
        (Base.Fix1(logpdf, ReLUBackwardMessage(1.0, 2.0)), normal),
        (Base.Fix1(logpdf, GammaShapeRate(3.0, 2.0)), lognormal),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), lognormal),
        (Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), lognormal),
        (Base.Fix1(logpdf, ReLUForwardMessage(1.0, 2.0)), lognormal),
        (Base.Fix1(logpdf, ReLUBackwardMessage(1.0, 2.0)), lognormal),
        (Base.Fix1(logpdf, MvNormalMeanCovariance([0.0, 1.0], [2.0 0.0; 0.0 1.0])), mvnormal),
        (Base.Fix1(logpdf, LinearLogGamma(2.0, 3.0, [0.5, -1.0])), mvnormal),
        (Base.Fix1(logpdf, prod(GenericProd(), LogNormal(0.1, 0.9), GammaShapeRate(3.0, 2.0))), gamma),
        (x -> x^2, Categorical([0.2, 0.3, 0.5]))
    ]

    for (f, q) in pairs
        @testset let f = f, q = q
            @test has_closed_form(cfe, f, q)
            @test BayesBase.supports_expectation(cfe, f, q)
            @test BayesBase.is_expectation_exact(cfe, f, q)
            @test isfinite(BayesBase.expectation(cfe, f, q))
        end
    end
end

@testitem "The expectation protocol: an integrand as a Logpdf, as Fix1 or as the distribution" begin
    using BayesBase, Distributions, ExponentialFamily

    cfe = ClosedFormExpectation()
    p = LogNormal(0.3, 0.7)
    q = Gamma(2.0, 1.5)
    expected = mean(cfe, Logpdf(p), q)

    for f in (Logpdf(p), Base.Fix1(logpdf, p), p)
        @test BayesBase.expectation(cfe, f, q) == expected
    end

    # A `GammaShapeRate` and the exponential family form of `q` are converted to a `Gamma`
    @test BayesBase.expectation(cfe, Base.Fix1(logpdf, p), GammaShapeRate(2.0, 1 / 1.5)) ≈ expected
    @test BayesBase.expectation(cfe, Base.Fix1(logpdf, p), convert(ExponentialFamilyDistribution, q)) ≈ expected
end

@testitem "The expectation protocol: pairs without a closed form" begin
    using BayesBase, Distributions, ExponentialFamily
    import ClosedFormExpectations: has_closed_form

    cfe = ClosedFormExpectation()

    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, Beta(2.0, 2.0)), Normal(0.0, 1.0))
    @test !BayesBase.supports_expectation(cfe, sin, Gamma(2.0, 1.0))
    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, LogGamma(1.0, 2.0)), Gamma(2.0, 1.0))
    @test !BayesBase.is_expectation_exact(cfe, sin, Gamma(2.0, 1.0))
    @test_throws BayesBase.UnsupportedExpectationError BayesBase.expectation(cfe, sin, Gamma(2.0, 1.0))

    # Every integrand under a `Categorical` has a closed form, but the backend does not claim
    # `F = Any`: numerical backends with points serve products better
    @test has_closed_form(cfe, Any, Categorical{Float64, Vector{Float64}})
    @test !BayesBase.supports_expectation(cfe, Any, Categorical{Float64, Vector{Float64}})
    @test BayesBase.supports_expectation(cfe, typeof(sin), Categorical{Float64, Vector{Float64}})
end

@testitem "The expectation protocol: the backend is registered and routed to" begin
    using BayesBase, Distributions

    cfe = ClosedFormExpectation()
    f = Base.Fix1(logpdf, LogNormal(0.3, 0.7))
    q = Gamma(2.0, 1.5)

    @test BayesBase.expectation_backend_priority(cfe) == 0
    @test BayesBase.has_expectation_label(cfe, BayesBase.ExpectationExact())
    @test !BayesBase.is_expectation_stochastic(cfe)
    @test BayesBase.which_expectation_backend(BayesBase.ExpectationFirstOf(cfe), f, q) === cfe
    @test BayesBase.expectation(BayesBase.ExpectationFirstOf(cfe), f, q) == BayesBase.expectation(cfe, f, q)
    @test BayesBase.expectation(BayesBase.ExpectationDynamicRouter(), f, q) == BayesBase.expectation(cfe, f, q)
end

@testitem "The Williams product: every pair of the table has a closed form" begin
    using BayesBase, Distributions, ExponentialFamily
    import ClosedFormExpectations: has_closed_form, Square, Abs

    williams = ClosedWilliamsProduct()
    gamma = Gamma(2.0, 1.5)
    exponential = Exponential(2.0)
    normal = Normal(0.3, 0.8)

    pairs = [
        (log, exponential),
        (log ∘ ExpLogSquare(0.1, 0.9), exponential),
        (Logpdf(LogNormal(0.1, 0.9)), exponential),
        (Logpdf(Exponential(1.5)), exponential),
        (log, gamma),
        (Square() ∘ log, gamma),
        (log ∘ ExpLogSquare(0.1, 0.9), gamma),
        (Logpdf(LogNormal(0.1, 0.9)), gamma),
        (Logpdf(GammaShapeRate(3.0, 2.0)), gamma),
        (Logpdf(NormalMeanVariance(1.0, 2.0)), gamma),
        (Abs(), normal),
        (Logpdf(NormalMeanVariance(1.0, 2.0)), normal),
        (Logpdf(Laplace(0.5, 2.0)), normal),
        (Logpdf(LogGamma(2.0, 3.0)), normal),
        (Logpdf(LogGamma(2.0, 3.0)), NormalMeanVariance(0.3, 0.64)),
        (Logpdf(Laplace(0.5, 2.0)), NormalMeanVariance(0.3, 0.64)),
        (Logpdf(LogGamma(2.0, 3.0)), convert(ExponentialFamilyDistribution, NormalMeanVariance(0.3, 0.64))),
        (Logpdf(LogNormal(0.1, 0.9)), convert(ExponentialFamilyDistribution, gamma)),
        (Logpdf(prod(GenericProd(), LogNormal(0.1, 0.9), GammaShapeRate(3.0, 2.0))), convert(ExponentialFamilyDistribution, gamma)),
        (x -> x^2, convert(ExponentialFamilyDistribution, Categorical([0.2, 0.3, 0.5])))
    ]

    for (f, q) in pairs
        @testset let f = f, q = q
            @test has_closed_form(williams, f, q)
            @test all(isfinite, mean(williams, f, q))
        end
    end

    @test !has_closed_form(williams, Logpdf(Beta(2.0, 2.0)), normal)
    @test !has_closed_form(williams, Logpdf(Laplace(0.5f0, 2.0f0)), normal)
    @test !has_closed_form(williams, Logpdf(ReLUBackwardMessage(1.0, 2.0)), gamma)
end

@testitem "The Williams product of Base.Fix1(logpdf, p)" begin
    using Distributions, ExponentialFamily

    williams = ClosedWilliamsProduct()
    p = LogNormal(0.3, 0.7)
    q = Gamma(2.0, 1.5)

    @test mean(williams, Base.Fix1(logpdf, p), q) == mean(williams, Logpdf(p), q)

    ef = convert(ExponentialFamilyDistribution, q)
    @test mean(williams, Base.Fix1(logpdf, p), ef) == mean(williams, Logpdf(p), ef)

    g = LogGamma(2.0, 3.0)
    @test mean(williams, Base.Fix1(logpdf, g), NormalMeanVariance(0.3, 0.64)) ==
          mean(williams, Logpdf(g), NormalMeanVariance(0.3, 0.64))
end
