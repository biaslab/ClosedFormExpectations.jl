@testitem "The expectation protocol: every pair of the table has a closed form" begin
    import BayesBase
    using Distributions
    using Distributions, ExponentialFamily, LogExpFunctions
    import ClosedFormExpectations: Product, Square, Power, Abs, xlog2x

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
        (Base.Fix1(logpdf, prod(BayesBase.GenericProd(), LogNormal(0.1, 0.9), GammaShapeRate(3.0, 2.0))), gamma),
        (x -> x^2, Categorical([0.2, 0.3, 0.5]))
    ]

    for (f, q) in pairs
        @testset let f = f, q = q
            @test BayesBase.supports_expectation(cfe, f, q)
            @test BayesBase.is_expectation_exact(cfe, f, q)
            @test isfinite(BayesBase.expectation(cfe, f, q))
        end
    end
end

@testitem "The expectation protocol: distributions converted to the closed forms" begin
    import BayesBase
    using Distributions
    using Distributions, ExponentialFamily

    cfe = ClosedFormExpectation()
    p = LogNormal(0.3, 0.7)
    q = Gamma(2.0, 1.5)
    expected = BayesBase.expectation(cfe, Base.Fix1(logpdf, p), q)

    # A `GammaShapeRate` and the exponential family form of `q` are converted to a `Gamma`
    @test BayesBase.expectation(cfe, Base.Fix1(logpdf, p), GammaShapeRate(2.0, 1 / 1.5)) ≈ expected
    @test BayesBase.expectation(cfe, Base.Fix1(logpdf, p), convert(ExponentialFamilyDistribution, q)) ≈ expected
end

@testitem "The expectation protocol: pairs without a closed form" begin
    import BayesBase
    using Distributions
    using Distributions, ExponentialFamily

    cfe = ClosedFormExpectation()

    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, Beta(2.0, 2.0)), Normal(0.0, 1.0))
    @test !BayesBase.supports_expectation(cfe, sin, Gamma(2.0, 1.0))
    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, LogGamma(1.0, 2.0)), Gamma(2.0, 1.0))
    @test !BayesBase.is_expectation_exact(cfe, sin, Gamma(2.0, 1.0))
    @test_throws BayesBase.UnsupportedExpectationError BayesBase.expectation(cfe, sin, Gamma(2.0, 1.0))

    # Every integrand under a `Categorical` has a closed form, but the backend does not claim
    # `F = Any`: numerical backends with points serve products better
    @test !BayesBase.supports_expectation(cfe, Any, Categorical{Float64, Vector{Float64}})
    @test BayesBase.supports_expectation(cfe, typeof(sin), Categorical{Float64, Vector{Float64}})
end

@testitem "The expectation protocol: the backend is registered and routed to" begin
    import BayesBase
    using Distributions
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

@testitem "The gradient: every pair of the table has a closed form" begin
    import BayesBase
    using Distributions
    using BayesBase, Distributions, ExponentialFamily
    import ClosedFormExpectations: Square, Abs

    williams = ClosedFormExpectation()
    gamma = Gamma(2.0, 1.5)
    exponential = Exponential(2.0)
    normal = Normal(0.3, 0.8)

    pairs = [
        (log, exponential),
        (log ∘ ExpLogSquare(0.1, 0.9), exponential),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), exponential),
        (Base.Fix1(logpdf, Exponential(1.5)), exponential),
        (log, gamma),
        (Square() ∘ log, gamma),
        (log ∘ ExpLogSquare(0.1, 0.9), gamma),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), gamma),
        (Base.Fix1(logpdf, GammaShapeRate(3.0, 2.0)), gamma),
        (Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), gamma),
        (Abs(), normal),
        (Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), normal),
        (Base.Fix1(logpdf, Laplace(0.5, 2.0)), normal),
        (Base.Fix1(logpdf, LogGamma(2.0, 3.0)), normal),
        (Base.Fix1(logpdf, LogGamma(2.0, 3.0)), NormalMeanVariance(0.3, 0.64)),
        (Base.Fix1(logpdf, Laplace(0.5, 2.0)), NormalMeanVariance(0.3, 0.64)),
        (Base.Fix1(logpdf, LogGamma(2.0, 3.0)), convert(ExponentialFamilyDistribution, NormalMeanVariance(0.3, 0.64))),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), convert(ExponentialFamilyDistribution, gamma)),
        (Base.Fix1(logpdf, prod(BayesBase.GenericProd(), LogNormal(0.1, 0.9), GammaShapeRate(3.0, 2.0))), convert(ExponentialFamilyDistribution, gamma)),
        (x -> x^2, convert(ExponentialFamilyDistribution, Categorical([0.2, 0.3, 0.5])))
    ]

    for (f, q) in pairs
        @testset let f = f, q = q
            @test BayesBase.supports_expectation_gradient(williams, f, q)
            @test BayesBase.is_expectation_exact(williams, BayesBase.ExpectationScoreProduct(f, q), q)
            @test all(isfinite, BayesBase.expectation_gradient(williams, f, q))
            @test BayesBase.expectation_gradient(BayesBase.ExpectationFirstOf(williams), f, q) == BayesBase.expectation_gradient(williams, f, q)
        end
    end

    @test !BayesBase.supports_expectation_gradient(williams, Base.Fix1(logpdf, Beta(2.0, 2.0)), normal)
    @test !BayesBase.supports_expectation_gradient(williams, Base.Fix1(logpdf, Laplace(0.5f0, 2.0f0)), normal)
    @test !BayesBase.supports_expectation_gradient(williams, Base.Fix1(logpdf, ReLUBackwardMessage(1.0, 2.0)), gamma)
    # A plain `Categorical` has no score, and a `GammaShapeRate` no gradient formula
    @test !BayesBase.supports_expectation_gradient(williams, identity, Categorical([0.5, 0.5]))
    @test !BayesBase.supports_expectation_gradient(williams, log, GammaShapeRate(2.0, 1.0))
    @test_throws BayesBase.UnsupportedExpectationError BayesBase.expectation_gradient(williams, sin, gamma)
end

@testitem "Every closed form declares its support" begin
    import BayesBase
    using Distributions
    import BayesBase: supports_expectation, ExpectationScoreProduct

    # The generic method of the package, for the pairs that follow from others by a rule
    generic = which(supports_expectation, Tuple{ClosedFormExpectation, Type{Int}, Type{Int}})
    declared = filter(m -> m.module === ClosedFormExpectations && m !== generic, collect(methods(supports_expectation)))

    function declaration(m, integrand)
        B, F, Q = Base.unwrap_unionall(m.sig).parameters[2:4]
        return Base.rewrap_unionall(Tuple{typeof(supports_expectation), B, Type{<:integrand(F, Q)}, Type{<:Q}}, m.sig)
    end
    is_declared(m, integrand) = any(d -> declaration(m, integrand) <: d.sig, declared)

    # Methods with a type variable as their integrand are rules, not closed forms
    closed_forms(f) = filter(collect(methods(f))) do m
        F = Base.unwrap_unionall(m.sig).parameters[3]
        m.module === ClosedFormExpectations && !(F isa TypeVar)
    end

    expectations = closed_forms(BayesBase.expectation)
    gradients = closed_forms(BayesBase.expectation_gradient)
    @test length(expectations) > 40
    @test length(gradients) > 10
    for m in expectations
        @test is_declared(m, (F, Q) -> F)
    end
    for m in gradients
        @test is_declared(m, (F, Q) -> ExpectationScoreProduct{<:F, <:Q})
    end
end

@testitem "The gradient agrees with the expectation of f times the score" begin
    import BayesBase
    using Distributions
    using Distributions, ExponentialFamily
    import ClosedFormExpectations: Square, Abs, LogGamma

    # A fine midpoint rule in the quantiles of `q`, for any integrand
    struct Quantiles end
    BayesBase.supports_expectation(::Quantiles, ::Type, ::Type{<:Union{Normal, Gamma, Exponential}}) = true
    function BayesBase.prepare_expectation(::Quantiles, q)
        n = 200_000
        points = quantile.(q, ((1:n) .- 0.5) ./ n)
        return BayesBase.ExpectationPointSet(points, fill(1 / n, n))
    end

    cfe = ClosedFormExpectation()
    gamma, normal, exponential = Gamma(3.5, 1.5), Normal(0.3, 0.8), Exponential(2.0)
    pairs = [
        (log, gamma),
        (Square() ∘ log, gamma),
        (Base.Fix1(logpdf, LogNormal(0.1, 0.9)), gamma),
        (Base.Fix1(logpdf, Gamma(3.0, 0.5)), gamma),
        (Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), gamma),
        (Abs(), normal),
        (Base.Fix1(logpdf, Normal(1.0, 2.0)), normal),
        (Base.Fix1(logpdf, Laplace(0.5, 2.0)), normal),
        (Base.Fix1(logpdf, LogGamma(2.0, 3.0)), normal),
        (log, exponential),
        (Base.Fix1(logpdf, Exponential(1.5)), exponential),
    ]
    for (f, q) in pairs
        @testset let f = f, q = q
            @test BayesBase.expectation_gradient(cfe, f, q) ≈ BayesBase.expectation_gradient(Quantiles(), f, q) rtol = 1.0e-3
        end
    end

    # A router chooses the closed form, and the numerical backend for the rest
    router = BayesBase.ExpectationFirstOf(cfe, Quantiles())
    @test BayesBase.expectation_gradient(router, log, gamma) == BayesBase.expectation_gradient(cfe, log, gamma)
    @test BayesBase.expectation_gradient(router, sin, normal) ≈ BayesBase.expectation_gradient(Quantiles(), sin, normal)
end
