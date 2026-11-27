@testsetup module SupportCases
    export cases, mvnormal, finite
    using ClosedFormExpectations
    import BayesBase
    using Distributions, ExponentialFamily, LogExpFunctions
    import ClosedFormExpectations: Product, Square, Power, Abs, xlog2x

    # Integrands of every kind the package has a closed form for, and some it has none for
    integrands = Any[
        log, log ∘ identity, xlogx, xlog2x, sin, Abs(),
        Square() ∘ log, Power(Val(3)) ∘ log, log ∘ ExpLogSquare(0.1, 0.9),
        log ∘ Product((ExpLogSquare(0.1, 0.9), ExpLogSquare(1.0, 2.0))),
        LogGammaKernel(2.0, 3.0), LogGammaKernel(2.0, 0.0), LogGammaKernel(2.0, -0.5),
        Base.Fix1(logpdf, Exponential(1.5)),
        Base.Fix1(logpdf, LogNormal(0.1, 0.9)),
        Base.Fix1(logpdf, Gamma(3.0, 0.5)),
        Base.Fix1(logpdf, GammaShapeRate(3.0, 2.0)),
        Base.Fix1(logpdf, Normal(1.0, 2.0)),
        Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)),
        Base.Fix1(logpdf, NormalMeanPrecision(1.0, 2.0)),
        Base.Fix1(logpdf, Laplace(0.5, 2.0)),
        Base.Fix1(logpdf, Beta(2.0, 2.0)),
        Base.Fix1(logpdf, LogGamma(2.0, 3.0)),
        Base.Fix1(logpdf, ReLUForwardMessage(1.0, 2.0)),
        Base.Fix1(logpdf, ReLUBackwardMessage(1.0, 2.0)),
        Base.Fix1(logpdf, MvNormalMeanCovariance([0.0, 1.0], [2.0 0.0; 0.0 1.0])),
        Base.Fix1(logpdf, LinearLogGamma(2.0, 3.0, [0.5, -1.0])),
        Base.Fix1(logpdf, prod(BayesBase.GenericProd(), LogNormal(0.1, 0.9), GammaShapeRate(3.0, 2.0))),
    ]

    mvnormal = MvNormalMeanCovariance([0.1, -0.2], [1.0 0.3; 0.3 2.0])
    distributions = Any[
        Gamma(2.0, 1.5), GammaShapeRate(2.0, 0.5), Exponential(2.0), LogNormal(0.2, 0.6),
        Normal(0.3, 0.8), NormalMeanVariance(0.3, 0.64), NormalMeanPrecision(0.3, 1.5),
        NormalWeightedMeanPrecision(0.3, 1.5), Beta(2.0, 3.0), mvnormal, MvNormal([0.1, -0.2], [1.0 0.3; 0.3 2.0]),
    ]
    # The exponential family forms of these
    distributions = vcat(distributions, Any[
        convert(ExponentialFamilyDistribution, Gamma(2.0, 1.5)),
        convert(ExponentialFamilyDistribution, NormalMeanVariance(0.3, 0.64)),
        convert(ExponentialFamilyDistribution, LogNormal(0.2, 0.6)),
        convert(ExponentialFamilyDistribution, Exponential(2.0)),
        convert(ExponentialFamilyDistribution, Beta(2.0, 3.0)),
        convert(ExponentialFamilyDistribution, mvnormal),
    ])

    # Under a `Categorical`, any integrand of its categories
    categorical_integrands = Any[log, sin, x -> x^2, Abs(), LogGammaKernel(2.0, -0.5), Base.Fix1(logpdf, Normal(1.0, 2.0))]
    categoricals = Any[Categorical([0.2, 0.3, 0.5]), convert(ExponentialFamilyDistribution, Categorical([0.2, 0.3, 0.5]))]

    cases = vcat(
        vec([(f, q) for f in integrands, q in distributions]),
        vec([(f, q) for f in categorical_integrands, q in categoricals]),
    )

    finite(x) = all(isfinite, x)
end

@testitem "Support implies success: every pair declared supported is computed" setup = [SupportCases] begin
    import BayesBase
    using Distributions, ExponentialFamily

    cfe = ClosedFormExpectation()
    supported = 0
    for (f, q) in cases
        if BayesBase.supports_expectation(cfe, f, q)
            supported += 1
            @testset let f = f, q = q
                @test finite(BayesBase.expectation(cfe, f, q))
            end
        end
        if BayesBase.supports_expectation_gradient(cfe, f, q)
            supported += 1
            @testset let f = f, q = q
                @test finite(BayesBase.expectation_gradient(cfe, f, q))
            end
        end
    end
    @test supported > 100

    # Univariate closed forms are not declared under multivariate normals, and the reverse
    @test !BayesBase.supports_expectation(cfe, Abs(), mvnormal)
    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, NormalMeanVariance(1.0, 2.0)), mvnormal)
    @test !BayesBase.supports_expectation(cfe, Base.Fix1(logpdf, mvnormal), Normal(0.0, 1.0))
end

@testitem "Support implies success: every declaration is exercised" setup = [SupportCases] begin
    import BayesBase
    import BayesBase: supports_expectation, ExpectationScoreProduct

    cfe = ClosedFormExpectation()
    generic = which(supports_expectation, Tuple{ClosedFormExpectation, Type{Int}, Type{Int}})
    declared = filter(collect(methods(supports_expectation))) do m
        return m.module === ClosedFormExpectations && m !== generic &&
            Base.unwrap_unionall(m.sig).parameters[2] <: ClosedFormExpectation
    end

    reached = Set{Method}()
    for (f, q) in cases
        for F in (typeof(f), ExpectationScoreProduct{typeof(f), typeof(q)})
            signature = Tuple{ClosedFormExpectation, Type{F}, Type{typeof(q)}}
            m = which(supports_expectation, signature)
            m in declared && supports_expectation(cfe, F, typeof(q)) && push!(reached, m)
        end
    end
    # The product of distributions is declared unsupported; every other declaration has a case
    unreached = filter(m -> !(m in reached) && !occursin("FactorizedJoint", string(m.sig)), declared)
    @test isempty(unreached)
end
