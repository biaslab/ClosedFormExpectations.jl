@testitem "Categorical expectation Monte Carlo" begin
    import BayesBase
    using Distributions
    using ClosedFormExpectations
    using Distributions
    include("../test_utils.jl")

    rng = StableRNG(123)
    for _ in 1:10
        K = rand(rng, 2:5)
        weights = rand(rng, K) .+ 0.1
        q = Categorical(weights ./ sum(weights))
        scores = randn(rng, K)
        target_weights = rand(rng, K) .+ 0.1
        target = Categorical(target_weights ./ sum(target_weights))
        normal = Normal(randn(rng), rand(rng) + 0.5)
        for f in (k -> scores[k], Base.Fix1(logpdf, target), Base.Fix1(logpdf, normal))
            central_limit_theorem_test(ClosedFormExpectation(), f, q)
        end
    end
end

@testitem "Categorical exact expectation" begin
    import BayesBase
    using Distributions
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution

    struct CategoryScore{T}
        values::T
    end
    (f::CategoryScore)(k) = f.values[k]

    q = Categorical([0.2, 0.3, 0.5])
    scores = [1, 2, 4]
    strategy = ClosedFormExpectation()

    @test BayesBase.expectation(strategy, k -> scores[k], q) ≈ 2.8
    @test BayesBase.expectation(strategy, CategoryScore(scores), q) ≈ 2.8
    @test BayesBase.expectation(strategy, _ -> 7, q) ≈ 7
    @test BayesBase.expectation(strategy, identity, Categorical([1.0])) == 1
    @test BayesBase.expectation(strategy, CategoryScore(scores), convert(ExponentialFamilyDistribution, q)) ≈ 2.8

    @testset "Skip zero mass and evaluate each included score once" begin
        sparse_q = Categorical([0.0, 0.25, 0.0, 0.75, 0.0])
        calls = zeros(Int, 5)
        f = k -> begin
            k in (2, 4) || error("Evaluated a zero-probability category")
            calls[k] += 1
            return k
        end
        @test BayesBase.expectation(strategy, f, sparse_q) == 3.5
        @test calls == [0, 1, 0, 1, 0]
        @test probs(sparse_q) == [0.0, 0.25, 0.0, 0.75, 0.0]
    end

    @testset "Numeric promotion" begin
        q32 = Categorical(Float32[0.2, 0.3, 0.5])
        result32 = BayesBase.expectation(strategy, CategoryScore(Float32[1, 2, 4]), q32)
        @test result32 isa Float32
        @test result32 ≈ 2.8f0
        @test BayesBase.expectation(strategy, CategoryScore(BigFloat[1, 2, 4]), q32) isa BigFloat
    end
end

@testitem "Categorical expectation target bridges" begin
    import BayesBase
    using Distributions
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, ProductOf

    strategy = ClosedFormExpectation()
    q = Categorical([0.2, 0.3, 0.5])
    ef = convert(ExponentialFamilyDistribution, q)

    @testset "Log-density targets" begin
        for target in (Categorical([0.4, 0.1, 0.5]), Normal(0, 1), Poisson(2))
            expected = sum(probs(q)[k] * logpdf(target, k) for k in 1:3)
            @test BayesBase.expectation(strategy, Base.Fix1(logpdf, target), q) ≈ expected
            @test BayesBase.expectation(strategy, Base.Fix1(logpdf, target), ef) ≈ expected
        end
    end

    @testset "Zero mass with infinite log densities" begin
        sparse_q = Categorical([0.0, 0.4, 0.6])
        target = Categorical([0.0, 0.25, 0.75])
        expected = 0.4 * log(0.25) + 0.6 * log(0.75)
        @test logpdf(target, 1) == -Inf
        @test BayesBase.expectation(strategy, Base.Fix1(logpdf, target), sparse_q) ≈ expected
    end

    @testset "Recursive product targets" begin
        targets = (Normal(0, 1), Poisson(2), Categorical([0.4, 0.1, 0.5]))
        product = Base.Fix1(logpdf, ProductOf(targets[1], ProductOf(targets[2], targets[3])))
        expected = sum(probs(q)[k] * sum(logpdf(d, k) for d in targets) for k in 1:3)
        @test BayesBase.expectation(strategy, product, q) ≈ expected
        @test BayesBase.expectation(strategy, product, ef) ≈ expected

        expression = log ∘ ClosedFormExpectations.Product((identity, x -> x + 1))
        expected_expression = sum(probs(q)[k] * (log(k) + log(k + 1)) for k in 1:3)
        @test BayesBase.expectation(strategy, expression, q) ≈ expected_expression
    end
end
