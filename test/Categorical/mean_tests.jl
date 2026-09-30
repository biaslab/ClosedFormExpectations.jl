@testitem "Categorical exact expectation" begin
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

    @test mean(strategy, k -> scores[k], q) ≈ 2.8
    @test mean(strategy, CategoryScore(scores), q) ≈ 2.8
    @test mean(strategy, _ -> 7, q) ≈ 7
    @test mean(strategy, identity, Categorical([1.0])) == 1
    @test mean(strategy, CategoryScore(scores), convert(ExponentialFamilyDistribution, q)) ≈ 2.8

    @testset "Skip zero mass and evaluate each included score once" begin
        sparse_q = Categorical([0.0, 0.25, 0.0, 0.75, 0.0])
        calls = zeros(Int, 5)
        f = k -> begin
            k in (2, 4) || error("Evaluated a zero-probability category")
            calls[k] += 1
            return k
        end
        @test mean(strategy, f, sparse_q) == 3.5
        @test calls == [0, 1, 0, 1, 0]
        @test probs(sparse_q) == [0.0, 0.25, 0.0, 0.75, 0.0]
    end

    @testset "Numeric promotion" begin
        q32 = Categorical(Float32[0.2, 0.3, 0.5])
        result32 = mean(strategy, CategoryScore(Float32[1, 2, 4]), q32)
        @test result32 isa Float32
        @test result32 ≈ 2.8f0
        @test mean(strategy, CategoryScore(BigFloat[1, 2, 4]), q32) isa BigFloat
    end
end

@testitem "Categorical expectation target bridges" begin
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, ProductOf

    strategy = ClosedFormExpectation()
    q = Categorical([0.2, 0.3, 0.5])
    ef = convert(ExponentialFamilyDistribution, q)

    @testset "Distribution and Logpdf targets" begin
        for target in (Categorical([0.4, 0.1, 0.5]), Normal(0, 1), Poisson(2))
            expected = sum(probs(q)[k] * logpdf(target, k) for k in 1:3)
            @test mean(strategy, Logpdf(target), q) ≈ expected
            @test mean(strategy, target, q) ≈ expected
            @test mean(strategy, Base.Fix1(logpdf, target), q) ≈ expected
            @test mean(strategy, Logpdf(target), ef) ≈ expected
            @test mean(strategy, target, ef) ≈ expected
        end
    end

    @testset "Zero mass with infinite log densities" begin
        sparse_q = Categorical([0.0, 0.4, 0.6])
        target = Categorical([0.0, 0.25, 0.75])
        expected = 0.4 * log(0.25) + 0.6 * log(0.75)
        @test logpdf(target, 1) == -Inf
        @test mean(strategy, Logpdf(target), sparse_q) ≈ expected
        @test mean(strategy, target, sparse_q) ≈ expected
        @test mean(strategy, Base.Fix1(logpdf, target), sparse_q) ≈ expected
    end

    @testset "Recursive product targets" begin
        targets = (Normal(0, 1), Poisson(2), Categorical([0.4, 0.1, 0.5]))
        product = Logpdf(ProductOf(targets[1], ProductOf(targets[2], targets[3])))
        expected = sum(probs(q)[k] * sum(logpdf(d, k) for d in targets) for k in 1:3)
        @test mean(strategy, product, q) ≈ expected
        @test mean(strategy, product, ef) ≈ expected

        expression = log ∘ ClosedFormExpectations.Product((identity, x -> x + 1))
        expected_expression = sum(probs(q)[k] * (log(k) + log(k + 1)) for k in 1:3)
        @test mean(strategy, expression, q) ≈ expected_expression
    end
end

@testitem "Categorical expectation dispatch ambiguities" begin
    using ClosedFormExpectations
    using Distributions
    using Test

    # Other distribution interfaces have pre-existing ambiguities. Require every
    # method in the new Categorical interface to be unambiguous.
    categorical_signature = Tuple{typeof(mean), ClosedFormExpectation, Any, Categorical}
    ambiguities = Test.detect_ambiguities(ClosedFormExpectations; recursive = false)
    @test isempty(filter(ambiguities) do pair
        any(method -> method.sig <: categorical_signature, pair)
    end)
end
