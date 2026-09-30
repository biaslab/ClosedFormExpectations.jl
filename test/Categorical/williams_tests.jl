@testitem "Categorical EF Williams product Monte Carlo" begin
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, sufficientstatistics, gradlogpartition
    include("../test_utils.jl")

    # Categorical's sufficient statistics are a tuple containing one one-hot vector.
    score(q::ExponentialFamilyDistribution{<:Categorical}, k) = only(sufficientstatistics(q, k)) .- gradlogpartition(q)

    rng = StableRNG(123)
    for _ in 1:10
        K = rand(rng, 2:5)
        weights = rand(rng, K) .+ 0.1
        ef = convert(ExponentialFamilyDistribution, Categorical(weights ./ sum(weights)))
        scores = randn(rng, K)
        target_weights = rand(rng, K) .+ 0.1
        target = Categorical(target_weights ./ sum(target_weights))
        normal = Normal(randn(rng), rand(rng) + 0.5)
        for f in (k -> scores[k], Logpdf(target), Logpdf(normal))
            central_limit_theorem_test(ClosedWilliamsProduct(), f, ef, score)
        end
    end
end

@testitem "Categorical Williams product in natural coordinates" begin
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters

    struct CategoryScore{T}
        values::T
    end
    (f::CategoryScore)(k) = f.values[k]

    strategy = ClosedWilliamsProduct()
    q = Categorical([0.2, 0.3, 0.5])
    ef = convert(ExponentialFamilyDistribution, q)
    scores = [1, 2, 4]
    gradient = mean(strategy, k -> scores[k], ef)
    @test gradient ≈ [-0.36, -0.24, 0.6]
    @test length(gradient) == 3
    @test gradient[end] ≈ 0.6
    @test sum(gradient) ≈ 0 atol = 1e-14
    @test mean(strategy, CategoryScore(scores), ef) ≈ gradient
    @test mean(strategy, _ -> 7.0, ef) ≈ zeros(3) atol = 1e-14
    @test mean(strategy, k -> scores[k] + 5, ef) ≈ gradient
    @test_throws MethodError mean(strategy, identity, q)

    @testset "Finite differences in reference logits" begin
        for p in ([0.2, 0.3, 0.5], [0.7, 0.3], [0.1, 0.2, 0.3, 0.4])
            ef = convert(ExponentialFamilyDistribution, Categorical(p))
            η = getnaturalparameters(ef)
            f = k -> k^2 - 2k + 0.3
            g = mean(strategy, f, ef)
            h = 1e-5
            for k in 1:(length(p) - 1)
                plus, minus = copy(η), copy(η)
                plus[k] += h
                minus[k] -= h
                μ_plus = mean(ClosedFormExpectation(), f, ExponentialFamilyDistribution(Categorical, plus, length(p)))
                μ_minus = mean(ClosedFormExpectation(), f, ExponentialFamilyDistribution(Categorical, minus, length(p)))
                @test isapprox(g[k], (μ_plus - μ_minus) / (2h); atol = 1e-9, rtol = 1e-7)
            end
        end
    end

    @testset "Numeric promotion" begin
        ef32 = convert(ExponentialFamilyDistribution, Categorical(Float32[0.2, 0.3, 0.5]))
        g32 = mean(strategy, CategoryScore(Float32[1, 2, 4]), ef32)
        @test eltype(g32) == Float32
        @test g32 ≈ Float32[-0.36, -0.24, 0.6]
        @test eltype(mean(strategy, CategoryScore(BigFloat[1, 2, 4]), ef32)) == BigFloat
    end
end

@testitem "Categorical Williams product skips zero mass" begin
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters

    strategy = ClosedWilliamsProduct()
    ef = convert(ExponentialFamilyDistribution, Categorical([0.0, 0.8, 0.2]))
    calls = zeros(Int, 3)
    f = k -> begin
        k == 1 && error("Evaluated a zero-probability category")
        calls[k] += 1
        return k
    end
    η = copy(getnaturalparameters(ef))
    g = mean(strategy, f, ef)
    @test g ≈ [0.0, -0.16, 0.16]
    @test iszero(g[1])
    @test calls == [0, 1, 1]
    @test getnaturalparameters(ef) == η

    target = Categorical([0.0, 0.5, 0.5])
    @test logpdf(target, 1) == -Inf
    @test mean(strategy, Logpdf(target), ef) ≈ zeros(3) atol = 1e-14
    @test mean(strategy, target, ef) ≈ zeros(3) atol = 1e-14

    # Stable softmax also handles extreme finite logits and underflowed mass.
    extreme = ExponentialFamilyDistribution(Categorical, [1000.0, 999.0, 0.0], 3)
    extreme_score = k -> k == 3 ? error("Evaluated underflowed mass") : k
    a = inv(1 + exp(-1))
    @test mean(strategy, extreme_score, extreme) ≈ [-a * (1 - a), a * (1 - a), 0.0]
end

@testitem "Categorical Williams product target bridges and dispatch" begin
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily: ExponentialFamilyDistribution, ProductOf
    using Enzyme
    using Test

    strategy = ClosedWilliamsProduct()
    p = [0.2, 0.3, 0.5]
    ef = convert(ExponentialFamilyDistribution, Categorical(p))
    targets = (Categorical([0.4, 0.1, 0.5]), Normal(0, 1), Poisson(2))
    for target in targets
        # Independently enumerate f(k) times the score vector e_k - p.
        expected = sum(p[k] * logpdf(target, k) .* ([j == k for j in 1:3] .- p) for k in 1:3)
        @test mean(strategy, target, ef) ≈ expected
        @test mean(strategy, Logpdf(target), ef) ≈ expected
        @test mean(strategy, Base.Fix1(logpdf, target), ef) ≈ expected
    end

    product = Logpdf(ProductOf(targets[1], ProductOf(targets[2], targets[3])))
    @test mean(strategy, product, ef) ≈ sum(mean(strategy, d, ef) for d in targets)

    # The manual specialization must remain disjoint from the Enzyme backend.
    categorical_signature = Tuple{typeof(ClosedFormExpectations.mean_ef_impl), ClosedWilliamsProduct, Any, ExponentialFamilyDistribution{<:Categorical}}
    ambiguities = Test.detect_ambiguities(ClosedFormExpectations, Base.get_extension(ClosedFormExpectations, :ClosedFormExpectationsEnzymeExt); recursive = false)
    @test isempty(filter(ambiguities) do pair
        any(method -> method.sig <: categorical_signature, pair)
    end)
end
