@testitem "mean(::ClosedWilliamsProduct, p::Logpdf{Normal}, q::Normal)" begin
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        μ2, σ2 = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(Normal(μ1, σ1)), Normal(μ2, σ2), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, p::Logpdf{Laplace}, q::Normal)" begin
    include("normal_utils.jl")
    using FiniteDifferences
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ = rand(rng)*10, 1 + rand(rng)
        μ_2, θ = μ_1, 1 + rand(rng)
        func_var(x) = mean(ClosedFormExpectation(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, x))
        williams_result = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, σ))
        @test central_fdm(5, 1)(func_var, σ) ≈ williams_result[2]
        # the derivative over the mean is numerically unstable, the test is made against the exact value
        # @test central_fdm(5, 1)(x -> mean(ClosedFormExpectation(), Logpdf(Laplace(μ_2, θ)), Normal(x, σ)), μ_1) ≈ williams_result[1]
        @test -1/(2*θ) ≈ williams_result[1]
    end
end
