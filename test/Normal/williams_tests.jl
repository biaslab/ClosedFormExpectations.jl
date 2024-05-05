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
    using SpecialFunctions
    rng = StableRNG(123)
    for _ in 1:10
        μ_1, σ = rand(rng)*10, rand(rng)*10
        μ_2, θ = μ_1, rand(rng)*10
        func_var = (x) -> mean(ClosedFormExpectation(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, x))
        
        @testset "ClosedFormExpectation gradient is ClosedWilliamsProduct" begin
            williams_result = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, θ)), Normal(μ_1, σ))
            @test central_fdm(5, 1)(func_var, σ) ≈ williams_result[2]
            @test -1/(2*θ) ≈ williams_result[1]
        end
        
        @testset "Some comparison with known results" begin
            
            williams_result_right = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, 1/2)), Normal(μ_2 + σ, σ))
            @test -1 + sqrt(2/(MathConstants.e * π)) - erf(1/sqrt(2)) ≈ williams_result_right[1]

            williams_result_left = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, 1/2)), Normal(μ_2 - σ, σ))
            @test -1 - sqrt(2/(MathConstants.e * π)) + erf(1/sqrt(2)) ≈ williams_result_left[1]

            @test (williams_result_right[1] + williams_result_left[1]) ≈ -2

            centered_laplace = Laplace(0, θ)
            right_normal = Normal(μ_2, σ)
            left_normal  = Normal(-μ_2, σ)

            @test (mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), right_normal) + mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), left_normal))[1] ≈ -1/θ
            @test mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), right_normal)[2] ≈ mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), left_normal)[2]
        end
    end
end
