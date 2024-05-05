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
    using SpecialFunctions
    rng = StableRNG(123)

    for _ in 1:10
        @testset "μ and loc are equal" begin
            μ_1, σ = rand(rng)*10, rand(rng)*10
            θ = rand(rng)*10
            williams_result = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_1, θ)), Normal(μ_1, σ))
            @test -1/(2*θ) ≈ williams_result[1]
            @test williams_result[2] ≈ -2/(sqrt(2π)*θ)
        end

        @testset "μ = loc ± σ" begin
            μ_2, θ = rand(rng)*10, rand(rng)*10
            σ = rand(rng)*10
        
            williams_result_right = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, 1/2)), Normal(μ_2 + σ, σ))
            @test -1 + sqrt(2/(MathConstants.e * π)) - erf(1/sqrt(2)) ≈ williams_result_right[1]

            williams_result_left = mean(ClosedWilliamsProduct(), Logpdf(Laplace(μ_2, 1/2)), Normal(μ_2 - σ, σ))
            @test -1 - sqrt(2/(MathConstants.e * π)) + erf(1/sqrt(2)) ≈ williams_result_left[1]

            @test williams_result_right[2] ≈ williams_result_left[2]
        end

        @testset "μ ≠ loc, loc = 0" begin
            
            μ_1, σ = rand(rng)*10, rand(rng)*10
            θ = rand(rng)*10

            centered_laplace = Laplace(0, θ)
            right_normal = Normal(μ_1, σ)
            left_normal  = Normal(-μ_1, σ)

            williams_result_left = mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), left_normal)
            williams_result_right = mean(ClosedWilliamsProduct(), Logpdf(centered_laplace), right_normal)

            @test williams_result_left[1] + williams_result_right[1] ≈ -1/θ
            @test williams_result_left[2] ≈ williams_result_right[2]
            
            exp_part = exp(-μ_1^2/(2*σ^2))
            @test williams_result_right[2] ≈ exp_part * (-μ_1^2-2*σ^2)/(sqrt(2π)*θ*σ^2)
        end
    end
end
