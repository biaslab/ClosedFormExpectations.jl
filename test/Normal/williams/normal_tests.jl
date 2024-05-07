@testitem "mean(::ClosedWilliamsProduct, p::Abs, q::Normal)" begin
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Abs(), Normal(μ, σ), score, 10^5)
    end
end

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
    
    rng = StableRNG(123)

    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*5
        loc, θ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(Laplace(loc, θ)), Normal(μ, σ), score)
    end

    @testset "compare Logpdf(Laplace) gradient with Abs gradient" begin
        for _ in 1:10
            μ, σ = rand(rng)*10, rand(rng)*5
            θ = rand(rng)*10
            williams_result_abs = -1/θ*mean(ClosedWilliamsProduct(), Abs(), Normal(μ, σ))
            williams_result_laplace = mean(ClosedWilliamsProduct(), Logpdf(Laplace(0, θ)), Normal(μ, σ))
            @test williams_result_abs ≈ williams_result_laplace
        end
    end   
end
