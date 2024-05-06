@testitem "mean(::ClosedWilliamsProduct, p::Logpdf{Normal}, q::NormalMeanVariance)" begin
    include("normal_mean_variance_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ1, σ1 = rand(rng)*10, rand(rng)*5
        μ2, σ2 = rand(rng)*10, rand(rng)*5
        central_limit_theorem_test(ClosedWilliamsProduct(), Logpdf(Normal(μ1, σ1)), NormalMeanVariance(μ2, σ2^2), score)
    end
end

@testitem "mean(::ClosedWilliamsProduct, p, q::NormalMeanVariance)" begin
    include("normal_mean_variance_utils.jl")

    @testset "equal to Normal under jacobian" begin
        rng = StableRNG(123)        
        μ, σ = rand(rng)*10, rand(rng)*10
        loc, θ = rand(rng)*10, rand(rng)*10 
            
        laplace = Laplace(loc, θ)
        normal = Normal(μ, σ)
        nmv = NormalMeanVariance(μ, σ^2)
        
        williams_result = mean(ClosedWilliamsProduct(), Logpdf(laplace), nmv)
        williams_result_mv = mean(ClosedWilliamsProduct(), Logpdf(laplace), normal)

        jacobian = [1 0; 0 0.5/std(normal)]

        @test williams_result ≈ jacobian * williams_result_mv
    end
end