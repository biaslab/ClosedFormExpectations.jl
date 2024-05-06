@testitem "mean(::ClosedWilliamsProduct, p::Abs, q::Normal)" begin
    include("normal_utils.jl")
    rng = StableRNG(123)
    for _ in 1:10
        μ, σ = rand(rng)*10, rand(rng)*10
        central_limit_theorem_test(ClosedWilliamsProduct(), Abs(), Normal(μ, σ), score, 10^5)
    end
end