@testitem "mean(::ClosedFormExpectation, p::Logpdf{MultivariateNormalDistributionsFamily}, q::MultivariateNormalDistributionsFamily)" begin
    include("mvnormal_utils.jl")

    rng = StableRNG(123)
    for _ in 1:10
        n = rand(rng, 2:10)
        μ1, Σ1 = rand(rng, n), generate_random_posdef_matrix(rng, n)
        μ2, Σ2 = rand(rng, n), generate_random_posdef_matrix(rng, n)
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(MvNormal(μ1, Σ1)), MvNormal(μ2, Σ2), 10^6)
    end
end

@testitem "mean(::ClosedFormExpectation, p::Logpdf{LinearLogGamma}, q::MultivariateNormalDistributionsFamily)" begin
    include("mvnormal_utils.jl")

    rng = StableRNG(123)
    for _ in 1:10
        n = rand(rng, 2:5)
        μ, Σ = rand(rng, n), generate_random_posdef_matrix(rng, n)
        α, β = rand(rng)*10, 1+rand(rng)
        weights = rand(rng, n)
        central_limit_theorem_test(ClosedFormExpectation(), Logpdf(LinearLogGamma(α, β, weights)), MvNormal(μ, Σ), 10^6)
    end
end
