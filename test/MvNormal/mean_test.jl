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