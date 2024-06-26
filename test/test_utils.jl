using ClosedFormExpectations
import Distributions: Normal, Laplace, Gamma, LogNormal, std, shape, scale
using StableRNGs

function sigma_rule(expectation, mean, std, N)::Bool
    return mean - 3*std/sqrt(N) < expectation < mean + 3*std/sqrt(N)
end

"""
    concentration_rule(::Real, ::Real ,::Int, ::Real)

Use concentration inequality to estimate if the statistical_mean probable with probabilty 1-α.
lip_const : Lipschitz constant of function function
sobolev_const : the constant of the logarithmic Sobolev inequality (https://en.wikipedia.org/wiki/Logarithmic_Sobolev_inequalities)
"""
function concentration_rule(expectation, statistical_mean, N, α,sobolev_const, lip_const)
    ϵ = sqrt((log(2 * α^(-1)) * sobolev_const ) / N ) * lip_const
    return statistical_mean - ϵ < expectation < statistical_mean + ϵ
end

function prepare_samples(::Any, samples)
    return samples
end

function central_limit_theorem_test(::ClosedFormExpectation, f, q, N = 10^6)
    rng = StableRNG(123)
    samples = rand(rng, q, N)
    samples = prepare_samples(q, samples)
    transformed_samples = map(f, samples)
    expectation = mean(ClosedFormExpectation(), f, q)
    result = sigma_rule(expectation, mean(transformed_samples), std(transformed_samples), N)
    if !result
        println("Test failed for functions $f and $q")
        println("expectation: ", expectation)
        println("mean: ", mean(transformed_samples))
        println("std: ", std(transformed_samples))
    end
    @test sigma_rule(expectation, mean(transformed_samples), std(transformed_samples), N)
end

function central_limit_theorem_test(::ClosedWilliamsProduct, f, q, score, N = 10^6)
    rng = StableRNG(123)
    samples = rand(rng, q, N)
    transformed_samples = map(x -> score(q, x)*f(x), samples)
    expectation = mean(ClosedWilliamsProduct(), f, q)
    for (expectation, mean, std) in zip(expectation, mean(transformed_samples), std(transformed_samples))
        @test sigma_rule(expectation, mean, std, N)
    end
end