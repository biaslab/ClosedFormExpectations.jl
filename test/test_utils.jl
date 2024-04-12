function sigma_rule(expectation, mean, std, N)
    return mean - 3*std/sqrt(N) < expectation < mean + 3*std/sqrt(N)
end