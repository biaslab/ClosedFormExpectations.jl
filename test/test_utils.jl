function sigma_rule(expectation, mean, std, N)
    return mean - 3*std/sqrt(N) < expectation < mean + 3*std/sqrt(N)
end

"""
    concentration_rule(::Real, ::Real ,::Int, ::Real)

Use concentration inequality to estimate if the statistical_mean probable with probabilty 1-α.
"""
function concentration_rule(expectation, statistical_mean, N, α)
end