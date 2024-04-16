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
