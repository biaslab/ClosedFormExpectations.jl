import Distributions: Categorical
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters
import LogExpFunctions: softmax

# Return the full score gradient in softmax logits, including the reference
# category. Coordinate reduction and the inverse Fisher belong to the caller.
function mean_ef_impl(::ClosedWilliamsProduct{Nothing}, f, q::ExponentialFamilyDistribution{<:Categorical})
    p = softmax(getnaturalparameters(q))
    values = [iszero(p[k]) ? zero(eltype(p)) : f(k) for k in eachindex(p)]
    μ = sum(p .* values)
    return [iszero(p[k]) ? zero(μ) : p[k] * (values[k] - μ) for k in eachindex(p)]
end
