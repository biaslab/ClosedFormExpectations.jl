import Distributions: Categorical, probs

# The exact sum `Σₖ probs(q)[k] f(k)` over the categories of `q`, for any integrand. Categories with
# zero probability are skipped without evaluating `f(k)`. It is reached through the protocol's
# generic method in `protocol.jl`, which does not claim `F = Any`.
function categorical_expectation(f, q::Categorical)
    p = probs(q)
    return sum(p[k] * f(k) for k in eachindex(p) if !iszero(p[k]))
end
