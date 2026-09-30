import Distributions: Categorical, probs
import ExponentialFamily: ProductOf

"""
    mean(::ClosedFormExpectation, f, q::Categorical)

Compute the exact expectation `sum(probs(q)[k] * f(k))` over the categories of `q`.
Categories with zero probability are skipped without evaluating `f(k)`.
"""
function mean(::ClosedFormExpectation, f, q::Categorical)
    p = probs(q)
    return sum(p[k] * f(k) for k in eachindex(p) if !iszero(p[k]))
end

# Resolve intersections with the generic target wrappers and decompositions.
function mean(expectation::ClosedFormExpectation, d::Distribution, q::Categorical)
    return mean(expectation, Logpdf(d), q)
end

function mean(expectation::ClosedFormExpectation, f::Base.Fix1{typeof(logpdf), D}, q::Categorical) where {D}
    return mean(expectation, Logpdf(f.x), q)
end

function mean(expectation::ClosedFormExpectation, p::Logpdf{<:ProductOf}, q::Categorical)
    return mean(expectation, Logpdf(p.dist.left), q) + mean(expectation, Logpdf(p.dist.right), q)
end

function mean(expectation::ClosedFormExpectation, p::ComposedFunction{typeof(log), Product{T}}, q::Categorical) where {T}
    return sum(mean(expectation, log ∘ p_i, q) for p_i in p.inner.multipliers)
end
