import ExponentialFamily: ExponentialFamilyDistribution, ProductOf
import Distributions: Distribution

function mean(expectation::ClosedWilliamsProduct, p::Logpdf{<:ProductOf}, q)
    # E[ (∑ log p_i) * ∇ log q ] = ∑ E[ log p_i * ∇ log q ]
    left = p.dist.left
    right = p.dist.right
    return mean(expectation, Logpdf(left), q) .+ mean(expectation, Logpdf(right), q)
end

function mean(expectation::ClosedFormExpectation, p::Logpdf{<:ProductOf}, q)
    # E[ ∑ log p_i ] = ∑ E[ log p_i ]
    left = p.dist.left
    right = p.dist.right
    return mean(expectation, Logpdf(left), q) + mean(expectation, Logpdf(right), q)
end
