import Base: convert
import ExponentialFamily: ExponentialFamilyDistribution, ProductOf, NormalMeanVariance, Gamma
import Distributions: Distribution

function mean(expectation::ClosedFormExpectation, f, q::ExponentialFamilyDistribution)
    dist = Base.convert(Distribution, q)
    return mean(expectation, f, dist)
end

# -----------------------------------------------------------------------
# Ambiguity Resolutions and Generic Dispatch
# -----------------------------------------------------------------------

# We use a specialized implementation function for generic EF distributions
# to allow Specific-F methods to win over Generic-F/Specific-Q methods.
function mean_ef_impl(::ClosedWilliamsProduct, ::Any, ::Any) 
    error("mean_ef_impl not implemented for this combination")
end

# Generic entry point for ExponentialFamilyDistribution
function mean(expectation::ClosedWilliamsProduct, f, q::ExponentialFamilyDistribution)
    return mean_ef_impl(expectation, f, q)
end

# 1. Distribution vs ExponentialFamilyDistribution
#    This method is Specific-F (Distribution) and Generic-Q (EF).
#    It overrides the Generic-F entry point above.
function mean(expectation::ClosedWilliamsProduct, d::Distribution, q::ExponentialFamilyDistribution)
    return mean(expectation, Logpdf(d), q)
end

function mean(expectation::ClosedFormExpectation, d::Distribution, q::ExponentialFamilyDistribution)
    return mean(expectation, Logpdf(d), q)
end

# 2. Logpdf{ProductOf} vs ExponentialFamilyDistribution
#    This method is Specific-F (Logpdf{ProductOf}) and Generic-Q (EF).
#    It overrides the Generic-F entry point above.
function mean(expectation::ClosedWilliamsProduct, p::Logpdf{<:ProductOf}, q::ExponentialFamilyDistribution)
    return mean(expectation, Logpdf(p.dist.left), q) .+ mean(expectation, Logpdf(p.dist.right), q)
end
