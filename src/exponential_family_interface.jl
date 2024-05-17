import Base: convert
import ExponentialFamily: ExponentialFamilyDistribution

function mean(expectation::ClosedFormExpectation, f, q::ExponentialFamilyDistribution)
    dist = Base.convert(Distribution, q)
    return mean(expectation, f, dist)
end