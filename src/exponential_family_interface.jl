import ExponentialFamily: ExponentialFamilyDistribution

function mean(expectation::ClosedFormExpectation, f, q::ExponentialFamilyDistribution)
    dist = convert(Distribution, q)
    return mean(expectation, f, dist)
end