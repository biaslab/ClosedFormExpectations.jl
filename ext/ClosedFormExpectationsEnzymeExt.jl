module ClosedFormExpectationsEnzymeExt

using ClosedFormExpectations
import BayesBase
import BayesBase: ExpectationScoreProduct, UnsupportedExpectationError
import Enzyme
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, getconditioner

# Map our mode types to Enzyme's mode singletons.
_enzyme_mode(::EnzymeReverse) = Enzyme.Reverse
_enzyme_mode(::EnzymeForward) = Enzyme.Forward

const EnzymeExpectation = ClosedFormExpectation{<:EnzymeBackend}

# For an ExponentialFamilyDistribution{T} where the expectation under T has a closed form, the
# gradient of the expectation with respect to the natural parameters is computed by
# differentiating the closed form:
#
#   E_q[f(x) ∇_η log q(x; η)] = ∇_η E_q[f(x)]
#
# The support depends on the support of the expectation, so it is declared by hand, together with
# the method of `expectation` that routers reach, as `@supports_expectation_gradient` would.
function BayesBase.supports_expectation(
        ::EnzymeExpectation, ::Type{ExpectationScoreProduct{F, Q}}, ::Type{Q}
    ) where {F, Q <: ExponentialFamilyDistribution}
    return BayesBase.supports_expectation(ClosedFormExpectation(), F, Q)
end

function BayesBase.expectation(
        backend::EnzymeExpectation, product::ExpectationScoreProduct{F, Q}, q::Q
    ) where {F, Q <: ExponentialFamilyDistribution}
    return BayesBase.expectation_gradient(backend, product.f, q)
end

function BayesBase.expectation_gradient(
        backend::EnzymeExpectation, f::F, q::ExponentialFamilyDistribution{T}
    ) where {F, T}
    BayesBase.supports_expectation_gradient(backend, F, typeof(q)) ||
        throw(UnsupportedExpectationError(backend, ExpectationScoreProduct(f, q), q))
    mode = _enzyme_mode(backend.gradient.mode)
    η    = collect(Float64, getnaturalparameters(q))
    c    = getconditioner(q)
    g = let f = f, T = T, c = c
        params -> BayesBase.expectation(ClosedFormExpectation(), f, ExponentialFamilyDistribution(T, params, c))
    end
    return first(Enzyme.gradient(mode, g, η))
end

end # module
