module ClosedFormExpectationsEnzymeExt

using ClosedFormExpectations
import ADTypes: AutoEnzyme
import BayesBase
import BayesBase: ExpectationScoreProduct, UnsupportedExpectationError
import Distributions: UnivariateDistribution
import Enzyme
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, getconditioner

const EnzymeAD = ClosedFormAD{<:AutoEnzyme}

# Reverse mode unless the `AutoEnzyme` names a mode
enzyme_mode(::AutoEnzyme{Nothing}) = Enzyme.Reverse
enzyme_mode(ad::AutoEnzyme) = ad.mode

# For a univariate `ExponentialFamilyDistribution{T}` where the expectation under `T` has a closed
# form, the gradient of the expectation with respect to the natural parameters is computed by
# differentiating the closed form:
#
#   E_q[f(x) ∇_η log q(x; η)] = ∇_η E_q[f(x)]
#
# The support depends on the support of the expectation, so it is declared by hand, together with
# the method of `expectation` that routers reach, as `@supports_expectation_gradient` would.
# Multivariate forms are not claimed: their closed forms go through factorisations that Enzyme
# cannot differentiate.
function BayesBase.supports_expectation(
        ::EnzymeAD, ::Type{ExpectationScoreProduct{F, Q}}, ::Type{Q}
    ) where {F, T, Q <: ExponentialFamilyDistribution{T}}
    T <: UnivariateDistribution || return false
    return BayesBase.supports_expectation(ClosedFormExpectation(), F, Q)
end

function BayesBase.expectation(
        backend::EnzymeAD, product::ExpectationScoreProduct{F, Q}, q::Q
    ) where {F, Q <: ExponentialFamilyDistribution}
    return BayesBase.expectation_gradient(backend, product.f, q)
end

function BayesBase.expectation_gradient(backend::EnzymeAD, f::F, q::ExponentialFamilyDistribution{T}) where {F, T}
    BayesBase.supports_expectation_gradient(backend, F, typeof(q)) ||
        throw(UnsupportedExpectationError(backend, ExpectationScoreProduct(f, q), q))
    η = float.(collect(getnaturalparameters(q)))
    c = getconditioner(q)
    g = let f = f, c = c
        params -> BayesBase.expectation(ClosedFormExpectation(), f, ExponentialFamilyDistribution(T, params, c))
    end
    return first(Enzyme.gradient(enzyme_mode(backend.ad), Enzyme.Const(g), η))
end

end # module
