module ClosedFormExpectationsEnzymeExt

using ClosedFormExpectations
import ClosedFormExpectations: mean_ef_impl
import Enzyme
import ExponentialFamily: ExponentialFamilyDistribution, getnaturalparameters, getconditioner

# Map our mode types to Enzyme's mode singletons.
_enzyme_mode(::EnzymeReverse) = Enzyme.Reverse
_enzyme_mode(::EnzymeForward) = Enzyme.Forward

# For any ExponentialFamilyDistribution{T} where a ClosedFormExpectation is defined,
# the Williams product equals the gradient of the expectation w.r.t. natural parameters:
#
#   E_q[f(x) ∇_η log q(x; η)] = ∇_η E_q[f(x)]
#
# We exploit this identity by differentiating the closed-form expectation via Enzyme.
function mean_ef_impl(
    strategy::ClosedWilliamsProduct{EnzymeBackend{M}},
    f,
    q::ExponentialFamilyDistribution{T},
) where {M, T}
    mode = _enzyme_mode(strategy.backend.mode)
    η    = collect(Float64, getnaturalparameters(q))
    c    = getconditioner(q)
    g = let f = f, T = T, c = c
        params -> mean(ClosedFormExpectation(), f, ExponentialFamilyDistribution(T, params, c))
    end
    return first(Enzyme.gradient(mode, g, η))
end

end # module
