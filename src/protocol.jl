import BayesBase: supports_expectation, supports_expectation_gradient, has_expectation_label, ExpectationExact,
    ExpectationScoreProduct, UnsupportedExpectationError, register_expectation_backend!
import Distributions: Distribution, Categorical, Gamma, Normal
import ExponentialFamily: ExponentialFamilyDistribution, ProductOf, NormalMeanVariance, GammaShapeRate

# Most closed forms are methods of `expectation` and `expectation_gradient` defined with
# `@supports_expectation` and `@supports_expectation_gradient`, which also declare them. The pairs
# below follow from those by a rule instead: a sum over a `Categorical`, a product of log-densities
# or a `Product` expression split into its parts, and a distribution converted to the one the
# closed forms take. The rules are tried in a fixed order, in one method each for the support, the
# expectation and the gradient, so that they cannot be ambiguous with each other. The macros'
# methods are more specific and are chosen before them.

function supports_expectation(backend::ClosedFormExpectation, ::Type{F}, ::Type{Q}) where {F, Q}
    F <: ExpectationScoreProduct && return supports_derived_gradient(backend, F, Q)
    F <: LogpdfOf{<:ProductOf} && return supports_both_parts(backend, F, Q)
    F <: ComposedFunction{typeof(log), <:Product} && return supports_every_multiplier(backend, F, Q)
    # Any integrand under a `Categorical`, by an exact sum. `F = Any` asks for every integrand,
    # which the backend does not claim: an expectation under a product of distributions needs the
    # points of each component, which a numerical backend has
    Q <: Categorical && return F !== Any
    Q <: GammaShapeRate && return supports_expectation(backend, F, Gamma{parameter_type(Q)})
    Q <: ExponentialFamilyDistribution && return supports_expectation(backend, F, standard_type(Q))
    return false
end

# A product of distributions is computed from the points of its components
supports_expectation(::ClosedFormExpectation, ::Type, ::Type{BayesBase.FactorizedJoint{T}}) where {T <: Tuple} = false

function expectation(backend::ClosedFormExpectation, f::F, q) where {F}
    supports_expectation(backend, F, typeof(q)) || throw(UnsupportedExpectationError(backend, f, q))
    F <: ExpectationScoreProduct && return derived_gradient(backend, f.f, q)
    if F <: LogpdfOf{<:ProductOf}
        product = f.x
        return expectation(backend, Base.Fix1(logpdf, product.left), q) +
            expectation(backend, Base.Fix1(logpdf, product.right), q)
    end
    F <: ComposedFunction{typeof(log), <:Product} &&
        return sum(multiplier -> expectation(backend, log ∘ multiplier, q), f.inner.multipliers)
    q isa Categorical && return categorical_expectation(f, q)
    q isa GammaShapeRate && return expectation(backend, f, convert(Gamma, q))
    return expectation(backend, f, convert(Distribution, q)) # an `ExponentialFamilyDistribution`
end

# The type of the parameters of a distribution
parameter_type(::Type{<:GammaShapeRate{T}}) where {T} = T
parameter_type(::Type{<:NormalMeanVariance{T}}) where {T} = T
parameter_type(::Type{<:Gamma{T}}) where {T} = T
parameter_type(::Type{<:Normal{T}}) where {T} = T

# The distribution an `ExponentialFamilyDistribution` converts to
standard_type(::Type{<:ExponentialFamilyDistribution{T}}) where {T} = T
function standard_type(::Type{<:ExponentialFamilyDistribution{T, P}}) where {T, P <: AbstractArray}
    return with_parameter_type(T, eltype(P))
end

with_parameter_type(::Type{T}, ::Type) where {T} = T
with_parameter_type(::Type{NormalMeanVariance}, ::Type{E}) where {E} = NormalMeanVariance{E}
with_parameter_type(::Type{Gamma}, ::Type{E}) where {E} = Gamma{E}

product_parts(::Type{LogpdfOf{ProductOf{L, R}}}) where {L, R} = (L, R)

function supports_both_parts(backend, ::Type{F}, ::Type{Q}) where {F, Q}
    L, R = product_parts(F)
    return supports_expectation(backend, LogpdfOf{L}, Q) && supports_expectation(backend, LogpdfOf{R}, Q)
end

function supports_every_multiplier(backend, ::Type{ComposedFunction{typeof(log), Product{T}}}, ::Type{Q}) where {T <: Tuple, Q}
    return all(M -> supports_expectation(backend, ComposedFunction{typeof(log), M}, Q), fieldtypes(T))
end
supports_every_multiplier(backend, ::Type, ::Type) = false

## Gradients that follow from others by a rule, for the formulas of the package

supports_derived_gradient(backend, ::Type, ::Type) = false
function supports_derived_gradient(
        backend::ClosedFormExpectation{Nothing}, ::Type{ExpectationScoreProduct{G, Q}}, ::Type{Q}
    ) where {G, Q}
    # The gradient of a product of log-densities is the sum of the gradients of its parts
    if G <: LogpdfOf{<:ProductOf}
        L, R = product_parts(G)
        return supports_expectation_gradient(backend, LogpdfOf{L}, Q) &&
            supports_expectation_gradient(backend, LogpdfOf{R}, Q)
    end
    # Under a `NormalMeanVariance` and the exponential family forms of a `NormalMeanVariance` and a
    # `Gamma`, from the gradient under a `Normal` or a `Gamma` and the Jacobian
    Q <: NormalMeanVariance && return supports_expectation_gradient(backend, G, Normal{parameter_type(Q)})
    if Q <: ExponentialFamilyDistribution{NormalMeanVariance}
        return supports_expectation_gradient(backend, G, Normal{parameter_type(standard_type(Q))})
    end
    Q <: ExponentialFamilyDistribution{Gamma} && return supports_expectation_gradient(backend, G, standard_type(Q))
    return false
end

function derived_gradient(backend, f::G, q) where {G}
    if G <: LogpdfOf{<:ProductOf}
        product = f.x
        return expectation_gradient(backend, Base.Fix1(logpdf, product.left), q) .+
            expectation_gradient(backend, Base.Fix1(logpdf, product.right), q)
    end
    q isa NormalMeanVariance && return normalmeanvariance_gradient(backend, f, q)
    q isa ExponentialFamilyDistribution{NormalMeanVariance} && return normalmeanvariance_ef_gradient(backend, f, q)
    return gamma_ef_gradient(backend, f, q) # the exponential family form of a `Gamma`
end

## Labels and registration

has_expectation_label(::ClosedFormExpectation, ::ExpectationExact) = true

function __init__()
    register_expectation_backend!(ClosedFormExpectation())
    return nothing
end
