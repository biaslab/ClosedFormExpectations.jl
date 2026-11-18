import BayesBase: expectation, supports_expectation, has_expectation_label, ExpectationExact,
    UnsupportedExpectationError, register_expectation_backend!
import Distributions: Distribution, Categorical, Exponential, Gamma, LogNormal, Laplace, Normal, logpdf
import ExponentialFamily: ExponentialFamilyDistribution, ProductOf, NormalMeanVariance, GammaShapeRate,
    GammaDistributionsFamily, UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily
import LogExpFunctions: xlogx

"""
    has_closed_form(strategy, F::Type, Q::Type)
    has_closed_form(strategy, f, q)

Whether `mean(strategy, f, q)` has a closed form for integrands of type `F` under distributions of
type `Q`, where `strategy` is `ClosedFormExpectation()` or `ClosedWilliamsProduct()`.

The integrand can be written as a `Logpdf(p)`, as `Base.Fix1(logpdf, p)` or as the distribution `p`
itself. For `ClosedFormExpectation()` the distribution `q` can also be a `GammaShapeRate` or an
`ExponentialFamilyDistribution` whose standard form has a closed form. For
`ClosedWilliamsProduct()` the parametrization of the result depends on `Q`: natural parameters for an
`ExponentialFamilyDistribution`, standard parameters otherwise, as in
[Supported Pairs](@ref lib-supported-pairs).

```jldoctest
julia> ClosedFormExpectations.has_closed_form(ClosedFormExpectation(), Base.Fix1(logpdf, LogNormal(0.0, 1.0)), Gamma(2.0, 1.0))
true

julia> ClosedFormExpectations.has_closed_form(ClosedFormExpectation(), Base.Fix1(logpdf, Beta(2.0, 2.0)), Normal(0.0, 1.0))
false
```
"""
function has_closed_form(strategy, ::Type{F}, ::Type{Q}) where {F, Q}
    return closed_form_pair(strategy, integrand_type(F), distribution_type(strategy, Q))
end
has_closed_form(strategy, f, q) = has_closed_form(strategy, typeof(f), typeof(q))

# The integrand as the methods of `mean` see it
integrand_type(::Type{F}) where {F} = F
integrand_type(::Type{Base.Fix1{typeof(logpdf), D}}) where {D} = Logpdf{D}
integrand_type(::Type{D}) where {D <: Distribution} = Logpdf{D}

closed_form_integrand(f) = f
closed_form_integrand(f::Base.Fix1{typeof(logpdf)}) = Logpdf(f.x)
closed_form_integrand(d::Distribution) = Logpdf(d)

# The distribution as the methods of `mean` see it: `ClosedFormExpectation` converts an exponential
# family distribution to its standard form, and the protocol converts a `GammaShapeRate` to a `Gamma`
distribution_type(strategy, ::Type{Q}) where {Q} = Q
distribution_type(::ClosedFormExpectation, ::Type{<:GammaShapeRate{T}}) where {T} = Gamma{T}
distribution_type(::ClosedFormExpectation, ::Type{<:ExponentialFamilyDistribution{T}}) where {T} = T

closed_form_distribution(q) = q
closed_form_distribution(q::GammaShapeRate) = convert(Gamma, q)

# The pairs of the methods of `mean`, by hand: the methods for any integrand under an
# `ExponentialFamilyDistribution` or a `ProductOf` would claim every pair
closed_form_pair(strategy, ::Type, ::Type) = false

# Products of log-densities, and of expressions, split into a sum
function closed_form_pair(strategy, ::Type{Logpdf{ProductOf{L, R}}}, ::Type{Q}) where {L, R, Q}
    return closed_form_pair(strategy, Logpdf{L}, Q) && closed_form_pair(strategy, Logpdf{R}, Q)
end
function closed_form_pair(
        strategy::ClosedFormExpectation, ::Type{ComposedFunction{typeof(log), Product{T}}}, ::Type{Q}
    ) where {T <: Tuple, Q}
    return all(P -> closed_form_pair(strategy, ComposedFunction{typeof(log), P}, Q), fieldtypes(T))
end

## `ClosedFormExpectation()`

# Any integrand under a `Categorical`, by an exact sum
closed_form_pair(::ClosedFormExpectation, ::Type, ::Type{<:Categorical}) = true

const LogExpLogSquare = ComposedFunction{typeof(log), <:ExpLogSquare}

const ExpectationUnderExponential = Union{
    Logpdf{<:Exponential}, Logpdf{<:LogNormal}, typeof(log), ComposedFunction{typeof(log), typeof(identity)},
    LogExpLogSquare,
}
closed_form_pair(::ClosedFormExpectation, ::Type{<:ExpectationUnderExponential}, ::Type{<:Exponential}) = true

const ExpectationUnderGamma = Union{
    typeof(log), typeof(xlogx), typeof(xlog2x), ComposedFunction{Square, typeof(log)},
    ComposedFunction{Power{Val{3}}, typeof(log)}, LogExpLogSquare, Logpdf{<:LogNormal},
    Logpdf{<:GammaDistributionsFamily}, Logpdf{<:UnivariateNormalDistributionsFamily},
    Logpdf{<:ReLUForwardMessage}, Logpdf{<:ReLUBackwardMessage},
}
closed_form_pair(::ClosedFormExpectation, ::Type{<:ExpectationUnderGamma}, ::Type{<:Gamma}) = true

const ExpectationUnderNormal = Union{
    Logpdf{<:UnivariateNormalDistributionsFamily}, Logpdf{<:Laplace}, Abs, Logpdf{<:LogGamma},
    Logpdf{<:ReLUBackwardMessage},
}
function closed_form_pair(
        ::ClosedFormExpectation, ::Type{<:ExpectationUnderNormal}, ::Type{<:UnivariateNormalDistributionsFamily}
    )
    return true
end

const ExpectationUnderLogNormal = Union{
    Logpdf{<:GammaDistributionsFamily}, Logpdf{<:LogNormal}, Logpdf{<:UnivariateNormalDistributionsFamily},
    Logpdf{<:ReLUForwardMessage}, Logpdf{<:ReLUBackwardMessage},
}
closed_form_pair(::ClosedFormExpectation, ::Type{<:ExpectationUnderLogNormal}, ::Type{<:LogNormal}) = true

const ExpectationUnderMvNormal = Union{Logpdf{<:MultivariateNormalDistributionsFamily}, Logpdf{<:LinearLogGamma}}
function closed_form_pair(
        ::ClosedFormExpectation, ::Type{<:ExpectationUnderMvNormal}, ::Type{<:MultivariateNormalDistributionsFamily}
    )
    return true
end

## `ClosedWilliamsProduct()`, hand-coded

const WilliamsUnderExponential = Union{
    typeof(log), LogExpLogSquare, Logpdf{<:LogNormal}, Logpdf{<:Exponential},
}
closed_form_pair(::ClosedWilliamsProduct{Nothing}, ::Type{<:WilliamsUnderExponential}, ::Type{<:Exponential}) = true

const WilliamsUnderGamma = Union{
    typeof(log), ComposedFunction{Square, typeof(log)}, LogExpLogSquare, Logpdf{<:LogNormal},
    Logpdf{<:GammaDistributionsFamily}, Logpdf{<:UnivariateNormalDistributionsFamily},
}
closed_form_pair(::ClosedWilliamsProduct{Nothing}, ::Type{<:WilliamsUnderGamma}, ::Type{<:Gamma}) = true
function closed_form_pair(
        ::ClosedWilliamsProduct{Nothing}, ::Type{<:WilliamsUnderGamma}, ::Type{<:ExponentialFamilyDistribution{<:Gamma}}
    )
    return true
end

# Under a `NormalMeanVariance` and its exponential family form the product is computed under a
# `Normal{Float64}`, whose Laplace method needs a `Laplace{Float64}`
const WilliamsUnderNormal = Union{Abs, Logpdf{<:UnivariateNormalDistributionsFamily}, Logpdf{<:LogGamma}}
closed_form_pair(::ClosedWilliamsProduct{Nothing}, ::Type{<:WilliamsUnderNormal}, ::Type{<:Normal}) = true
closed_form_pair(::ClosedWilliamsProduct{Nothing}, ::Type{Logpdf{Laplace{T}}}, ::Type{Normal{T}}) where {T} = true
function closed_form_pair(
        ::ClosedWilliamsProduct{Nothing}, ::Type{<:Union{WilliamsUnderNormal, Logpdf{Laplace{Float64}}}},
        ::Type{<:Union{NormalMeanVariance, ExponentialFamilyDistribution{NormalMeanVariance}}}
    )
    return true
end

# Any integrand under the exponential family form of a `Categorical`, by an exact sum
function closed_form_pair(
        ::ClosedWilliamsProduct{Nothing}, ::Type, ::Type{<:ExponentialFamilyDistribution{<:Categorical}}
    )
    return true
end

# With Enzyme, wherever the expectation has a closed form under the standard form
function closed_form_pair(
        ::ClosedWilliamsProduct{<:EnzymeBackend}, ::Type{F}, ::Type{<:ExponentialFamilyDistribution{T}}
    ) where {F, T}
    return closed_form_pair(ClosedFormExpectation(), F, T)
end

## The expectation protocol of BayesBase

# `ClosedFormExpectation()` is a backend: `BayesBase.expectation(ClosedFormExpectation(), f, q)` is
# `mean(ClosedFormExpectation(), f, q)` for the pairs with a closed form. `F = Any` asks for every
# integrand, which only the exact sum under a `Categorical` could claim; a numerical backend serves
# that better, since an expectation under a product needs the points of each component.
function supports_expectation(strategy::ClosedFormExpectation, ::Type{F}, ::Type{Q}) where {F, Q}
    return F !== Any && has_closed_form(strategy, F, Q)
end

has_expectation_label(::ClosedFormExpectation, ::ExpectationExact) = true

function expectation(strategy::ClosedFormExpectation, f::F, q) where {F}
    supports_expectation(strategy, F, typeof(q)) || throw(UnsupportedExpectationError(strategy, f, q))
    return mean(strategy, closed_form_integrand(f), closed_form_distribution(q))
end

## `Base.Fix1(logpdf, p)` for the Williams product, as for the expectation

mean(c::ClosedWilliamsProduct, f::Base.Fix1{typeof(logpdf)}, q) = mean(c, Logpdf(f.x), q)
mean(c::ClosedWilliamsProduct, f::Base.Fix1{typeof(logpdf)}, q::ExponentialFamilyDistribution) = mean(c, Logpdf(f.x), q)
mean(c::ClosedWilliamsProduct, f::Base.Fix1{typeof(logpdf)}, q::NormalMeanVariance) = mean(c, Logpdf(f.x), q)

function __init__()
    register_expectation_backend!(ClosedFormExpectation())
    return nothing
end
