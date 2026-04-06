# [Expression Types](@id lib-expressions)

The package defines an `Expression` abstract type and several concrete expression types that can be used as the function argument `f` in `mean(strategy, f, q)`. These types support function composition via the `∘` operator.

## [ExpLogSquare](@id lib-explogs)

```@docs
ExpLogSquare
```

Represents the function:

```math
\text{ExpLogSquare}(\mu, \sigma)(x) = \exp\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)
```

Typically used in composed form with `log`:

```julia
# Computes E_q[-(log x - μ)² / (2σ²)]
mean(ClosedFormExpectation(), log ∘ ExpLogSquare(μ, σ), q)
```

## [Square](@id lib-square)

```@docs
Square
```

Represents ``x^2``. Typically used composed with `log` to compute expectations of ``(\log x)^2``:

```julia
# Computes E_q[(log x)²]
mean(ClosedFormExpectation(), Square() ∘ log, q)
```

## [Power](@id lib-power)

```@docs
Power
```

Represents ``x^N`` for compile-time constant ``N``. Used composed with `log`:

```julia
# Computes E_q[(log x)³]
mean(ClosedFormExpectation(), Power(Val(3)) ∘ log, q)
```

## [Abs](@id lib-abs)

```@docs
Abs
```

Represents the absolute value function ``|x|``:

```julia
# Computes E_q[|x|] where q is Normal
mean(ClosedFormExpectation(), Abs(), Normal(0, 1))
```

## [Product](@id lib-product)

```@docs
Product
```

Represents a product of multiple expressions. When composed with `log`, the logarithm of the product decomposes into a sum:

```math
\mathbb{E}_q[\log(f_1 \cdot f_2 \cdots f_n)] = \sum_i \mathbb{E}_q[\log f_i]
```

## [Standalone Functions](@id lib-standalone-functions)

### `xlog2x`

```@docs
xlog2x
```

Computes ``x (\log x)^2`` with proper handling of ``x = 0``.

### `xlogx`

The package also uses `xlogx` from [LogExpFunctions.jl](https://github.com/JuliaStats/LogExpFunctions.jl), which computes ``x \log x``.
