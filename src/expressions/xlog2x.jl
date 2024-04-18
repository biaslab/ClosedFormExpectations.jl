export xlog2x

"""
Return `x * log(x)^2` for `x ≥ 0`, handling ``x = 0`` by taking the downward limit.

```jldoctest
julia> xlog2x(0)
0.0
```
"""
function xlog2x(x::Number)
    result = x * (log(x))^2
    return iszero(x) ? zero(result) : result
end