using StaticArrays

function mean(::ClosedWilliamsProduct, p::Abs, q::Normal)
    μ, σ = q.μ, q.σ
    return @SVector [
        erf(μ/(sqrt(2)*σ)),
        sqrt(2/π)*exp(-μ^2/(2*σ^2))
    ]
end