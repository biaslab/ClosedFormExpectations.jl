@testitem  "LogGamma" begin
    include("distributions_utils.jl")
    @test isa(convert(LogGamma{Float64}, Float16(1), Float16(1)), LogGamma{Float64})
    d = LogGamma(1, 1)
    @test convert(LogGamma{Float64}, d) === d
    @test isa(convert(LogGamma{Float32}, d), LogGamma{Float32})
    @test minimum(d) == -Inf
    @test maximum(d) == Inf

    # `@check_args` argument validation (regression for the Distributions.jl tuple-form macro change)
    @test_throws DomainError LogGamma(-1, 1)
    @test_throws DomainError LogGamma(1, -1)
    @test_throws DomainError LogGamma(0, 1)
    @test LogGamma(-1, 1; check_args=false) isa LogGamma
end
