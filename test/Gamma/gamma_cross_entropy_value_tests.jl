@testitem "Gamma Cross Entropy" begin
    using ClosedFormExpectations
    using Distributions
    using Test
    using SpecialFunctions
    
    @testset "Logpdf{Gamma} on Gamma" begin
        p_dist = Gamma(2.0, 2.0)
        p = Logpdf(p_dist)
        q = Gamma(3.0, 1.0)
        
        α_p, θ_p = params(p_dist)
        α_q, θ_q = params(q)
        
        # E_q[log p]
        # = -log Γ(α_p) - α_p log θ_p + (α_p - 1) E[log x] - E[x]/θ_p
        E_log_x = digamma(α_q) + log(θ_q)
        E_x = α_q * θ_q
        
        expected = -loggamma(α_p) - α_p * log(θ_p) + (α_p - 1) * E_log_x - E_x / θ_p
        
        val = mean(ClosedFormExpectation(), p, q)
        
        @test val ≈ expected
    end
end


