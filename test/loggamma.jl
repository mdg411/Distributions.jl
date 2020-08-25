using Distributions
using Test

@testset "LogGamma" begin

    @test isa(convert(LogGamma{Float64}, Float16(1), Float16(1), Float16(1)),
              LogGamma{Float64})
    @test logpdf(LogGamma(), Inf) === -Inf
    @test iszero(pdf(LogGamma(), 1-eps()))
    @test pdf(LogGamma(1), 1) == 1.0
    @test pdf(LogGamma(1,0.5), 1) == 2.0
    @test pdf(LogGamma(1,0.1), 1) == 10.0
    @test iszero(pdf(LogGamma(1+eps()), 1))
    @test iszero(pdf(LogGamma(1+eps(),0.5), 1))
    @test iszero(pdf(LogGamma(1+eps(),0.1), 1))
    @test iszero(logcdf(LogGamma(), Inf))
    @test isone(minimum(LogGamma()))
    @test isone(quantile(LogGamma(), 0))
    @test quantile(LogGamma(), 1) === Inf
    @test mean(LogGamma(1,0.1)) ≈ 1/0.9 rtol=1e-12
    @test var(LogGamma(1,0.1)) ≈ 0.015432098765432167 rtol=1e-12
    @test isfinite(mean(LogGamma(1,1-eps())))
    @test !isfinite(mean(LogGamma(1,1)))
    @test isfinite(var(LogGamma(1,0.5-eps())))
    @test !isfinite(var(LogGamma(1,0.5)))
    @test isfinite(skewness(LogGamma(1,1/3-eps())))
    @test !isfinite(skewness(LogGamma(1,1/3)))
    @test isfinite(kurtosis(LogGamma(1,0.25-eps())))
    @test !isfinite(kurtosis(LogGamma(1,0.25)))
    @test shapelogx(LogGamma(1,0.5)) == 1
    @test scalelogx(LogGamma(1,0.5)) == 0.5
    @test ratelogx(LogGamma(1,0.5)) ≈ 1/0.5
    @test LogGamma(1) == LogGamma(1, 1, 1)
    @test LogGamma(1,0.5) == LogGamma(1,0.5, 1)
    @test mean(LogGamma(1,0.5,1)) == mean(LogGamma(1,0.5,2)) - 1
    @test quantile(LogGamma(1,0.5,1), 0.5) == quantile(LogGamma(1,0.5,2), 0.5) - 1
    @test cquantile(LogGamma(1,0.5,1), 0.5) == cquantile(LogGamma(1,0.5,2), 0.5) - 1
    @test var(LogGamma(1,0.1,1)) == var(LogGamma(1,0.1,2))
    @test skewness(LogGamma(1,0.1,1)) == skewness(LogGamma(1,0.1,2))
    @test kurtosis(LogGamma(1,0.1,1)) == kurtosis(LogGamma(1,0.1,2))
    @test cdf(LogGamma(1,0.5,1), 3) == cdf(LogGamma(1,0.5,2), 3+1)
    @test minimum(LogGamma()) == 1
    @test minimum(LogGamma(1,1,2)) == 2
    @test insupport(LogGamma(1,1,2), 2-1e-12) == false
    @test insupport(LogGamma(1,1,1.9), 2-1e-12) == true
    @test invlogcdf(LogGamma(1,0.1), log(cdf(LogGamma(1,0.1), 3))) ≈ 3.0
    @test invlogcdf(LogGamma(1,1,2), log(cdf(LogGamma(1,1,2), 3))) ≈ 3.0
end


# using Plots
# grid = 1:0.05:5
# plot(grid, pdf.(LogGamma(2.2,0.2), grid))
# plot!(grid, pdf.(LogGamma(2.2,0.2,1.5), grid))
# plot!(grid, pdf.(LogGamma(2.2,0.2,2), grid))
#
# plot(grid, cdf.(LogGamma(2.2,0.2), grid))
# plot!(grid, cdf.(LogGamma(2.2,0.2,1.5), grid))
# plot!(grid, cdf.(LogGamma(2.2,0.2,2), grid))
#
# histogram(rand(LogGamma(2.2,0.2,1), 100000), xlims=(1,5), alpha=0.5)
# histogram!(rand(LogGamma(2.2,0.2,1.5), 100000), xlims=(1,5), alpha=0.5)
# histogram!(rand(LogGamma(2.2,0.2,2), 100000), xlims=(1,5), alpha=0.5)
