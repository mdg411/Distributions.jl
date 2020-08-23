using Distributions
using Test

@testset "LogGamma" begin

    @test isa(convert(LogGamma{Float64}, Float16(1), Float16(1)),
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

end
