"""
    LogGamma(α,θ)
The *Log-Gamma distribution* is the distribution of the exponential of a [`Gamma`](@ref) variate: if ``X \\sim \\operatorname{Gamma}(\\alpha, \\theta)`` then
``\\exp(X) \\sim \\operatorname{LogGamma}(\\alpha,\\theta)``. The probability density function is
```math
f(x; \\alpha, \\theta) = \\frac{x^{-(1+1/\\theta)} \\log(x)^{\\alpha-1}}{\\Gamma(\\alpha) \\theta^\\alpha},
	\\quad x \\geq 1
```
```julia
LogGamma()       # LogGamma distribution with unit shape and unit scale, i.e. LogGamma(1, 1)
LogGamma(α)      # LogGamma distribution with shape α and unit scale, i.e. LogGamma(α, 1)
LogGamma(α, θ)   # LogGamma distribution with shape α and scale θ
params(d)        # Get the parameters, i.e. (α, θ)
shapelogx(d)     # Get the shape of log(x), i.e. α
scalelogx(d)     # Get the scale of log(x), i.e. θ
ratelogx(d)      # Get the rate of log(x), i.e. 1/θ
```
External links

* [Log normal distribution on Wolfram](https://reference.wolfram.com/language/ref/LogGammaDistribution.html)

(no Wikipedia entry available)

"""
struct LogGamma{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T
    LogGamma{T}(α, θ) where {T} = new{T}(α, θ)
end

function LogGamma(α::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogGamma, α > zero(α) && θ > zero(θ))
    return LogGamma{T}(α, θ)
end

LogGamma(α::Real, θ::Real) = LogGamma(promote(α, θ)...)
LogGamma(α::Integer, θ::Integer) = LogGamma(float(α), float(θ))
LogGamma(α::T) where {T <: Real} = LogGamma(α, one(T))
LogGamma() = LogGamma(1.0, 1.0, check_args=false)

@distr_support LogGamma 1.0 Inf

#### Conversions
convert(::Type{LogGamma{T}}, α::S, θ::S) where {T <: Real, S <: Real} = LogGamma(T(α), T(θ))
convert(::Type{LogGamma{T}}, d::LogGamma{S}) where {T <: Real, S <: Real} = LogGamma(T(d.α), T(d.θ), check_args=false)

#### Parameters

shapelogx(d::LogGamma) = d.α
scalelogx(d::LogGamma) = d.θ
ratelogx(d::LogGamma) = 1 / d.θ

params(d::LogGamma) = (d.α, d.θ)
partype(::LogGamma{T}) where {T} = T


#### Statistics
function mean(d::LogGamma)
    (α, θ) = params(d)
    θ < 1 ? (1-θ)^(-α) : Inf
end
function var(d::LogGamma)
    (α, θ) = params(d)
    θ < 1/2 ? (1-2θ)^(-α) - (1-θ)^(-2α) : Inf
end
function skewness(d::LogGamma)
    (α, θ) = params(d)
    if θ < 1/3
        ( (1-3θ)^(-α) - 3*(1 - 3θ + 2*θ^2)^(-α) + 2*(1-θ)^(-3α) ) / var(d)^(3/2)
    else
        Inf
    end
end
function kurtosis(d::LogGamma)
    (α, θ) = params(d)
    if θ < 1/4
        ( (1-4θ)^(-α) - 4*(1 - 4θ + 3*θ^2)^(-α) + 6*(1-2θ)^(-α)*(1-θ)^(-2α) ) / var(d)^2
    else
        Inf
    end
end

function mode(d::LogGamma)
    (α, θ) = params(d)
    α > 1 ? exp(θ*(α-1)/(θ+1)) : error("LogGamma has no mode when α <= 1")
end

function pdf(d::LogGamma, x::Real)
    (α, θ) = params(d)
    x >= 1.0 ? 1 / gamma(α) / θ^α * x^(-1-1/θ) * log(x)^(α-1) : 0.0
end
cdf(d::LogGamma, x::Real) = cdf(Gamma(params(d)...),log(x))

quantile(d::LogGamma, q::Real) = exp(quantile(Gamma(params(d)...), q))
cquantile(d::LogGamma, q::Real) = exp(cquantile(Gamma(params(d)...), q))
invlogcdf(d::LogGamma, lq::Real) = exp(invlogcdf(Gamma(params(d)...), lq))
invlogccdf(d::LogGamma, lq::Real) = exp(invlogccdf(Gamma(params(d)...), lq))



#### Sampling

rand(rng::AbstractRNG, d::LogGamma) = exp(rand(Gamma(params(d)...)))
