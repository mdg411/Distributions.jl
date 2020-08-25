"""
    LogGamma(α,θ)
The *Log-Gamma distribution* is the distribution of the exponential of a [`Gamma`](@ref) variate: if ``X \\sim \\operatorname{Gamma}(\\alpha, \\theta)`` then
``\\exp(X) \\sim \\operatorname{LogGamma}(\\alpha,\\theta)``. The probability density function is
```math
f(x; \\alpha, \\theta) = \\frac{(1+x-x_{min})^{-(1+1/\\theta)} \\log(1+x-x_{min})^{\\alpha-1}}{\\Gamma(\\alpha) \\theta^\\alpha},
	\\quad x \\geq x_{min}
```
where ``\\alpha > 0``, ``\\theta > 0`` and ``\\lambda \\geq 1``
```julia
LogGamma()       # LogGamma distribution with unit shape, unit scale and xmin = 1, i.e. LogGamma(1, 1, 1)
LogGamma(α)      # LogGamma distribution with shape α and unit scale and xmin = 1, i.e. LogGamma(α, 1, 1)
LogGamma(α, θ)   # LogGamma distribution with shape α and scale θ and xmin = 1
LogGamma(α, θ, xmin)  # LogGamma distribution with shape α and scale θ and xmin = 1
params(d)        # Get the parameters, i.e. (α, θ, xmin)
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
    xmin::T
    LogGamma{T}(α, θ, xmin) where {T} = new{T}(α, θ, xmin)
end

function LogGamma(α::T, θ::T, xmin::T; check_args=true) where {T <: Real}
    check_args && @check_args(LogGamma, α > zero(α) && θ > zero(θ) && xmin >= 1)
    return LogGamma{T}(α, θ, xmin)
end

LogGamma(α::Real, θ::Real, xmin::Real) = LogGamma(promote(α, θ, xmin)...)
LogGamma(α::Integer, θ::Integer, xmin::Integer) = LogGamma(float(α), float(θ), float(xmin))
LogGamma(α::Integer, θ::Integer) = LogGamma(α, θ, 1)
LogGamma(α::T, θ::S) where {T <: Real, S <: Real} = LogGamma(α, θ, one(T))
LogGamma(α::T) where {T <: Real} = LogGamma(α, one(T), one(T))
LogGamma() = LogGamma(1.0, 1.0, 1.0, check_args=false)

@distr_support LogGamma 1.0 Inf

#### Conversions
convert(::Type{LogGamma{T}}, α::S, θ::S, xmin::R) where {T <: Real, S <: Real, R <: Real} = LogGamma(T(α), T(θ), T(xmin))
convert(::Type{LogGamma{T}}, d::LogGamma{S}) where {T <: Real, S <: Real} = LogGamma(T(d.α), T(d.θ), T(d.xmin), check_args=false)

#### Parameters

shapelogx(d::LogGamma) = d.α
scalelogx(d::LogGamma) = d.θ
ratelogx(d::LogGamma) = 1 / d.θ

params(d::LogGamma) = (d.α, d.θ, d.xmin)
partype(::LogGamma{T}) where {T} = T


#### Statistics
minimum(d::LogGamma) = d.xmin

function mean(d::LogGamma)
    (α, θ, xmin) = params(d)
    θ < 1 ? (1-θ)^(-α) + xmin - 1 : Inf
end
function var(d::LogGamma)
    (α, θ, xmin) = params(d)
    θ < 1/2 ? (1-2θ)^(-α) - (1-θ)^(-2α) : Inf
end
function skewness(d::LogGamma)
    (α, θ, xmin) = params(d)
    if θ < 1/3
        ( (1-3θ)^(-α) - 3*(1 - 3θ + 2*θ^2)^(-α) + 2*(1-θ)^(-3α) ) / var(d)^(3/2)
    else
        Inf
    end
end
function kurtosis(d::LogGamma)
    (α, θ, xmin) = params(d)
    if θ < 1/4
        ( (1-4θ)^(-α) - 4*(1 - 4θ + 3*θ^2)^(-α) + 6*(1-2θ)^(-α)*(1-θ)^(-2α) ) / var(d)^2
    else
        Inf
    end
end

function mode(d::LogGamma)
    (α, θ, xmin) = params(d)
    α > 1 ? exp(θ*(α-1)/(θ+1)) + xmin - 1 : error("LogGamma has no mode when α <= 1")
end

function pdf(d::LogGamma, x::Real)
    (α, θ, xmin) = params(d)
    x >= xmin ? 1 / gamma(α) / θ^α * (1 + x - xmin)^(-1 - 1/θ) * log(1 + x - xmin)^(α-1) : 0.0
end

function cdf(d::LogGamma, x::Real)
    (α, θ, xmin) = params(d)
    x >= xmin ? cdf(Gamma(α, θ), log(x - xmin + 1)) : 0.0
end

quantile(d::LogGamma, q::Real) = exp(quantile(Gamma(d.α, d.θ), q)) + d.xmin - 1
cquantile(d::LogGamma, q::Real) = exp(cquantile(Gamma(d.α, d.θ), q)) + d.xmin - 1
invlogcdf(d::LogGamma, lq::Real) = exp(invlogcdf(Gamma(d.α, d.θ), lq)) + d.xmin - 1
invlogccdf(d::LogGamma, lq::Real) = exp(invlogccdf(Gamma(d.α, d.θ), lq)) + d.xmin - 1



#### Sampling

rand(rng::AbstractRNG, d::LogGamma) = exp(rand(Gamma(d.α, d.θ))) + d.xmin - 1
