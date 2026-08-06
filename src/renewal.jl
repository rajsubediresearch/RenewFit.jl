# =====================================================================
# The renewal equation
#
#   lambda_t = R_t * sum_{s=1..L} w_s * y_{t-s}
#
# Multiple peaks emerge whenever R_t crosses 1. There is no mixture, no
# number of components to select, no onset times, no threshold parameter --
# so none of the combinatorial search problems that dominate the
# n-sub-epidemic framework exist here. It is also a convolution rather than
# an ODE: no solver, no stiffness, no dense interpolant.
# =====================================================================

"""
    force_of_infection(y, w, t)

`sum_s w_s * y_{t-s}` over available history. Returns 0 when `t` has no
history, which excludes that point from the likelihood.
"""
@inline function force_of_infection(y::AbstractVector, w::AbstractVector, t::Integer)
    acc = zero(eltype(y)) * zero(eltype(w))
    @inbounds for s in 1:min(length(w), t - 1)
        acc += w[s] * y[t - s]
    end
    return acc
end

"""
    fit_window(y, w; min_force=1.0)

Indices where the model is identifiable: the force of infection must exceed
`min_force`, so the earliest points (with almost no history behind them) are
excluded rather than fitted with an essentially arbitrary R.
"""
function fit_window(y::AbstractVector, w::AbstractVector; min_force::Real=1.0)
    idx = Int[]
    for t in 2:length(y)
        force_of_infection(y, w, t) > min_force && push!(idx, t)
    end
    return idx
end

"""
    expected_incidence(y, w, R, idx)

Expected counts at `idx` given observed history `y` and R values aligned with
`idx`. This is the ONE-STEP-AHEAD mean: each point conditions on observed
history, not on simulated history. Forecasting propagates simulated counts
forward instead (see `forecast`).
"""
function expected_incidence(y::AbstractVector, w::AbstractVector,
                            R::AbstractVector, idx::AbstractVector{<:Integer})
    length(R) == length(idx) || error("R and idx must have the same length")
    return [R[i] * force_of_infection(y, w, idx[i]) for i in eachindex(idx)]
end

"""
    dow_index(t, start_dow)

Day of week (1 = Monday) for observation index `t`, given the weekday of the
first observation.
"""
@inline dow_index(t::Integer, start_dow::Integer) =
    mod(start_dow - 1 + (t - 1), 7) + 1

"""
    build_delta(v)

Seven multiplicative day-of-week factors from six free parameters, constrained
to a geometric mean of 1 so they cannot absorb the overall level (that is R's
job). `delta[7] = exp(-sum(v))`.
"""
build_delta(v::AbstractVector) = exp.(vcat(collect(v), -sum(v)))

"""
    simulate_renewal(y0, w, R; rng=nothing, phi=nothing, delta=nothing, start_dow=1)

Propagate the renewal equation forward from seed history `y0`.

With `delta` supplied, the day-of-week cycle is handled on BOTH sides: the
force of infection is built from DECONVOLVED history (`y / delta`), and the
expected observation is multiplied back by that day's factor. Using raw counts
in the convolution would leave the weekly cycle contaminating the transmission
term, which defeats the point.

Propagate the renewal equation forward from seed history `y0` for
`length(R)` steps. With `rng` and `phi` given, each step draws a negative
binomial observation and feeds it into the next step's force of infection --
the correct way to build a multi-step forecast, since uncertainty compounds
through the history. Without them, the deterministic mean path is returned.
"""
function simulate_renewal(y0::AbstractVector, w::AbstractVector, R::AbstractVector;
                          rng=nothing, phi::Union{Nothing,Real}=nothing,
                          delta::Union{Nothing,AbstractVector}=nothing,
                          start_dow::Integer=1)
    d = delta === nothing ? ones(7) : delta
    y = collect(float.(y0))
    yadj = [y[t] / d[dow_index(t, start_dow)] for t in eachindex(y)]
    out = Float64[]
    for h in eachindex(R)
        t = length(y) + 1
        dd = d[dow_index(t, start_dow)]
        mu = R[h] * force_of_infection(yadj, w, t) * dd
        isfinite(mu) || (mu = 1e12)
        val = (rng === nothing || phi === nothing) ? mu :
              float(rand(rng, nb_sampler(mu, phi)))
        push!(y, val); push!(yadj, val / dd); push!(out, val)
    end
    return out
end

"""
    nb_sampler(mu, phi; mu_max=1e12)

Negative binomial with mean `mu` and dispersion `phi`: var = mu + mu^2/phi.

The clamps are load-bearing, not defensive noise. An undamped random walk on
log R compounds through the simulated history, so over ~20 steps the mean can
overflow to `Inf`, and `phi/(phi+Inf)` is `NaN` — which `NegativeBinomial`
rejects. Clamping degrades an explosive path to an absurd-but-finite one so it
still shows up in the forecast quantiles as the blow-up it is, instead of
taking the whole run down.
"""
function nb_sampler(mu::Real, phi::Real; mu_max::Real=1e12)
    m = isfinite(mu) ? clamp(float(mu), 1e-10, float(mu_max)) : float(mu_max)
    r = isfinite(phi) ? clamp(float(phi), 1e-8, 1e12) : 1e-8
    p = clamp(r / (r + m), nextfloat(0.0), 1.0)
    return NegativeBinomial(r, p)
end

"""
    nb_logpdf(y, mu, phi)

Written out rather than taken from Distributions so it differentiates cleanly
under ForwardDiff for any parameter type.
"""
@inline function nb_logpdf(y::Real, mu::Real, phi::Real)
    m = max(mu, 1e-10)
    return loggamma(y + phi) - loggamma(phi) - loggamma(y + 1) +
           phi * log(phi / (phi + m)) + y * log(m / (phi + m))
end

"""
    phi_from(z, phi_max)

Dispersion from its unconstrained parameter: `phi = phi_max / (1 + exp(-z))`.

Defined once and used by the fitter, the forecaster and the fitted-curve
sampler. When these drifted apart -- fit.jl using the logistic transform while
forecast.jl still read `exp(z)` -- a fitted phi of 2.7 was interpreted
downstream as 0.00027, which put nearly all the negative binomial mass at zero
and made every forecast median exactly 0.
"""
@inline phi_from(z::Real, phi_max::Real) = phi_max / (1 + exp(-z))
