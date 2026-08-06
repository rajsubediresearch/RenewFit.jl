# =====================================================================
# Forecasting
#
# THE HONEST WEAKNESS of this whole approach: forecasting requires
# extrapolating R_t, and a pure random walk gives explosive intervals by
# 30 days -- exactly the horizon SubEpiPredict targets. The growth model's
# saturating final size K was quietly doing real work there.
#
# So R is DAMPED toward 1: log R_{T+h} = damp^h * log R_T. `damp = 1` is a
# pure random walk (no damping), `damp = 0` snaps immediately to R = 1
# (incidence plateaus). This is an explicit assumption with a dial on it,
# not something buried in the machinery, and it should be reported alongside
# any forecast.
# =====================================================================

"""
    forecast(fit, horizon; B=500, damp=0.9, seed, rw_sd=nothing)

Simulate `B` forecast paths. Returns `(curves, point)` where `curves` is
`horizon x B` predictive samples and `point` is the median path.

`fix_phi=true` (the default) holds the dispersion at its MAP rather than
drawing it from the Laplace marginal. Because phi enters through a logistic
transform against a fairly flat likelihood, its marginal in z-space is wide,
so some draws land at a tiny phi and enormous overdispersion, fattening the
predictive tails. That is what produced 100% coverage and mean 95% interval
widths of ~25,700 on counts of ~20,000 at a ONE-DAY horizon. Plugging in the
MAP is the usual treatment for a weakly identified nuisance parameter; set
`fix_phi=false` to restore the fully propagated version.

Uncertainty otherwise enters three ways:
  * parameter uncertainty, from the Laplace draws of log R_T and log phi
  * process uncertainty, from a random walk on log R with SD `rw_sd`
    (defaults to the SD of fitted log R increments)
  * observation uncertainty, from the negative binomial draw at each step

Counts are fed forward into the next step's force of infection, so
uncertainty compounds through the history as it should.
"""
function forecast(fit::RenewalFit, horizon::Integer; B::Integer=500,
                  damp::Real=0.9, seed::Integer=20260101,
                  rw_sd::Union{Nothing,Real}=nothing, fix_phi::Bool=true)
    n = length(fit.idx)
    dlog = diff(log.(fit.R))
    sd = rw_sd === nothing ? (length(dlog) > 1 ? std(dlog) : 0.1) : float(rw_sd)

    thetas = sample_theta(fit, B; seed=seed)
    curves = zeros(Float64, horizon, B)
    y_hist = fit.y

    for b in 1:B
        th = thetas[b]
        rng = Xoshiro(hash((seed, b)))
        logR = th[n]                       # last fitted log R for this draw
        phi = fix_phi ? fit.phi : phi_from(th[end], fit.phi_max)
        Rpath = Float64[]
        for h in 1:horizon
            # clamped so an undamped walk cannot overflow exp(); |log R| = 20
            # is already far outside anything epidemiologically meaningful
            logR = clamp(damp * logR + sd * randn(rng), -20.0, 20.0)
            push!(Rpath, exp(logR))
        end
        curves[:, b] .= simulate_renewal(y_hist, fit.gi.w, Rpath; rng=rng, phi=phi,
                                         delta=fit.delta, start_dow=fit.start_dow)
    end

    point = [median(@view curves[h, :]) for h in 1:horizon]
    return curves, point
end

"""
    fitted_curves(fit; B=500, seed)

One-step-ahead predictive samples over the calibration window, for
calibration metrics and for plotting the fit with uncertainty.
"""
function fitted_curves(fit::RenewalFit; B::Integer=500, seed::Integer=20260101,
                       fix_phi::Bool=true)
    n = length(fit.idx)
    thetas = sample_theta(fit, B; seed=seed)
    curves = zeros(Float64, n, B)
    for b in 1:B
        th = thetas[b]
        phi = fix_phi ? fit.phi : phi_from(th[end], fit.phi_max)
        rng = Xoshiro(hash((seed, b, :fit)))
        yadj = [fit.y[t] / fit.delta[dow_index(t, fit.start_dow)]
                for t in eachindex(fit.y)]
        for i in 1:n
            t = fit.idx[i]
            mu = exp(th[i]) * force_of_infection(yadj, fit.gi.w, t) *
                 fit.delta[dow_index(t, fit.start_dow)]
            curves[i, b] = float(rand(rng, nb_sampler(mu, phi)))
        end
    end
    return curves
end

"""
    rt_quantiles(fit, qs; B=500, seed)

Pointwise quantiles of R_t from the Laplace draws — the interpretable output
this model class gives you that a phenomenological growth model does not.
"""
function rt_quantiles(fit::RenewalFit, qs=(0.025, 0.5, 0.975);
                      B::Integer=500, seed::Integer=20260101)
    n = length(fit.idx)
    thetas = sample_theta(fit, B; seed=seed)
    M = hcat([exp.(th[1:n]) for th in thetas]...)
    return [ [quantile(@view(M[i, :]), q) for i in 1:n] for q in qs ]
end
