# =====================================================================
# US COVID-19 daily CASES, Feb-May 2020, with a 14-day forecast.
#
# Cases rather than deaths on purpose: the renewal equation relates
# infections to infections, so a death series would first need
# deconvolving by an infection-to-death delay. Applying a generation
# interval directly to deaths silently mis-times R_t.
#
# Run:  julia --project=. --threads=auto examples/covid_cases.jl
# =====================================================================

using RenewFit, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
HORIZON, B, DAMP = 14, 500, 0.9

y_cal = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-cases-USA-05-11-20.txt"); column=52)
y_ext = load_series(joinpath(ROOT, "input",
    "cumulative-daily-coronavirus-cases-USA-05-25-20.txt"); column=52)
@assert isapprox(y_ext[1:length(y_cal)], y_cal) "files do not nest"
truth = y_ext[(length(y_cal) + 1):(length(y_cal) + HORIZON)]

gi = generation_interval(:covid_ancestral)
@printf("generation interval: mean %.2f d, sd %.2f, %d lags\n  source: %s\n",
        gi.mean, gi.sd, length(gi), gi.source)

# sigma_R is selected by held-out forecast score rather than guessed. The
# first run of this example used a free-floating smooth=10 with no scale
# relative to the likelihood; R_t interpolated the data (MAE 56.6 on counts of
# ~25,000, R up to 6.14) and phi ran to 3.7e8.
sel = select_sigma(y_cal, gi; horizon=HORIZON, B=200)
println("\nsigma_R selection (held-out $(HORIZON)-day WIS):")
println("  sigma_R     WIS      MAE   cov%      phi    maxR  conv")
for r in sel.table
    @printf("  %7.3f  %8.1f %8.1f %6.1f %8.1f %7.2f  %s\n",
            r.sigma_R, r.WIS, r.MAE, r.Coverage95, r.phi, r.maxR, r.converged)
end
@printf("selected sigma_R = %.3f\n\n", sel.sigma)

t0 = time()
f = fit_renewal(y_cal, gi; sigma_R=sel.sigma)
@printf("fit: %d points in %.2f s | converged=%s | phi=%.2f\n",
        length(f.idx), time() - t0, f.converged, f.phi)

lo, md, hi = rt_quantiles(f; B=B)
@printf("R_t at the forecast origin: %.2f (95%% CI %.2f-%.2f)\n", md[end], lo[end], hi[end])
@printf("R_t range over the window : %.2f to %.2f\n", minimum(md), maximum(md))
@printf("days with R>1: %d of %d\n", count(>(1.0), md), length(md))

fc = fitted_curves(f; B=B)
pc = performance(fc, [median(@view fc[i, :]) for i in 1:size(fc, 1)],
                 y_cal[f.idx])
@printf("calibration: MAE=%.1f  WIS=%.1f  coverage=%.1f%%\n",
        pc.MAE, pc.WIS, pc.Coverage95)

curves, point = forecast(f, HORIZON; B=B, damp=DAMP)
pf = performance(curves, point, truth)
@printf("%d-day forecast (sigma_R=%.3f, damp=%.2f): MAE=%.1f  MSE=%.1f  WIS=%.1f  coverage=%.1f%%\n",
        HORIZON, sel.sigma, DAMP, pf.MAE, pf.MSE, pf.WIS, pf.Coverage95)

println("\nhorizon   truth    median      95% PI")
for h in 1:HORIZON
    row = @view curves[h, :]
    @printf("  %2d    %7.0f  %8.0f   %7.0f - %7.0f\n", h, truth[h], point[h],
            quantile(row, 0.025), quantile(row, 0.975))
end

println("""
`damp` is still an unselected assumption -- re-run with damp in 0.5..1.0 and
see how much the forecast moves. Watch phi in the table above: values near the
bound of 1e4 mean R_t is interpolating and that row should not be trusted
regardless of its WIS.""")
