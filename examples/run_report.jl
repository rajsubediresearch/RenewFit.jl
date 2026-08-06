# =====================================================================
# Single-origin report for any dataset: fit, forecast, score, save, plot.
#
#   julia --project=examples examples/run_report.jl              # covid_usa
#   julia --project=examples examples/run_report.jl mpox_usa
#
# Output goes to output/<dataset>/report/.
# =====================================================================

using RenewFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "report")
B   = 500
H   = D.horizon

# calibrate on everything except the final H points, forecast those
y_cal = D.y[1:(end - H)]
truth = D.y[(end - H + 1):end]

@printf("dataset %s | %d %ss calibration, %d-%s horizon\n",
        D.name, length(y_cal), D.unit, H, D.unit)
@printf("GI: mean %.2f %ss, sd %.2f, %d lags%s\n  %s\n",
        D.gi.mean, D.gi.unit, D.gi.sd, length(D.gi),
        D.gi.serial ? "  [SERIAL-interval stand-in]" : "", D.gi.source)

sel = select_params(y_cal, D.gi; horizon=H, B=200,
                    sigma_grid=D.sigma_grid, damp_grid=D.damp_grid)
@printf("selected sigma_R=%.3f damp=%.1f\n", sel.sigma, sel.damp)

f  = fit_renewal(y_cal, D.gi; sigma_R=sel.sigma)
fc = fitted_curves(f; B=B)
curves, point = forecast(f, H; B=B, damp=sel.damp)

lo, md, hi = rt_quantiles(f; B=B)
@printf("phi=%.2f converged=%s | R final %.2f (95%% CI %.2f-%.2f) | R>1 in %d of %d\n",
        f.phi, f.converged, md[end], lo[end], hi[end], count(>(1.0), md), length(md))

pc = performance(fc, [median(@view fc[i, :]) for i in 1:size(fc, 1)], y_cal[f.idx])
pf = performance(curves, point, truth)
@printf("calibration: MAE=%.1f WIS=%.1f cov=%.1f%%\n", pc.MAE, pc.WIS, pc.Coverage95)
@printf("forecast   : MAE=%.1f WIS=%.1f cov=%.1f%%\n", pf.MAE, pf.WIS, pf.Coverage95)

bc, bp = persistence_forecast(y_cal, H; B=B)
pb = performance(bc, bp, truth)
@printf("persistence: MAE=%.1f WIS=%.1f cov=%.1f%%   skill %+.1f%%\n",
        pb.MAE, pb.WIS, pb.Coverage95, 100 * (1 - pf.WIS / pb.WIS))

ph = performance_by_horizon(curves, point, truth)
println("\n  h    truth    point       AE  width95  covered      WIS")
for r in ph
    @printf("  %2d %8.0f %8.0f %8.0f %8.0f %8d %8.1f\n", r.horizon, r.truth,
            r.point, r.AE, r.width95, r.covered, r.WIS)
end

save_fit(OUT, f, fc); save_rt(OUT, f; B=B)
save_forecast(OUT, curves, point; truth=truth, t0=f.idx[end])
save_performance(OUT, "performance.csv",
    [merge((window = "calibration", model = "renewal"), pc),
     merge((window = "forecast", model = "renewal"), pf),
     merge((window = "forecast", model = "persistence"), pb)])
save_performance(OUT, "performance-by-horizon.csv", ph)

extra = (damp = sel.damp, horizon = H, bootstrap_draws = B, unit = D.unit,
         selection = "held-out $H-$(D.unit) WIS")
if D.start_dow === nothing
    save_settings(OUT, f; extra..., day_of_week = "not applicable (non-daily grid)")
else
    wr = weekday_ratios(f; start_dow=D.start_dow)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    println("\nobserved/expected by weekday (DIAGNOSTIC ONLY -- this estimator")
    println("fails a known-truth test, see README):")
    for d in 1:7
        @printf("  %s %.3f%s\n", names[d], wr[d], abs(wr[d]-1) > 0.10 ? "  <==" : "")
    end
    save_performance(OUT, "weekday-ratios.csv",
        [(weekday = names[d], ratio = wr[d]) for d in 1:7])
    savefig(plot_weekday(f; start_dow=D.start_dow), joinpath(OUT, "weekday.png"))
    save_settings(OUT, f; extra..., start_dow = D.start_dow)
end

savefig(plot_fit(f, fc), joinpath(OUT, "fit.png"))
savefig(plot_rt(f; B=B), joinpath(OUT, "rt.png"))
savefig(plot_forecast(f, curves, point; truth=truth), joinpath(OUT, "forecast.png"))
savefig(plot_horizon(ph; metric=:WIS,
                     title="Forecast error by horizon (single origin)"),
        joinpath(OUT, "horizon-wis.png"))
savefig(plot_width_coverage(ph), joinpath(OUT, "horizon-width.png"))

println("\nwritten to $OUT")
