# =====================================================================
# How much does the generation interval drive the answer?
#
#   julia --project=examples examples/run_gi_sensitivity.jl            # covid_usa
#   julia --project=examples examples/run_gi_sensitivity.jl mpox_usa
#
# WHY THIS EXISTS. Every R(t) credible interval elsewhere in this package
# conditions on ONE fixed kernel, so it reflects sampling variability given
# that kernel and nothing about the fact that published COVID generation
# intervals range from about 3.95 to 5.9 days depending on study and method.
#
# R depends on the GI to first order. For a gamma kernel and exponential
# growth rate r,
#
#     R ~ (1 + r * sd^2 / mean)^(mean^2 / sd^2)
#
# so swapping one published estimate for another shifts R -- plausibly by more
# than the width of the interval we report -- and shifts the DATE R crosses 1,
# which is the quantity anyone actually acts on.
#
# Refitting across the published range is cheap (~1 s per kernel) and turns an
# unstated assumption into a reported one.
# =====================================================================

using RenewFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "gi_sensitivity")
B, H = 400, D.horizon

haskey(D, :gi_grid) || error("dataset $(D.name) has no gi_grid in datasets.jl")

y_cal = D.y[1:(end - H)]
truth = D.y[(end - H + 1):end]
@printf("%s | %d %ss calibration, %d-%s horizon | %d kernels\n\n",
        D.name, length(y_cal), D.unit, H, D.unit, length(D.gi_grid))

"""First time index where the median R path drops below 1, or nothing."""
function crossing_time(idx, med)
    for i in 2:length(med)
        med[i - 1] >= 1.0 > med[i] && return float(idx[i])
    end
    return nothing
end

rows = NamedTuple[]
paths = Tuple{String,Vector{Float64},Vector{Float64}}[]

for (m, sd, src) in D.gi_grid
    gi = discretize_gamma(m, sd; unit=D.gi.unit, source=src)
    local sel, f
    try
        sel = select_params(y_cal, gi; horizon=H, B=150,
                            sigma_grid=D.sigma_grid, damp_grid=D.damp_grid)
        f = fit_renewal(y_cal, gi; sigma_R=sel.sigma, warn_unconverged=false)
    catch err
        @warn "kernel $src failed" err
        continue
    end
    lo, md, hi = rt_quantiles(f; B=B)
    curves, point = forecast(f, H; B=B, damp=sel.damp)
    pf = performance(curves, point, truth)
    xt = crossing_time(f.idx, md)

    push!(rows, (gi_mean = m, gi_sd = sd, source = src, lags = length(gi),
                 sigma_R = sel.sigma, damp = sel.damp, phi = f.phi,
                 at_bound = f.phi > 0.95 * f.phi_max ? 1 : 0,
                 R_end = md[end], R_lo = lo[end], R_hi = hi[end],
                 R_max = maximum(md),
                 cross_below_1 = xt === nothing ? NaN : xt,
                 MAE = pf.MAE, WIS = pf.WIS, Coverage95 = pf.Coverage95))
    push!(paths, (@sprintf("%.2f (%s)", m, src), collect(float.(f.idx)), md))

    @printf("mean %5.2f sd %4.2f  %-28s R_end %.2f (%.2f-%.2f)  Rmax %.2f  cross %s  WIS %8.1f%s\n",
            m, sd, src, md[end], lo[end], hi[end], maximum(md),
            xt === nothing ? "  n/a" : @sprintf("%5.1f", xt), pf.WIS,
            f.phi > 0.95 * f.phi_max ? "  [phi at bound]" : "")
end

isempty(rows) && error("no kernel produced a usable fit")

# ---- how big is the GI effect relative to the reported interval? --------
Rs = [r.R_end for r in rows]
ref = rows[1]
band = ref.R_hi - ref.R_lo
spread = maximum(Rs) - minimum(Rs)
println("\n", "="^74)
@printf("R at the forecast origin: %.2f to %.2f across kernels (spread %.3f)\n",
        minimum(Rs), maximum(Rs), spread)
@printf("95%% credible width under the reference kernel (%s): %.3f\n", ref.source, band)
@printf("ratio spread/band = %.2f  -> %s\n", spread / band,
        spread > band ?
        "GI CHOICE DOMINATES the reported uncertainty; a single-kernel interval understates it" :
        "GI choice is smaller than sampling uncertainty; the single-kernel interval is adequate")
xs = [r.cross_below_1 for r in rows if isfinite(r.cross_below_1)]
isempty(xs) || @printf("R crosses 1 between %s %.0f and %.0f across kernels\n",
                       D.unit, minimum(xs), maximum(xs))
println("="^74)

save_performance(OUT, "gi-sensitivity.csv", rows)

# ---- plots --------------------------------------------------------------
using Plots
p = plot(title=wrap_title("R(t) across published generation intervals ($(D.name))"),
         xlabel=D.unit, ylabel="R(t)", legend=:topright)
for (lab, t, md) in paths
    plot!(p, t, md; lw=2, label=lab)
end
hline!(p, [1.0]; color=:black, ls=:dash, lw=1.5, label="R = 1")
savefig(p, joinpath(OUT, "rt-envelope.png"))

q = scatter([r.gi_mean for r in rows], [r.R_end for r in rows];
            yerror=([r.R_end - r.R_lo for r in rows], [r.R_hi - r.R_end for r in rows]),
            ms=5, color=RGB(0.30, 0.45, 0.70), legend=false,
            title=wrap_title("R at the forecast origin vs GI mean ($(D.name))"),
            xlabel="generation interval mean ($(D.unit)s)", ylabel="R at origin")
hline!(q, [1.0]; color=:firebrick, ls=:dash)
savefig(q, joinpath(OUT, "r-vs-gi.png"))

if !isempty(xs)
    r2 = scatter([r.gi_mean for r in rows if isfinite(r.cross_below_1)], xs;
                 ms=5, color=RGB(0.55, 0.35, 0.60), legend=false,
                 title=wrap_title("When R crosses 1, by GI mean ($(D.name))"),
                 xlabel="generation interval mean ($(D.unit)s)",
                 ylabel="$(D.unit) of crossing")
    savefig(r2, joinpath(OUT, "crossing.png"))
end

println("\nwritten to $OUT")
