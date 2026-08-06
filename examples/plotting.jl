# Plot helpers. Included by the analysis scripts rather than shipped in the
# package, so Plots stays out of RenewFit's dependency list.

using Plots, Statistics

const BAND = RGBA(0.30, 0.45, 0.70, 0.25)
const LINE = RGB(0.15, 0.30, 0.55)

# ---------------------------------------------------------------------
# Global defaults, set once.
#
# Titles were clipping on the right ("... origin 75 (covid_u"). Two causes:
# the default canvas is narrow for the titles these scripts generate, and
# there is no top margin for a title to sit in. Fixed here rather than in each
# function so new plots inherit it.
# ---------------------------------------------------------------------
Plots.default(
    size = (900, 520),
    titlefontsize = 10,
    guidefontsize = 9,
    tickfontsize = 8,
    legendfontsize = 8,
    left_margin = 6Plots.mm,
    right_margin = 8Plots.mm,
    top_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
    fg_legend = :transparent,
    background_color_legend = RGBA(1, 1, 1, 0.75)
)

"""
    wrap_title(s; width=62)

Break a title onto multiple lines at word boundaries. Belt and braces with the
wider canvas above: dataset names and phase labels make these titles long, and
a two-line title is far better than a truncated one.
"""
function wrap_title(s::AbstractString; width::Integer=62)
    length(s) <= width && return String(s)
    lines, cur = String[], ""
    for w in split(s)
        if isempty(cur)
            cur = String(w)
        elseif length(cur) + 1 + length(w) <= width
            cur *= " " * w
        else
            push!(lines, cur); cur = String(w)
        end
    end
    isempty(cur) || push!(lines, cur)
    return join(lines, "\n")
end

function plot_fit(fit, curves; title=wrap_title("Model fit"))
    n = length(fit.idx)
    med = [median(@view curves[i, :]) for i in 1:n]
    lo  = [quantile(@view(curves[i, :]), 0.025) for i in 1:n]
    hi  = [quantile(@view(curves[i, :]), 0.975) for i in 1:n]
    p = plot(fit.idx, hi; fillrange=lo, fillcolor=BAND, linealpha=0,
             label="95% PI", legend=:topleft, title=wrap_title(title),
             xlabel="day", ylabel="cases")
    plot!(p, fit.idx, med; color=LINE, lw=2, label="fitted median")
    scatter!(p, fit.idx, fit.y[fit.idx]; color=:black, ms=2.5, alpha=0.7,
             label="observed")
    return p
end

function plot_rt(fit; B=500, seed=20260101, title=wrap_title("Effective reproduction number"))
    lo, md, hi = rt_quantiles(fit, (0.025, 0.5, 0.975); B=B, seed=seed)
    p = plot(fit.idx, hi; fillrange=lo, fillcolor=BAND, linealpha=0,
             label="95% CrI", legend=:topright, title=wrap_title(title),
             xlabel="day", ylabel="R(t)")
    plot!(p, fit.idx, md; color=LINE, lw=2, label="median")
    hline!(p, [1.0]; color=:firebrick, ls=:dash, lw=1.5, label="R = 1")
    return p
end

"""Mean observed/expected by weekday: flat near 1.0 means no reporting cycle."""
function plot_weekday(fit; start_dow=1, title=wrap_title("Observed / expected by weekday"))
    r = weekday_ratios(fit; start_dow=start_dow)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = [x < 1 ? RGB(0.75, 0.35, 0.30) : RGB(0.30, 0.50, 0.40) for x in r]
    p = bar(names, r; color=cols, legend=false, title=wrap_title(title),
            ylabel="observed / expected")
    hline!(p, [1.0]; color=:black, ls=:dash, lw=1.5)
    return p
end

function plot_forecast(fit, curves, point; truth=nothing, history=30,
                       title=wrap_title("Forecast"))
    H = size(curves, 1)
    tf = fit.idx[end] .+ (1:H)
    lo50 = [quantile(@view(curves[h, :]), 0.25) for h in 1:H]
    hi50 = [quantile(@view(curves[h, :]), 0.75) for h in 1:H]
    lo95 = [quantile(@view(curves[h, :]), 0.025) for h in 1:H]
    hi95 = [quantile(@view(curves[h, :]), 0.975) for h in 1:H]
    hstart = max(1, length(fit.idx) - history + 1)
    ht = fit.idx[hstart:end]

    p = plot(tf, hi95; fillrange=lo95, fillcolor=BAND, linealpha=0,
             label="95% PI", legend=:topleft, title=wrap_title(title), xlabel="day",
             ylabel="cases")
    plot!(p, tf, hi50; fillrange=lo50, fillcolor=RGBA(0.30, 0.45, 0.70, 0.45),
          linealpha=0, label="50% PI")
    plot!(p, tf, point; color=LINE, lw=2, label="median forecast")
    scatter!(p, ht, fit.y[ht]; color=:black, ms=2.5, alpha=0.7, label="observed")
    truth === nothing || scatter!(p, tf, truth; color=:firebrick, ms=3.5,
                                  label="truth")
    vline!(p, [fit.idx[end] + 0.5]; color=:grey, ls=:dash, label="")
    return p
end

"""
    plot_horizon(rows; baseline=nothing, title, metric=:WIS)

Degradation with lead time. `rows` comes from `performance_by_horizon`, or
from an aggregation of it across origins. Pass `baseline` rows to overlay a
persistence comparison — without one, a rising WIS curve is hard to read,
since some of the growth is just the series getting harder to predict.
"""
function plot_horizon(rows; baseline=nothing, title=wrap_title("Forecast skill by horizon"),
                      metric::Symbol=:WIS, ylabel=String(metric))
    h = [r.horizon for r in rows]
    v = [getproperty(r, metric) for r in rows]
    p = plot(h, v; color=LINE, lw=2, marker=:circle, ms=4, label="RenewFit",
             title=wrap_title(title), xlabel="forecast horizon (days)", ylabel=ylabel,
             legend=:topleft)
    if baseline !== nothing
        plot!(p, [r.horizon for r in baseline],
              [getproperty(r, metric) for r in baseline];
              color=:firebrick, lw=2, ls=:dash, marker=:diamond, ms=4,
              label="persistence")
    end
    return p
end

"""Interval width and coverage against horizon, on twin axes."""
function plot_width_coverage(rows; title=wrap_title("Interval width and coverage"))
    h = [r.horizon for r in rows]
    p = plot(h, [r.width95 for r in rows]; color=LINE, lw=2, marker=:circle,
             ms=4, label="95% width", title=wrap_title(title),
             xlabel="forecast horizon (days)", ylabel="interval width",
             legend=:topleft)
    if haskey(first(rows), :coverage)
        plot!(twinx(p), h, [r.coverage for r in rows]; color=:seagreen, lw=2,
              ls=:dash, marker=:square, ms=4, ylabel="coverage (%)",
              ylims=(0, 105), legend=:bottomright, label="coverage")
        hline!(twinx(p), [95.0]; color=:grey, ls=:dot, label="")
    end
    return p
end

"""
    plot_ensemble_fan(members, labels, ens, truth; t0)

Member forecasts and the ensemble on one panel: member medians as thin lines,
the ensemble median heavy with its 95% band, truth as points.
"""
function plot_ensemble_fan(members, labels, ens, truth; t0=0,
                           title=wrap_title("Ensemble vs members"))
    H = size(ens, 1)
    t = (1:H) .+ t0
    lo = [quantile(@view(ens[h, :]), 0.025) for h in 1:H]
    hi = [quantile(@view(ens[h, :]), 0.975) for h in 1:H]
    med = [median(@view(ens[h, :])) for h in 1:H]
    p = plot(t, hi; fillrange=lo, fillcolor=BAND, linealpha=0,
             label="ensemble 95% PI", title=wrap_title(title), xlabel="forecast horizon",
             ylabel="cases", legend=:outertopright)
    cols = [RGB(0.75, 0.45, 0.25), RGB(0.30, 0.55, 0.40), RGB(0.55, 0.35, 0.60)]
    for (i, m) in enumerate(members)
        plot!(p, t, [median(@view(m[h, :])) for h in 1:H];
              color=cols[mod1(i, length(cols))], lw=1.5, ls=:dash,
              label=labels[i])
    end
    plot!(p, t, med; color=LINE, lw=2.5, label="ensemble median")
    truth === nothing || scatter!(p, t, truth; color=:firebrick, ms=4, label="truth")
    return p
end

"""
    plot_predictive_density(members, labels, mix, vin; horizon=1)

Predictive densities at one horizon, showing what the two combination rules do
to disagreement: the mixture keeps both modes, vincentization collapses them.
Histogram-based so it needs no KDE dependency.
"""
function plot_predictive_density(members, labels, mix, vin; horizon::Int=1,
                                 title=wrap_title("Predictive density at h=$horizon"))
    p = plot(title=wrap_title(title), xlabel="cases", ylabel="density", legend=:topright)
    cols = [RGB(0.75, 0.45, 0.25), RGB(0.30, 0.55, 0.40), RGB(0.55, 0.35, 0.60)]
    for (i, m) in enumerate(members)
        histogram!(p, vec(m[horizon, :]); normalize=:pdf, bins=40, alpha=0.25,
                   linealpha=0, color=cols[mod1(i, length(cols))], label=labels[i])
    end
    histogram!(p, vec(mix[horizon, :]); normalize=:pdf, bins=40, alpha=0.0,
               linecolor=LINE, lw=2, seriestype=:steppost, label="mixture")
    histogram!(p, vec(vin[horizon, :]); normalize=:pdf, bins=40, alpha=0.0,
               linecolor=:firebrick, lw=2, ls=:dash, seriestype=:steppost,
               label="vincentized")
    return p
end

"""Bar chart of stacked ensemble weights."""
function plot_weights(labels, weights; title=wrap_title("Stacked ensemble weights"))
    p = bar(labels, weights; legend=false, title=wrap_title(title), ylabel="weight",
            color=RGB(0.30, 0.45, 0.70), ylims=(0, 1))
    hline!(p, [1 / length(weights)]; color=:black, ls=:dash, lw=1.5)
    return p
end
