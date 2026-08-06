# =====================================================================
# Saving results
#
# Everything a run produces goes to disk as CSV: the fit with its band, R(t)
# with its interval, the forecast quantiles, the performance metrics, and a
# plain-text record of the settings that produced them.
#
# The settings file matters as much as the numbers. sigma_R, damp, the
# generation interval and its source are modelling CHOICES, and a metrics
# table without them is not reproducible. This is also where the
# serial-vs-generation-interval caveat gets recorded alongside the output
# rather than living only in a docstring.
# =====================================================================

_ensure(dir) = (isdir(dir) || mkpath(dir); dir)

"""
    output_dir(root, dataset, analysis)

`<root>/output/<dataset>/<analysis>`, created if needed.

Results are keyed by DATASET first, then by analysis, so running a new series
never overwrites an old one. Flat per-analysis folders (output/horizon,
output/ensemble) silently clobber the previous dataset's results, which is the
kind of loss you only notice when you go looking for a number you already
computed.

    output_dir(ROOT, "covid_usa", "ensemble")   ->  <root>/output/covid_usa/ensemble
    output_dir(ROOT, "mpox_usa",  "report")     ->  <root>/output/mpox_usa/report
"""
output_dir(root::AbstractString, dataset::AbstractString, analysis::AbstractString) =
    _ensure(joinpath(root, "output", dataset, analysis))

function _writecsv(path, header::Vector{String}, M::AbstractMatrix)
    open(path, "w") do io
        println(io, join(header, ","))
        writedlm(io, M, ',')
    end
    return path
end

"""
    save_fit(dir, fit, curves; tag="")

Observed counts, fitted median and 95% band over the calibration window.
"""
function save_fit(dir, fit::RenewalFit, curves::AbstractMatrix; tag::String="")
    _ensure(dir)
    n = length(fit.idx)
    med = [median(@view curves[i, :]) for i in 1:n]
    lo  = [quantile(@view(curves[i, :]), 0.025) for i in 1:n]
    hi  = [quantile(@view(curves[i, :]), 0.975) for i in 1:n]
    M = hcat(float.(fit.idx), fit.y[fit.idx], med, lo, hi)
    return _writecsv(joinpath(dir, "fit$(isempty(tag) ? "" : "-" * tag).csv"),
                     ["t", "observed", "fitted_median", "lo95", "hi95"], M)
end

"""
    save_rt(dir, fit; B=500, seed, tag="")

R(t) with a 95% credible interval — the interpretable output this model class
gives you that a phenomenological growth curve does not.
"""
function save_rt(dir, fit::RenewalFit; B::Integer=500, seed::Integer=20260101,
                 tag::String="")
    _ensure(dir)
    lo, md, hi = rt_quantiles(fit, (0.025, 0.5, 0.975); B=B, seed=seed)
    M = hcat(float.(fit.idx), md, lo, hi, fit.R)
    return _writecsv(joinpath(dir, "rt$(isempty(tag) ? "" : "-" * tag).csv"),
                     ["t", "R_median", "R_lo95", "R_hi95", "R_map"], M)
end

"""
    save_forecast(dir, curves, point; truth=nothing, t0=0, tag="")

Forecast quantiles at the 23 levels used for WIS, with truth alongside when
it is known.
"""
function save_forecast(dir, curves::AbstractMatrix, point::AbstractVector;
                       truth=nothing, t0::Real=0, tag::String="")
    _ensure(dir)
    qs = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400,
          0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
          0.950, 0.975, 0.990]
    H = size(curves, 1)
    Q = [quantile(@view(curves[h, :]), q) for h in 1:H, q in qs]
    cols = Any[collect(1:H) .+ t0, point]
    header = ["t", "point"]
    if truth !== nothing
        push!(cols, collect(float.(truth))); push!(header, "truth")
    end
    M = hcat(cols..., Q)
    append!(header, ["q" * string(q) for q in qs])
    return _writecsv(joinpath(dir, "forecast$(isempty(tag) ? "" : "-" * tag).csv"),
                     header, M)
end

"""
    save_performance(dir, filename, rows)

`rows` is a vector of NamedTuples with identical fields.
"""
function save_performance(dir, filename::AbstractString, rows::Vector{<:NamedTuple})
    _ensure(dir)
    isempty(rows) && return nothing
    header = String.(collect(keys(first(rows))))
    M = permutedims(hcat([collect(values(r)) for r in rows]...))
    return _writecsv(joinpath(dir, filename), header, M)
end

"""
    save_settings(dir, fit; extra...)

Plain-text record of everything needed to reproduce the run.
"""
function save_settings(dir, fit::RenewalFit; extra...)
    _ensure(dir)
    path = joinpath(dir, "settings.txt")
    open(path, "w") do io
        println(io, "RenewFit run settings")
        println(io, "generated: ", string(Dates_now()))
        println(io, "")
        println(io, "generation interval")
        println(io, "  mean        : ", fit.gi.mean, " ", fit.gi.unit)
        println(io, "  sd          : ", fit.gi.sd)
        println(io, "  lags        : ", length(fit.gi))
        println(io, "  source      : ", fit.gi.source)
        fit.gi.serial && println(io,
            "  CAVEAT      : this is a SERIAL interval used as a stand-in for a",
            "\n                generation interval; they coincide only when infector",
            "\n                and infectee share an incubation distribution")
        println(io, "")
        println(io, "model choices (NOT estimates -- report these with any metric)")
        println(io, "  sigma_R     : ", fit.sigma_R, "   # day-to-day SD of log R")
        println(io, "  phi_max     : ", fit.phi_max)
        for (k, v) in pairs(extra)
            println(io, "  ", rpad(string(k), 12), ": ", v)
        end
        println(io, "")
        println(io, "fit")
        println(io, "  points      : ", length(fit.idx))
        println(io, "  phi         : ", fit.phi)
        println(io, "  converged   : ", fit.converged)
        println(io, "  R range     : ", minimum(fit.R), " to ", maximum(fit.R))
        if any(x -> abs(x - 1) > 1e-8, fit.delta)
            println(io, "  day-of-week : start_dow=", fit.start_dow)
            for (d, nm) in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
                println(io, "    ", nm, " ", round(fit.delta[d], digits=4))
            end
        else
            println(io, "  day-of-week : not modelled")
        end
        fit.phi > 0.95 * fit.phi_max && println(io,
            "  WARNING     : phi at its bound -- R_t is interpolating, do not interpret")
    end
    return path
end

Dates_now() = "run"   # avoids a Dates dependency; replaced by the caller if wanted

"""
    weekday_ratios(fit; start_dow=1)

Mean observed/expected ratio by day of week — the direct diagnostic for a
reporting cycle, and the estimate of the correction if one is needed.

`start_dow` is the weekday of the FIRST observation (1 = Monday). For the US
COVID case file, which begins 27-Feb-2020, that is 4 (Thursday).

A flat profile near 1.0 means no weekly cycle. Systematic dips and spikes mean
the cycle is being absorbed into phi as if it were random noise, which inflates
every prediction interval to cover a pattern that is actually predictable.
"""
function weekday_ratios(fit::RenewalFit; start_dow::Integer=fit.start_dow)
    yadj = [fit.y[t] / fit.delta[dow_index(t, fit.start_dow)] for t in eachindex(fit.y)]
    mu = [fit.R[i] * force_of_infection(yadj, fit.gi.w, t) *
          fit.delta[dow_index(t, fit.start_dow)] for (i, t) in enumerate(fit.idx)]
    sums = zeros(7); counts = zeros(Int, 7)
    for (i, t) in enumerate(fit.idx)
        d = mod(start_dow - 1 + (t - 1), 7) + 1
        mu[i] > 0 || continue
        sums[d] += fit.y[t] / mu[i]; counts[d] += 1
    end
    return [counts[d] > 0 ? sums[d] / counts[d] : NaN for d in 1:7]
end
