# =====================================================================
# Generation-interval distributions
#
# This is the whole point of a "known disease" toolbox: the generation
# interval is the parameter that is hardest to identify from an epidemic
# curve, and supplying it from the literature is what makes everything
# downstream well-conditioned.
#
# LITERATURE VALUES (retrieved 2026-08; check for newer estimates before use)
#
# COVID-19, ancestral lineage. Estimates vary substantially by study, method
# and population, so treat the default as one defensible choice rather than a
# settled value:
#   * Ganyani et al. 2020, symptom-onset data: mean 3.95 d (SD 1.51) from
#     Tianjin; mean 5.20 d from Singapore
#   * UK household study (Hart et al.): mean 4.2 d (95% CrI 3.3-5.3), SD 4.9
#   * South Korea infector-infectee pairs: mean 4.3 d (95% CI 4.2-4.4)
#   * Greater Toronto, gamma fit: mean 3.99 d, SD 2.96
#   Progressive SHORTENING with later variants is well documented, so a
#   2020-era value must not be reused for Omicron-era data.
#
# Mpox, 2022 global outbreak (clade IIb):
#   * US, 12 jurisdictions, 57 case pairs: mean serial interval 8.5 d
#     (95% CrI 7.3-9.9). NOTE: no SD is reported for this estimate. The 3.0 d
#     used in the default below is a PLAUSIBLE VALUE CHOSEN HERE, not a
#     literature figure -- run examples/run_gi_sensitivity.jl before relying
#     on anything that depends on it.
#   * Pooled 2022 estimate: 8.7 d, vs 14.2 d historically
#   * Italy, 16 pairs: mean GENERATION time 12.5 d (95% CrI 7.5-17.3)
#   Note these are mostly SERIAL intervals (onset-to-onset), not generation
#   intervals (infection-to-infection). They coincide only when infector and
#   infectee share an incubation distribution. `serial=true` records that the
#   default is a serial-interval stand-in.
#
# CAVEAT that matters more than the numbers: with WEEKLY mpox data and a
# ~8.5-day interval, the discretized kernel spans barely more than one time
# step, so R_t is close to a raw growth ratio and carries little extra
# information. Daily case data is where this approach earns its keep.
# =====================================================================

"""
    GenerationInterval

Discretized generation-interval kernel. `w[s]` is the probability that a
secondary infection occurs `s` time units after the primary, `s = 1..L`.
`w` sums to 1; lag 0 is excluded (no same-step transmission).
"""
struct GenerationInterval
    w::Vector{Float64}
    mean::Float64
    sd::Float64
    unit::String
    source::String
    serial::Bool
end

Base.length(gi::GenerationInterval) = length(gi.w)

"""
    discretize_gamma(mean, sd; unit="day", maxlag=nothing, source="", serial=false)

Discretize a gamma generation interval onto the observation grid by
differencing the CDF, dropping lag 0 and renormalizing. `maxlag` defaults to
the smallest lag covering 99% of the mass.
"""
function discretize_gamma(mean_::Real, sd_::Real; unit::String="day",
                          maxlag::Union{Nothing,Integer}=nothing,
                          source::String="", serial::Bool=false)
    mean_ > 0 && sd_ > 0 || error("generation interval mean and sd must be positive")
    shape = (mean_ / sd_)^2
    scale = sd_^2 / mean_
    d = Gamma(shape, scale)
    L = maxlag === nothing ? max(2, ceil(Int, quantile(d, 0.99))) : Int(maxlag)
    w = [cdf(d, s) - cdf(d, s - 1) for s in 1:L]
    s = sum(w)
    s > 0 || error("degenerate generation interval")
    return GenerationInterval(w ./ s, float(mean_), float(sd_), unit, source, serial)
end

"""
    generation_interval(disease; kwargs...)

Sourced defaults. `:covid_ancestral` and `:mpox_2022` only — deliberately not
a long catalogue, because a wrong default is worse than no default.
"""
function generation_interval(disease::Symbol; unit::String="day", kwargs...)
    if disease === :covid_ancestral
        return discretize_gamma(3.95, 1.51; unit=unit,
            source="Ganyani et al. 2020 (Tianjin), symptom-onset data", kwargs...)
    elseif disease === :mpox_2022
        return discretize_gamma(8.5, 3.0; unit=unit,
            source="mean 8.5 d from US 12 jurisdictions, 57 case pairs; " *
                   "SD 3.0 is ASSUMED, not reported; serial interval used as " *
                   "a generation-interval stand-in",
            serial=true, kwargs...)
    else
        error("no sourced default for :$disease — build one with discretize_gamma " *
              "and cite it")
    end
end

"""
    rescale(gi, factor; unit)

Convert a kernel to a coarser grid, e.g. daily to weekly with `factor = 7`.
Warns when the rescaled mean is under two time steps, where R_t degenerates
toward a plain growth ratio.
"""
function rescale(gi::GenerationInterval, factor::Real; unit::String="step")
    m, s = gi.mean / factor, gi.sd / factor
    m < 2 && @warn "generation interval spans < 2 time steps after rescaling " *
                   "(mean $(round(m, digits=2))); R_t will carry little information " *
                   "beyond a growth ratio"
    return discretize_gamma(m, s; unit=unit, source=gi.source, serial=gi.serial)
end
