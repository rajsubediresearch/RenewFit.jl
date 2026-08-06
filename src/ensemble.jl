# =====================================================================
# Ensembles
#
# THE DISTINCTION THAT MATTERS HERE: averaging probabilities versus averaging
# quantiles. They are not two implementations of the same idea.
#
#   * PROBABILITY averaging (a mixture) draws whole paths from member
#     distributions in proportion to their weights. If one member says the
#     peak is now and another says three weeks out, the result is genuinely
#     bimodal -- both possibilities survive.
#
#   * QUANTILE averaging (vincentization) averages the members' quantile
#     functions. The same disagreement collapses into one wide unimodal blob
#     centred between them, which is a forecast neither member believes.
#
# For multi-peak epidemic forecasting the difference is substantive, and it is
# the thing Eq. 20 of the SubEpiPredict paper does neither of cleanly. Both are
# implemented here so the choice is explicit and testable.
#
# Members are `horizon x B` predictive sample matrices, as returned by
# `forecast`. Paths are drawn WHOLE rather than resampled per horizon, so
# within-path correlation across lead times is preserved.
# =====================================================================

"""
    persistence_forecast(y, horizon; B=500, seed, window=7)

The baseline any method has to beat: carry the recent median forward with
negative binomial observation noise, dispersion estimated from recent
differences.

This lives in the package rather than in an example because a forecast method
without a naive comparison is not evaluated. On US COVID cases it beat
RenewFit at every horizon on a plateau, and from h=2 onward during growth.
"""
function persistence_forecast(y::AbstractVector, horizon::Integer;
                              B::Integer=500, seed::Integer=20260101,
                              window::Integer=7)
    yy = collect(float.(y))
    base = median(yy[max(1, end - window + 1):end])
    resid = diff(yy[max(1, end - 3window):end])
    phi = max(base^2 / max(var(resid) - base, 1.0), 0.5)
    curves = zeros(Float64, horizon, B)
    for b in 1:B
        rng = Xoshiro(hash((seed, b, :persistence)))
        for h in 1:horizon
            curves[h, b] = float(rand(rng, nb_sampler(base, phi)))
        end
    end
    return curves, fill(base, horizon)
end

_normalize(w) = collect(float.(w)) ./ sum(w)

"""
    ensemble_probability(members, weights; B=500, seed)

Mixture ensemble: each draw takes a WHOLE path from one member, chosen with
probability proportional to its weight. Preserves multimodality and
within-path correlation.
"""
function ensemble_probability(members::Vector{<:AbstractMatrix},
                              weights::AbstractVector=fill(1 / length(members),
                                                           length(members));
                              B::Integer=500, seed::Integer=20260101)
    length(members) == length(weights) ||
        error("need one weight per member")
    H = size(first(members), 1)
    all(size(m, 1) == H for m in members) ||
        error("all members must share the same horizon")
    w = _normalize(weights)
    cw = cumsum(w)
    rng = Xoshiro(seed)
    out = zeros(Float64, H, B)
    for b in 1:B
        u = rand(rng)
        m = something(findfirst(>=(u), cw), length(cw))
        col = rand(rng, 1:size(members[m], 2))
        @views out[:, b] .= members[m][:, col]
    end
    return out
end

"""
    ensemble_quantile(members, weights; nq=201)

Vincentized ensemble: averages the members' quantile functions. Returns a
`horizon x nq` matrix of quantiles, usable anywhere a sample matrix is, since
an evenly spaced quantile grid represents the same distribution.

Provided for comparison. It smears genuine disagreement into a single wide
mode — see `ensemble_probability`.
"""
function ensemble_quantile(members::Vector{<:AbstractMatrix},
                           weights::AbstractVector=fill(1 / length(members),
                                                        length(members));
                           nq::Integer=201)
    H = size(first(members), 1)
    w = _normalize(weights)
    qs = range(0.0, 1.0; length=nq)
    out = zeros(Float64, H, nq)
    for h in 1:H, (k, q) in enumerate(qs)
        out[h, k] = sum(w[i] * quantile(@view(members[i][h, :]), q)
                        for i in eachindex(members))
    end
    return out
end

"""
    stack_weights(members, truth; iters=3000, seed, B=400)

Simplex weights minimizing out-of-sample WIS of the mixture ensemble, by
random search on the simplex.

This is the principled alternative to information-criterion weights. AICc
weights rank members by FIT; stacking ranks them by the thing you actually
report. On the SubEpiPredict COVID example the Akaike weights degenerated to
w = 1.000 on the top model at all six forecast origins, so they never
ensembled at all.

Returns `(weights, WIS)`.
"""
function stack_weights(members::Vector{<:AbstractMatrix}, truth::AbstractVector;
                       iters::Integer=3000, seed::Integer=20260101,
                       B::Integer=400)
    I = length(members)
    I == 1 && return ([1.0], wis(first(members), truth))
    rng = Xoshiro(seed)
    best_w = fill(1 / I, I)
    best = wis(ensemble_probability(members, best_w; B=B, seed=seed), truth)
    # include the single-member corners, which stacking should be able to reach
    for i in 1:I
        w = zeros(I); w[i] = 1.0
        s = wis(ensemble_probability(members, w; B=B, seed=seed), truth)
        s < best && ((best, best_w) = (s, w))
    end
    for _ in 1:iters
        w = rand(rng, I)
        w ./= sum(w)
        s = wis(ensemble_probability(members, w; B=B, seed=seed), truth)
        s < best && ((best, best_w) = (s, w))
    end
    return (weights = best_w, WIS = best)
end

"""
    ensemble_point(curves)

Median path of a predictive sample matrix — the point forecast for a
combined ensemble.
"""
ensemble_point(curves::AbstractMatrix) =
    [median(@view curves[h, :]) for h in 1:size(curves, 1)]
