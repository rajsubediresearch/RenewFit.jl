# =====================================================================
# Ensembling RenewFit with the persistence baseline
#
# WHY THIS IS THE RIGHT PAIR. The horizon sweep showed the two methods win in
# different places: RenewFit beats persistence at h=1 during epidemic growth
# (+22.6%) and loses from h=2 onward, while persistence wins at every horizon
# on a plateau. Two methods with complementary strengths and no clear
# dominance is exactly the case where combining should beat either alone.
#
# It also puts the earlier negative result in context: "renewal loses to
# persistence" is a statement about one member, not about what you should
# forecast with.
#
# Weights are stacked on out-of-sample WIS inside the TRAINING data only --
# the last `horizon` points of the training series are held out for weight
# selection, and the evaluation window is never seen.
#
# Setup once:  julia --project=examples examples/setup_examples.jl
# Run:         julia --project=examples examples/covid_ensemble.jl
# =====================================================================

using RenewFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "ensemble")

HORIZON, B, SEED = D.horizon, 500, 20260101
ORIGIN_SETS = D.phases
LABELS = ["renewal", "persistence"]
y_full, gi = D.y, D.gi

"""
    members_at(ytr, horizon; sel)

Fit both members and return their forecast sample matrices.

`sel` (sigma_R, damp) is passed in rather than re-selected, because the inner
weight-selection fit is short: a nested selection would need origin length
>= inner holdout + select_params' own ~24-point minimum, which the early
growth origins do not have.

The tradeoff, stated plainly: sigma_R and damp are chosen using the full
training series, including the inner holdout used for stacking weights. That is
mild leakage into WEIGHT selection only. The evaluation window is never touched
by either selection, so the reported scores remain out-of-sample.
"""
function members_at(ytr, horizon, sel)
    f = fit_renewal(ytr, gi; sigma_R=sel.sigma, warn_unconverged=false)
    r, _ = forecast(f, horizon; B=B, damp=sel.damp, seed=SEED)
    p, _ = persistence_forecast(ytr, horizon; B=B, seed=SEED)
    return [r, p], f
end

"""Inner holdout length: shrinks at short origins so early growth still runs."""
inner_holdout(L) = min(HORIZON, max(3, L ÷ 3))

rows, wrows = NamedTuple[], NamedTuple[]

for (phase, origins) in ORIGIN_SETS
    println("\n", "="^78)
    @printf("%s phase\n", uppercase(phase))
    println("="^78)
    println("origin  w_renew w_persist |  renewal  persist   equal  stacked   vincent")

    for L in origins
        L + HORIZON <= length(y_full) || continue
        ytr = y_full[1:L]
        truth = y_full[(L + 1):(L + HORIZON)]

        sel = select_params(ytr, gi; horizon=HORIZON, B=150, seed=SEED,
                            sigma_grid=D.sigma_grid, damp_grid=D.damp_grid)

        # --- weights chosen on a holdout INSIDE the training data ---
        ih = inner_holdout(L)
        inner = ytr[1:(end - ih)]
        inner_truth = ytr[(end - ih + 1):end]
        sw = if length(inner) >= D.min_inner
            im, _ = members_at(inner, ih, sel)
            stack_weights(im, inner_truth; iters=1500, B=300, seed=SEED)
        else
            (weights = [0.5, 0.5], WIS = NaN)   # too short to stack; fall back
        end

        # --- evaluate on the real horizon ---
        mem, f = members_at(ytr, HORIZON, sel)
        eq   = ensemble_probability(mem, [0.5, 0.5]; B=B, seed=SEED)
        st   = ensemble_probability(mem, sw.weights; B=B, seed=SEED)
        vin  = ensemble_quantile(mem, sw.weights)

        scores = Dict(
            "renewal"     => performance(mem[1], ensemble_point(mem[1]), truth),
            "persistence" => performance(mem[2], ensemble_point(mem[2]), truth),
            "equal"       => performance(eq, ensemble_point(eq), truth),
            "stacked"     => performance(st, ensemble_point(st), truth),
            "vincentized" => performance(vin, ensemble_point(vin), truth))

        @printf("  %2d     %.2f    %.2f    | %8.1f %8.1f %7.1f %8.1f %9.1f\n",
                L, sw.weights[1], sw.weights[2], scores["renewal"].WIS,
                scores["persistence"].WIS, scores["equal"].WIS,
                scores["stacked"].WIS, scores["vincentized"].WIS)

        for (k, v) in scores
            push!(rows, (phase = phase, origin = L, method = k, WIS = v.WIS,
                         MAE = v.MAE, MSE = v.MSE, Coverage95 = v.Coverage95))
        end
        push!(wrows, (phase = phase, origin = L, w_renewal = sw.weights[1],
                      w_persistence = sw.weights[2], holdout_WIS = sw.WIS,
                      sigma_R = sel.sigma, damp = sel.damp, R_end = f.R[end]))

        if L == last(origins)
            savefig(plot_ensemble_fan(mem, LABELS, st, truth; t0=L,
                        title="Ensemble vs members -- $phase, origin $L ($(D.name))"),
                    joinpath(OUT, "fan-$phase.png"))
            savefig(plot_predictive_density(mem, LABELS, st, vin; horizon=min(7, HORIZON),
                        title="Predictive density -- $phase ($(D.name))"),
                    joinpath(OUT, "density-$phase.png"))
            savefig(plot_weights(LABELS, sw.weights;
                        title="Stacked weights -- $phase, origin $L"),
                    joinpath(OUT, "weights-$phase.png"))
        end
    end

    println("\n  mean WIS by method:")
    for m in ["renewal", "persistence", "equal", "stacked", "vincentized"]
        sel = [r for r in rows if r.phase == phase && r.method == m]
        isempty(sel) && continue
        @printf("    %-12s WIS %8.1f   coverage %5.1f%%\n", m,
                mean(r.WIS for r in sel), mean(r.Coverage95 for r in sel))
    end
end

save_performance(OUT, "ensemble-performance.csv", rows)
save_performance(OUT, "ensemble-weights.csv", wrows)

println("""

How to read this
  * If `stacked` or `equal` beats BOTH members, combining is earning its keep
    and that is the method to report.
  * `vincentized` should be worse than `stacked` wherever the members
    disagree: averaging quantiles invents a middle forecast neither member
    believes. Where they agree, the two will be indistinguishable.
  * Watch the weights: stacking that collapses to w=1.00 on one member is
    telling you the other adds nothing at that origin, which is itself worth
    reporting.""")
println("\nwritten to $OUT")
