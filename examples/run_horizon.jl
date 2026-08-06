# =====================================================================
# Degradation with lead time, averaged over origins, per epidemic phase.
#
#   julia --project=examples examples/run_horizon.jl            # covid_usa
#   julia --project=examples examples/run_horizon.jl mpox_usa
#
# A single origin gives a binary hit/miss at each horizon, not a coverage
# rate. Averaging the same lead time across origins turns it into one.
# Phases stay separate: they behave completely differently.
# =====================================================================

using RenewFit, Statistics, Printf
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
D   = dataset(cli_dataset())
OUT = output_dir(ROOT, D.name, "horizon")
B, SEED, H = 300, 20260101, D.horizon

function aggregate(per_origin)
    hmax = maximum(r.horizon for rows in per_origin for r in rows)
    [begin
        rs = [r for rows in per_origin for r in rows if r.horizon == h]
        (horizon = h, n = length(rs), MAE = mean(r.AE for r in rs),
         RMSE = sqrt(mean(r.SE for r in rs)), WIS = mean(r.WIS for r in rs),
         width95 = mean(r.width95 for r in rs),
         coverage = 100 * mean(r.covered for r in rs))
     end for h in 1:hmax if any(r.horizon == h for rows in per_origin for r in rows)]
end

all_rows = NamedTuple[]
for (phase, origins) in D.phases
    mrows, brows, used = Vector{NamedTuple}[], Vector{NamedTuple}[], Int[]
    for L in origins
        L + H <= length(D.y) || continue
        ytr = D.y[1:L]; truth = D.y[(L + 1):(L + H)]
        sel = try
            select_params(ytr, D.gi; horizon=H, B=150, seed=SEED,
                          sigma_grid=D.sigma_grid, damp_grid=D.damp_grid)
        catch err
            @warn "origin $L skipped" err; continue
        end
        f = fit_renewal(ytr, D.gi; sigma_R=sel.sigma, warn_unconverged=false)
        c, pt = forecast(f, H; B=B, damp=sel.damp, seed=SEED)
        bc, bp = persistence_forecast(ytr, H; B=B, seed=SEED)
        push!(mrows, performance_by_horizon(c, pt, truth))
        push!(brows, performance_by_horizon(bc, bp, truth))
        push!(used, L)
    end
    isempty(mrows) && (@warn "no usable origins for $phase"; continue)

    m, b = aggregate(mrows), aggregate(brows)
    println("\n", "="^76)
    @printf("%s -- %d origins %s (%s)\n", uppercase(phase), length(used), used, D.name)
    println("="^76)
    println("   h      WIS   baseWIS    skill      MAE   width95   cov%   basecov%")
    for (rm, rb) in zip(m, b)
        @printf("  %2d %8.1f %9.1f  %+7.1f%% %8.1f %9.1f %6.1f %9.1f\n",
                rm.horizon, rm.WIS, rb.WIS, 100 * (1 - rm.WIS / rb.WIS),
                rm.MAE, rm.width95, rm.coverage, rb.coverage)
    end
    cross = findfirst(i -> m[i].WIS > b[i].WIS, eachindex(m))
    println(cross === nothing ? "  -> beats persistence at every horizon tested" :
            cross == 1 ? "  -> never beats persistence, even at h = 1" :
            "  -> beats persistence through h = $(cross-1), loses from h = $cross")

    for r in m; push!(all_rows, merge((phase = phase, model = "renewfit"), r)); end
    for r in b; push!(all_rows, merge((phase = phase, model = "persistence"), r)); end

    savefig(plot_horizon(m; baseline=b, metric=:WIS,
                title="WIS by horizon -- $phase ($(D.name))"),
            joinpath(OUT, "wis-$phase.png"))
    savefig(plot_horizon(m; baseline=b, metric=:MAE, ylabel="MAE",
                title="MAE by horizon -- $phase ($(D.name))"),
            joinpath(OUT, "mae-$phase.png"))
    savefig(plot_width_coverage(m;
                title="Width and coverage -- $phase ($(D.name))"),
            joinpath(OUT, "width-$phase.png"))
end

save_performance(OUT, "performance-by-horizon.csv", all_rows)
println("\nwritten to $OUT")
