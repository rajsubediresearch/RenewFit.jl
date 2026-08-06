# =====================================================================
# Why does the day-of-week factor not recover an imposed cycle?
#
# Two guesses have already been wrong (R saturating on noiseless data; phi at
# its bound). Both are now ruled out by assertions that pass. This script
# stops guessing and asks the data three questions:
#
#   1. Does the SIMPLE ratio diagnostic recover the cycle on this same series?
#      weekday_ratios() worked on real COVID data. If it recovers truth here
#      but the joint fit does not, the model form is fine and the problem is
#      in the delta parameterization or the optimizer. If it ALSO fails, the
#      data construction or dow_index is wrong.
#
#   2. Is the fitted objective LOWER than the objective at the TRUE delta?
#      If the true delta scores better, the optimizer is failing. If the
#      fitted one scores better, the likelihood genuinely prefers flat delta
#      and the model or the DGP is mismatched.
#
#   3. Is the delta gradient at the starting point actually non-zero?
#      A zero gradient means the parameters are not connected to the
#      objective at all -- an indexing or wiring bug.
#
# Run:  julia --project=. examples/dow_debug.jl
# =====================================================================

using RenewFit, Statistics, Printf, Random, ForwardDiff

gi = discretize_gamma(4.0, 2.0)
truth_d = [0.85, 0.90, 1.00, 1.15, 1.15, 1.10, 0.88]
truth_d ./= exp(mean(log.(truth_d)))

# Long series with the burn-in discarded. Seeding a 9-lag kernel with only a
# few points makes the series collapse and recover, and that transient swamps
# the weekly cycle -- which is exactly what the first version of this script
# revealed (series mean 352 from a seed of 400-500, phi 1.2, cor -0.27).
seed_hist = fill(500.0, 12)
full = vcat(seed_hist,
            simulate_renewal(seed_hist, gi.w, fill(1.02, 200);
                             rng=Xoshiro(77), phi=60.0))
base = copy(full)                       # before the cycle is imposed
full = [max(full[t] * truth_d[dow_index(t, 1)], 1.0) for t in eachindex(full)]
drop = 40
y   = full[(drop + 1):end]
SD0 = dow_index(drop + 1, 1)

names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# --- 0. is the cycle actually IN the data? -------------------------------
# Measured directly as observed/clean per weekday, over the SAME window the
# fit uses, with no estimation involved. If this does not return truth_d, the
# data construction is wrong and no estimator could recover the cycle. Two
# independent estimators agreeing on a wrong answer (both put Sunday high)
# points here rather than at the fitting code.
imposed = zeros(7); cnt = zeros(Int, 7)
for i in eachindex(y)
    t = drop + i
    d = dow_index(t, 1)
    imposed[d] += full[t] / base[t]; cnt[d] += 1
end
imposed ./= max.(cnt, 1)
println("imposed ratio measured directly from the series (should equal truth):")
for d in 1:7
    @printf("  %s  truth %.3f   measured %.3f   n=%d%s\n", names[d], truth_d[d],
            imposed[d], cnt[d], abs(imposed[d] - truth_d[d]) > 0.01 ? "   <== MISMATCH" : "")
end
@printf("  cor(measured, truth) = %+.3f\n\n", cor(imposed, truth_d))
@printf("series length %d, mean %.0f\n\n", length(y), mean(y))

# --- 1. ratio diagnostic on a no-dow fit --------------------------------
f0 = fit_renewal(y, gi; sigma_R=0.01, warn_unconverged=false)
wr = weekday_ratios(f0; start_dow=SD0)
f1 = fit_renewal(y, gi; sigma_R=0.01, dow=true, start_dow=SD0, warn_unconverged=false)

println("        truth   ratio(f0)   delta(f1)")
for d in 1:7
    @printf("  %s   %6.3f     %6.3f      %6.3f\n", names[d], truth_d[d], wr[d], f1.delta[d])
end
@printf("\ncor(ratio,  truth) = %+.3f   <- does the simple diagnostic see it?\n",
        cor(wr, truth_d))
@printf("cor(delta,  truth) = %+.3f   <- does the joint fit see it?\n",
        cor(f1.delta, truth_d))
@printf("phi: no-dow %.1f, dow %.1f (bound %.0f)\n", f0.phi, f1.phi, f1.phi_max)
@printf("sd(log R): no-dow %.4f, dow %.4f\n\n",
        std(log.(f0.R)), std(log.(f1.R)))

# --- 2. warm start from the TRUE delta and re-optimize EVERYTHING ---------
#
# The first version of this check swapped in the true delta while holding log R
# frozen at values already fitted to compensate for the WRONG delta. Naturally
# truth scored worse; the comparison was meaningless. R and phi have to be free
# to re-adapt, otherwise this measures nothing.
#
# If the warm start reaches a LOWER objective than the cold fit, the cold fit
# is stuck in a local optimum and the fix is initialization. If it converges to
# the same (or worse) objective, delta and R are genuinely trading against each
# other and the model needs a constraint, not a better start.
using Optim
w = gi.w
idx = fit_window(y, w; min_force=1.0)
n = length(idx)
obj = th -> negloglik(th, y, w, idx, 0.01, 1e4; dow=true, start_dow=SD0)

theta_cold = f1.theta
theta_warm0 = vcat(theta_cold[1:(n + 1)], log.(truth_d[1:6]))
cfg = ForwardDiff.GradientConfig(obj, theta_warm0)
g!(G, th) = ForwardDiff.gradient!(G, obj, th, cfg)
res = optimize(obj, g!, theta_warm0, LBFGS(), Optim.Options(iterations=5000))
theta_warm = Optim.minimizer(res)
delta_warm = build_delta(theta_warm[(n + 2):(n + 7)])

@printf("objective  cold start : %.4f\n", obj(theta_cold))
@printf("objective  warm start : %.4f  (converged=%s)\n",
        Optim.minimum(res), Optim.converged(res))
println("\n        truth    cold    warm")
for d in 1:7
    @printf("  %s   %6.3f  %6.3f  %6.3f\n", names[d], truth_d[d], f1.delta[d], delta_warm[d])
end
@printf("cor(warm, truth) = %+.3f\n", cor(delta_warm, truth_d))
if Optim.minimum(res) < obj(theta_cold) - 1e-6
    println("  -> warm start is BETTER: the cold fit is a local optimum,")
    println("     so the fix is initialization (e.g. seed delta from weekday_ratios)")
elseif cor(delta_warm, truth_d) > 0.7
    println("  -> same objective, but warm start KEEPS the true delta: a flat")
    println("     ridge -- delta and R trade off, and the model needs a")
    println("     constraint or prior on delta, not a better start")
else
    println("  -> warm start walks AWAY from the truth to the same optimum:")
    println("     the likelihood genuinely prefers this delta on this data")
end

# --- 2b. a DGP that matches the model exactly ---------------------------
#
# The DGP above applies the cycle to the DRAWN value: obs = delta * c with
# c ~ NB(mu, phi), so Var(obs) = delta^2 (mu + mu^2/phi). The model assumes
# obs ~ NB(delta*mu, phi), i.e. Var = delta*mu + delta^2 mu^2/phi. The mean
# agrees, the variance does not, and the disagreement is delta-dependent --
# which can bias delta rather than merely phi.
#
# Here the cycle goes INSIDE the mean, so the model is exactly correct. If
# delta is recovered now, the earlier failure was the DGP's variance scaling
# and the implementation is fine. If it still fails, the problem is in the
# model or the code and none of my explanations so far are right.
function simulate_with_cycle(y0, w, R, delta, start_dow, phi, rng)
    y = collect(float.(y0))
    yadj = [y[t] / delta[dow_index(t, start_dow)] for t in eachindex(y)]
    for h in eachindex(R)
        t = length(y) + 1
        dd = delta[dow_index(t, start_dow)]
        mu = R[h] * force_of_infection(yadj, w, t) * dd
        val = float(rand(rng, nb_sampler(mu, phi)))
        push!(y, val); push!(yadj, val / dd)
    end
    return y
end

seed2 = fill(500.0, 12)
full2 = simulate_with_cycle(seed2, gi.w, fill(1.02, 200), truth_d, 1, 60.0,
                            Xoshiro(91))
y2 = full2[(drop + 1):end]
f2 = fit_renewal(y2, gi; sigma_R=0.01, dow=true, start_dow=SD0,
                 warn_unconverged=false)
println("\nDGP with the cycle INSIDE the NB mean (model exactly correct):")
println("        truth    delta")
for d in 1:7
    @printf("  %s   %6.3f  %6.3f\n", names[d], truth_d[d], f2.delta[d])
end
@printf("cor(delta, truth) = %+.3f   phi = %.1f\n", cor(f2.delta, truth_d), f2.phi)
if cor(f2.delta, truth_d) > 0.7
    println("  -> RECOVERED. The earlier failure was the DGP's variance scaling,")
    println("     not the implementation. Fix the test, keep the model.")
else
    println("  -> STILL FAILS on a DGP the model matches exactly: the problem is")
    println("     in the model or the code, and none of the explanations so far hold.")
end

# --- 3. is the delta gradient wired up at all? ---------------------------
g = ForwardDiff.gradient(obj, vcat(zeros(n), 0.0, zeros(6)))
@printf("\n|gradient| w.r.t. the 6 delta params at the start: %s\n",
        join([@sprintf("%.3g", abs(g[n + 1 + k])) for k in 1:6], ", "))
@printf("|gradient| w.r.t. log R (median over %d): %.3g\n",
        n, median(abs.(g[1:n])))
all(abs.(g[(n + 2):(n + 7)]) .< 1e-8) &&
    println("  -> delta gradients are ZERO: the parameters are not connected")
