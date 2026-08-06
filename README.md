# RenewFit.jl

Renewal-equation R(t) estimation and short-horizon forecasting for **known
diseases**, where the generation interval can be supplied from the literature
rather than estimated from the epidemic curve.

Sibling to `GrowthFit.jl` (phenomenological) and `MechFit.jl` (mechanistic).
This is the semi-mechanistic one.

```
I(t) = R(t) · Σ_s w(s) · I(t−s)
```

Multiple peaks emerge whenever R crosses 1. There is no mixture, no number of
components to select, no onset times, no threshold parameter — so none of the
combinatorial search problems that dominate n-sub-epidemic style frameworks
exist here. It is a convolution, not an ODE: no solver, no stiffness.

## Install and run

```
julia --project=. setup.jl                       # once
julia --project=. test/runtests.jl
julia --project=. examples/covid_cases.jl

julia --project=examples examples/setup_examples.jl   # plotting env, once
julia --project=examples examples/covid_report.jl
```

Dependencies are deliberately minimal: `NLopt`-free, just `Optim`,
`ForwardDiff`, `Distributions`, `SpecialFunctions` and stdlibs. `Plots` is
**not** a package dependency — plotting lives in `examples/` with its own
`Project.toml`.

## What it does well, and what it does not

Measured on US daily COVID-19 **cases**, Feb–May 2020, against a persistence
baseline (carry the recent median forward with NB noise), with parameters
selected inside the training data at every origin.

| | result |
|---|---|
| R(t) estimation | **calibration coverage 95.9%** vs nominal 95% |
| Fit speed | ~1.2 s for 75 points, deterministic |
| Forecast, growth phase | beats persistence at **h = 1** (+22.6%); loses from h = 2 |
| Forecast, plateau | loses at every horizon tested (h = 1…14) |
| Forecast intervals | **too wide** — 100% coverage at nearly every origin |

The honest scope: **R(t) estimation with well-calibrated intervals, and
one-day-ahead forecasting during epidemic growth.** Beyond that, extrapolating
R(t) is weak, which is a known property of the approach and not a defect of
this implementation. On a plateau, "carry the last value forward" is very hard
to beat and this model does not beat it.

Persistence, however, is *sharp but badly miscalibrated* during growth —
coverage ~25% against nominal 95%. RenewFit's advantage there is calibration as
much as accuracy, which usually matters more for decision support.

`examples/covid_ensemble.jl` combines the two, which is the natural response to
two methods with complementary strengths.

## Design decisions worth knowing

**Cases, not deaths.** The renewal equation relates infections to infections.
A death series needs deconvolving by an infection-to-death delay first;
applying a generation interval directly to deaths silently mis-times R(t).

**Deterministic.** Smooth objective, ForwardDiff gradients, L-BFGS. A test
asserts that two fits with identical inputs give bit-identical R.

**Laplace, not bootstrap.** Uncertainty comes from the Hessian at the mode, not
from hundreds of refits. Cheaper, and on this model class better calibrated.

**`sigma_R` is a modelling choice, not an estimate.** log R has one parameter
per observation, so the model is saturated and only the random-walk prior stops
it interpolating. `sigma_R` is the day-to-day SD of log R — interpretable, so a
wrong value is visible. Use `select_params` to choose it (and `damp`) by
held-out WIS, and report the value you used. `save_settings` writes it to disk
alongside every result for exactly this reason.

**`damp` controls R extrapolation.** `log R(T+h) = damp^h · log R(T)`. An
undamped random walk overflows to infinity within ~20 steps — that is not a
figure of speech, it crashed the first test run.

**φ is bounded.** Unbounded it ran to 3.7e8 on real data, flattening the
objective. A warning fires when it pins at its bound, which means R is
interpolating and the fit should not be interpreted.

## Ensembles

Two combination rules, and the difference matters:

- `ensemble_probability` — mixture. Draws whole paths from members in
  proportion to their weights, so genuine disagreement stays **bimodal**.
- `ensemble_quantile` — vincentization. Averages quantile functions, collapsing
  disagreement into one wide mode centred between the members: a forecast
  neither member believes.

`stack_weights` selects weights by out-of-sample WIS — ranking members by the
thing you report rather than by fit. Information-criterion weights are the
common alternative and they degenerate easily; on the SubEpiPredict COVID
example Akaike weights put w = 1.000 on the top model at all six origins and so
never ensembled at all.

## Known limitations

**Day-of-week factors do not work.** `dow=true` is implemented and **off by
default**. The recovery test is marked `@test_broken`, so the suite stays green
while flagging the moment it starts working. On synthetic data with a known imposed cycle it recovers Monday
through Friday well and gets Saturday and Sunday badly wrong, reproducibly,
across two data-generating processes and different seeds. Ruled out: R
saturating, φ at its bound, insufficient burn-in, a local optimum (warm-starting
from the true δ returns to the identical optimum), and DGP variance mismatch.
The data construction is verified exact and the δ gradients are non-zero. One
untested idea: a lag-7 partial degeneracy, since δ for a weekday multiplies the
mean at day *d* and also divides the deconvolved history feeding day *d* seven
days later.

**`weekday_ratios` fails the same test** (cor −0.013 on a known cycle), so its
apparent finding on real COVID data — a ±14% weekly cycle — is **not
trustworthy** until the estimator passes a known-truth check. Diagnostic use
only, not inference.

**Forecast intervals are too wide** and the cause is unresolved. Ruled out:
φ sampling (`fix_phi`, now the default, changes little) and the random walk on
R (`rw_sd=0` changes little). Remaining suspect: the Laplace variance of the
final log R, which the random-walk prior constrains from one side only — so
forecasts launch from the least well-estimated point on the curve. Consistent
with calibration coverage (interior points) being near-nominal while forecast
coverage is not.

**Weekly data with a short generation interval degrades badly.** With mpox
(~8.5 d interval) on a weekly grid the kernel spans barely one time step, and
there is no usable `sigma_R`: every value is either too stiff (R forced
constant, cannot track a decline) or interpolating (φ at its bound). Daily data
is where this approach earns its keep. `rescale` warns about this.

**Generation-interval uncertainty is not propagated.** Every R(t) credible
interval in this package conditions on ONE fixed kernel. R depends on the GI to
first order — for a gamma kernel and growth rate `r`,
`R ≈ (1 + r·sd²/mean)^(mean²/sd²)` — so the choice of published estimate shifts
both R and the date it crosses 1. `run_gi_sensitivity.jl` refits across the
published range and reports whether that spread is larger than the interval you
would otherwise quote. Run it before treating any single-kernel interval as the
uncertainty.

**The mpox SD is assumed, not sourced.** The 8.5-day mean comes from the
literature; no SD was reported with it, and the 3.0 in `:mpox_2022` is a
plausible value chosen here. The sensitivity sweep is the way to see whether it
matters.

**Generation intervals need checking before use.** Defaults are
`:covid_ancestral` (3.95 d, SD 1.51; Ganyani et al. 2020) and `:mpox_2022`
(8.5 d, flagged `serial=true` because it is a serial-interval stand-in).
Published COVID estimates range 3.95–5.9 depending on study and method, and
generation times shortened with each variant — a 2020 value must not be reused
for Omicron-era data.

**Single-origin `covered` is binary.** `performance_by_horizon` reports whether
each observation fell inside its interval; that only becomes a coverage rate
when averaged across origins, which `examples/horizon_degradation.jl` does.

## Output layout

Results are keyed by **dataset first**, then analysis:

```
output/
  covid_usa/
    report/      fit.csv, rt.csv, forecast.csv, performance*.csv, settings.txt, *.png
    horizon/     per-lead-time degradation, growth vs plateau
    ensemble/    member vs ensemble scores, stacked weights, fans and densities
  mpox_usa/
    report/
```

Use `output_dir(ROOT, dataset, analysis)` in any new script. Flat per-analysis
folders overwrite the previous dataset's results, which you tend to discover
only when looking for a number you already computed.

## Examples

| script | what it shows |
|---|---|
| `run_report.jl` | one origin: fit, forecast, score vs persistence, CSVs, `settings.txt`, plots |
| `run_horizon.jl` | per-lead-time degradation vs persistence, by epidemic phase |
| `run_ensemble.jl` | renewal + persistence, mixture vs vincentization, stacked weights |
| `covid_cases.jl` | minimal introduction, no plotting environment needed |
| `run_gi_sensitivity.jl` | R(t) and forecasts across published generation intervals |
| `dow_debug.jl` | diagnostics for the unresolved day-of-week failure |

The three `run_*` scripts take a dataset name:

```
julia --project=examples examples/run_report.jl              # covid_usa
julia --project=examples examples/run_report.jl mpox_usa
julia --project=examples examples/run_horizon.jl mpox_usa
julia --project=examples examples/run_ensemble.jl mpox_usa
```

Datasets are configured in `examples/datasets.jl` — generation interval,
horizon, phase origins, selection grids, and the weekday of the first
observation. **Adding a series means adding an entry there, not copying a
script.**

### A note on the mpox results

`mpox_usa` is included as an honest stress test, not a showcase. The US
outbreak peaked around week 12 and collapsed, and with a ~8.5-day interval on a
weekly grid there is no usable `sigma_R`: every value is either too stiff (R
forced constant, unable to track the decline) or interpolating (φ at its
bound). Expect poor forecasts and a φ-at-bound warning. That is the method
failing where it should be expected to fail, and the diagnostics say so.

There is also no usable growth phase — `select_params` needs more than
`horizon + 10` points, and by the time the series is that long the epidemic has
already peaked. The configured phases are "post-peak" and "tail".
