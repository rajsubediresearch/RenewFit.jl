# =====================================================================
# MAP fitting of a random walk on log R_t, with Laplace uncertainty
#
# Two deliberate departures from what SubFit does:
#
#  1. GRADIENT-BASED AND DETERMINISTIC. The objective is smooth and
#     differentiable, so L-BFGS with ForwardDiff gradients converges to the
#     same answer every run. SubFit's derivative-free multistart over a
#     multimodal surface gave different optima on different runs of the same
#     code with the same seed, which invalidated a day of analysis.
#
#  2. LAPLACE INSTEAD OF BOOTSTRAP. Uncertainty comes from the Hessian at the
#     mode, not from 300 refits. It is far cheaper and, on a state-space
#     model, usually better calibrated -- SubFit's parametric bootstrap gave
#     ~28% coverage against a nominal 95%.
# =====================================================================

"""
    RenewalFit

`R` is aligned with `idx` (the fitted time indices), `phi` is the negative
binomial dispersion, `theta`/`cov` are the parameter vector and its Laplace
covariance on the log scale.
"""
struct RenewalFit
    y::Vector{Float64}
    gi::GenerationInterval
    idx::Vector{Int}
    R::Vector{Float64}
    phi::Float64
    theta::Vector{Float64}
    cov::Matrix{Float64}
    sigma_R::Float64
    phi_max::Float64
    delta::Vector{Float64}
    start_dow::Int
    nll::Float64
    objective::Float64
    converged::Bool
end

"""
    negloglik(theta, y, w, idx, sigma_R, phi_max)

Penalized negative log-likelihood. `theta = [log R (one per idx); z]`, with the
dispersion bounded as `phi = phi_max / (1 + exp(-z))`.

TWO THINGS THAT HAD TO CHANGE after the first COVID run:

1. THE PENALTY IS NOW A PROPER RANDOM-WALK PRIOR, `sum (dlogR)^2 / (2 sigma_R^2)`,
   with `sigma_R` the day-to-day SD of log R. The previous free-floating
   `smooth` coefficient had no scale relative to the likelihood: log R carries
   one parameter per observation, so the model is saturated and only the
   penalty stops it interpolating. On counts of ~25,000 the likelihood gain
   from chasing daily noise swamped `smooth=10`, giving MAE 56.6 (0.2% error),
   R up to 6.14, and a near-singular Hessian. `sigma_R` is interpretable — how
   much can R really change in a day? — so a wrong value is visible rather
   than arbitrary.

2. PHI IS BOUNDED. Unbounded, it ran to 3.7e8: the interpolating fit has no
   residual variance left, so the negative binomial collapses toward Poisson
   and beyond, flattening the objective and preventing convergence. The
   logistic transform keeps phi in (0, phi_max) smoothly, so the gradient
   stays well behaved.
"""
function negloglik(theta::AbstractVector, y::AbstractVector, w::AbstractVector,
                   idx::AbstractVector{<:Integer}, sigma_R::Real, phi_max::Real;
                   dow::Bool=false, start_dow::Integer=1)
    n = length(idx)
    logR = @view theta[1:n]
    phi = phi_from(theta[n + 1], phi_max)

    # day-of-week factors, or a flat 1 when the cycle is not modelled
    d = dow ? build_delta(@view theta[(n + 2):(n + 7)]) :
              ones(eltype(theta), 7)
    yadj = [y[t] / d[dow_index(t, start_dow)] for t in eachindex(y)]

    nll = zero(eltype(theta))
    @inbounds for i in 1:n
        t = idx[i]
        mu = exp(logR[i]) * force_of_infection(yadj, w, t) * d[dow_index(t, start_dow)]
        nll -= nb_logpdf(y[t], mu, phi)
    end
    pen = zero(eltype(theta))
    @inbounds for i in 2:n
        dd = logR[i] - logR[i - 1]
        pen += dd * dd
    end
    return nll + pen / (2 * sigma_R^2)
end

"""
    fit_renewal(y, gi; sigma_R=0.05, phi_max=1e4, min_force=1.0, maxiter=5000)

Fit log R_t by penalized maximum likelihood and attach a Laplace covariance.

`sigma_R` is THE modelling choice here — the day-to-day SD of log R. Small
values (0.01-0.05) give a smooth R and a smooth fit; large values let R chase
noise and the model saturates. It is not estimated by default because the
marginal likelihood for a random-walk variance is weakly identified on a
single series; use [`select_sigma`](@ref) to pick it by out-of-sample score
and report the value you used.
"""
function fit_renewal(y::AbstractVector, gi::GenerationInterval;
                     sigma_R::Real=0.05, phi_max::Real=1e4,
                     min_force::Real=1.0, maxiter::Integer=5000,
                     warn_unconverged::Bool=true,
                     dow::Bool=false, start_dow::Integer=1)
    yy = collect(float.(y))
    w = gi.w
    idx = fit_window(yy, w; min_force=min_force)
    length(idx) >= 5 || error("only $(length(idx)) usable time points; the series " *
                              "is too short or too sparse for this generation interval")
    sigma_R > 0 || error("sigma_R must be positive")
    dow && length(idx) < 21 && error("day-of-week factors need at least three " *
        "weeks of usable points; got $(length(idx))")
    n = length(idx)

    obj = th -> negloglik(th, yy, w, idx, sigma_R, phi_max;
                          dow=dow, start_dow=start_dow)
    theta0 = dow ? vcat(zeros(n), 0.0, zeros(6)) : vcat(zeros(n), 0.0)
    cfg = ForwardDiff.GradientConfig(obj, theta0)
    g!(G, th) = ForwardDiff.gradient!(G, obj, th, cfg)

    res = optimize(obj, g!, theta0, LBFGS(), Optim.Options(iterations=maxiter))
    theta = Optim.minimizer(res)
    conv = Optim.converged(res)
    conv || warn_unconverged && @warn "fit_renewal did not converge in $maxiter iterations"

    H = ForwardDiff.hessian(obj, theta)
    Sigma = laplace_cov(H)
    phi = phi_from(theta[n + 1], phi_max)
    phi > 0.95 * phi_max && @warn "phi is at its upper bound ($(round(phi, digits=1)) " *
        "of $phi_max): the fit has essentially no residual overdispersion left, " *
        "which usually means sigma_R is too large and R_t is interpolating the data"
    d = dow ? build_delta(theta[(n + 2):(n + 7)]) : ones(7)

    return RenewalFit(yy, gi, idx, exp.(theta[1:n]), phi, theta, Sigma,
                      float(sigma_R), float(phi_max), d, Int(start_dow),
                      negloglik(theta, yy, w, idx, 1e12, phi_max;
                                dow=dow, start_dow=start_dow),
                      Optim.minimum(res), conv)
end

"""
    select_params(y, gi; sigma_grid, damp_grid, horizon=14, ...)

Choose BOTH smoothing (`sigma_R`) and forecast damping (`damp`) by held-out
forecast score: fit on all but the last `horizon` points, forecast, and score
with WIS against what was held out.

`damp` was the larger unselected assumption once `sigma_R` was being chosen
properly -- it decides how fast R is pulled back toward 1 over the horizon,
and at 14 days it moves the forecast more than most parameters do. Leaving it
at a hand-picked 0.9 while carefully selecting sigma_R was inconsistent.

The loop is structured fit-once-per-sigma, forecast-per-damp: `damp` affects
only the forecast, not the fit, so a 5x4 grid costs 5 fits rather than 20.

Returns `(sigma, damp, WIS, table)`. The table holds every combination so the
sensitivity stays visible.
"""
function select_params(y::AbstractVector, gi::GenerationInterval;
                       sigma_grid=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                       damp_grid=[0.5, 0.7, 0.9, 1.0],
                       horizon::Integer=14, B::Integer=300,
                       seed::Integer=20260101, kwargs...)
    yy = collect(float.(y))
    length(yy) > horizon + 10 || error("series too short to hold out $horizon points")
    ytrain = yy[1:(end - horizon)]
    truth = yy[(end - horizon + 1):end]
    rows = NamedTuple[]

    for s in sigma_grid
        f = try
            fit_renewal(ytrain, gi; sigma_R=s, warn_unconverged=false, kwargs...)
        catch
            nothing
        end
        for d in damp_grid
            if f === nothing
                push!(rows, (sigma_R = s, damp = d, WIS = Inf, MAE = Inf,
                             Coverage95 = NaN, phi = NaN, maxR = NaN,
                             converged = false, at_bound = true))
                continue
            end
            curves, point = forecast(f, horizon; B=B, damp=d, seed=seed)
            p = performance(curves, point, truth)
            push!(rows, (sigma_R = s, damp = d, WIS = p.WIS, MAE = p.MAE,
                         Coverage95 = p.Coverage95, phi = f.phi,
                         maxR = maximum(f.R), converged = f.converged,
                         at_bound = f.phi > 0.95 * f.phi_max))
        end
    end

    # rows with phi against its bound are interpolating fits; exclude them from
    # selection rather than letting a degenerate fit win on score alone
    ok = [r for r in rows if !r.at_bound && isfinite(r.WIS)]
    pool = isempty(ok) ? rows : ok
    best = pool[argmin([r.WIS for r in pool])]
    return (sigma = best.sigma_R, damp = best.damp, WIS = best.WIS, table = rows)
end

"""
    select_sigma(y, gi, grid; horizon=14, kwargs...)

Smoothing only, with `damp` held fixed. Thin wrapper over
[`select_params`](@ref).
"""
function select_sigma(y::AbstractVector, gi::GenerationInterval,
                      grid=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2];
                      damp::Real=0.9, kwargs...)
    r = select_params(y, gi; sigma_grid=grid, damp_grid=[damp], kwargs...)
    return (sigma = r.sigma, WIS = r.WIS, table = r.table)
end

"""
    laplace_cov(H; jitter=1e-8)

Invert the Hessian, adding increasing ridge terms until the result is positive
definite. Returns a matrix of `NaN` if it never is — a failure that must be
visible rather than silently producing intervals from a non-PD covariance.
"""
function laplace_cov(H::AbstractMatrix; jitter::Real=1e-8)
    A = Symmetric(Matrix(H))
    for k in 0:12
        try
            C = cholesky(A + (jitter * 10.0^k) * I)
            S = Symmetric(inv(C))
            cholesky(S)                      # the covariance must be PD too
            return Matrix(S)
        catch
            continue
        end
    end
    return fill(NaN, size(H))
end

"""
    sample_theta(fit, B; seed)

Draw `B` parameter vectors from the Laplace approximation.
"""
function sample_theta(fit::RenewalFit, B::Integer; seed::Integer=20260101)
    any(isnan, fit.cov) && error("Laplace covariance is not positive definite; " *
                                 "increase `smooth` or shorten the series")
    L = cholesky(Symmetric(fit.cov)).L
    rng = Xoshiro(seed)
    return [fit.theta .+ L * randn(rng, length(fit.theta)) for _ in 1:B]
end
