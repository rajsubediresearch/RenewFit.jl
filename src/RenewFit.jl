module RenewFit

using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Random: Xoshiro
using Statistics

using Distributions
using ForwardDiff
using Optim
using SpecialFunctions: loggamma

include("intervals.jl")
include("renewal.jl")
include("metrics.jl")
include("fit.jl")
include("forecast.jl")
include("ensemble.jl")
include("report.jl")

export GenerationInterval, discretize_gamma, generation_interval, rescale
export force_of_infection, fit_window, expected_incidence, simulate_renewal,
       nb_logpdf, nb_sampler, phi_from, dow_index, build_delta
export RenewalFit, fit_renewal, negloglik, laplace_cov, sample_theta, select_sigma, select_params
export forecast, fitted_curves, rt_quantiles

# ensembles
export persistence_forecast, ensemble_probability, ensemble_quantile,
       stack_weights, ensemble_point
export mae, mse, coverage, wis, performance, performance_by_horizon, WIS_ALPHAS
export output_dir, save_fit, save_rt, save_forecast, save_performance, save_settings,
       weekday_ratios
export load_series

"""
    load_series(path; column, cumulative=true)

Read a SubEpiPredict-format text file and return the incidence series.
"""
function load_series(path::AbstractString; column::Integer=1, cumulative::Bool=true)
    M = readdlm(path)
    y = ndims(M) == 1 ? Float64.(M) : Float64.(M[:, column])
    return cumulative ? vcat(y[1], diff(y)) : y
end

end # module
