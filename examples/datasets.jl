# =====================================================================
# Dataset configurations
#
# One place to describe a series, so the analysis scripts stay generic. Adding
# a dataset means adding an entry here, not copying a script.
#
# Every field that differs between diseases lives here: the generation
# interval and its time unit, the forecast horizon, which origins belong to
# which epidemic phase, and the sigma_R grid (which has to be coarser for
# short weekly series than for 75 daily points).
# =====================================================================

using RenewFit

const DATA_ROOT = normpath(joinpath(@__DIR__, "..", "input"))

"""
    dataset(name)

Returns a NamedTuple describing a series:

  `y`            full incidence series (calibration + evaluation)
  `gi`           generation interval on the OBSERVATION grid
  `horizon`      forecast horizon in time steps
  `phases`       vector of (label, origins) — kept separate because epidemic
                 phase changes the answer completely (see README)
  `sigma_grid`, `damp_grid`  selection grids
  `min_inner`    shortest inner series that can support weight stacking
  `start_dow`    weekday of the first observation, or `nothing` when the
                 grid is not daily
  `unit`         "day" or "week", for axis labels
"""
function dataset(name::AbstractString)
    if name == "covid_usa"
        y = load_series(joinpath(DATA_ROOT,
            "cumulative-daily-coronavirus-cases-USA-05-25-20.txt"); column=52)
        return (name = name, y = y, unit = "day",
                gi = generation_interval(:covid_ancestral),
                horizon = 14,
                # 45-75 is a plateau at 20-30k/day, where persistence is close
                # to optimal; 25-40 is the growth phase. Pooling them hides the
                # only interesting structure.
                phases = [("growth", [25, 30, 35, 40]),
                          ("plateau", [45, 50, 55, 60, 65, 70, 75])],
                sigma_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                damp_grid = [0.5, 0.7, 0.9, 1.0],
                min_inner = 20, start_dow = 4,   # series begins Thu 27-Feb-2020
                # Published ancestral-lineage estimates, each labelled with its
                # source so the sensitivity table reports real alternatives
                # rather than arbitrary perturbations.
                gi_grid = [(3.95, 1.51, "Ganyani 2020 Tianjin"),
                           (5.20, 1.72, "Ganyani 2020 Singapore"),
                           (4.20, 4.90, "UK household (Hart)"),
                           (4.30, 2.00, "South Korea pairs"),
                           (3.99, 2.96, "Toronto gamma fit"),
                           (4.60, 2.30, "interpolated"),
                           (5.50, 2.30, "Ferretti-type")])

    elseif name == "mpox_usa"
        y = load_series(joinpath(DATA_ROOT,
            "cumulative-weekly-monkeypox-cases-USA-07-20-2023.txt"))
        # ~8.5-day interval on a WEEKLY grid: the kernel spans barely one time
        # step, so R_t is close to a raw growth ratio. rescale warns about it.
        gi = rescale(generation_interval(:mpox_2022), 7; unit="week")
        return (name = name, y = y, unit = "week", gi = gi,
                horizon = 4,
                # the US mpox outbreak peaked around week 12 and then collapsed,
                # so there is no usable growth phase at origins long enough to
                # fit (select_params needs > horizon + 10 points).
                phases = [("post-peak", [16, 18, 20]),
                          ("tail", [26, 32, 38])],
                sigma_grid = [0.02, 0.05, 0.1, 0.2, 0.4],
                damp_grid = [0.5, 0.7, 0.9, 1.0],
                min_inner = 12, start_dow = nothing,
                # NOTE these are on the WEEKLY grid (days / 7). The spread here
                # is not a small perturbation: 8.5 d is a SERIAL interval and
                # 12.5 d is an estimated GENERATION time, so this sweep is
                # partly over WHICH QUANTITY is being used, not just its value.
                gi_grid = [(8.5 / 7, 3.0 / 7, "US serial interval 8.5d"),
                           (8.7 / 7, 3.0 / 7, "pooled serial 8.7d"),
                           (12.5 / 7, 4.0 / 7, "Italy generation time 12.5d"),
                           (14.2 / 7, 5.0 / 7, "historical clade I 14.2d")])
    else
        error("unknown dataset '$name'. Known: covid_usa, mpox_usa")
    end
end

"""Dataset name from the command line, defaulting to covid_usa."""
cli_dataset() = isempty(ARGS) ? "covid_usa" : String(ARGS[1])
