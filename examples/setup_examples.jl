# Run once:  julia --project=examples examples/setup_examples.jl
#
# Plots is a heavy dependency and is deliberately kept OUT of the package: the
# core stays low-dependency and installable anywhere, while plotting lives here
# in its own environment. Same split GrowthFit uses.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.develop(path = normpath(joinpath(@__DIR__, "..")))
Pkg.add(["Plots", "Statistics", "Printf"])
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "1"
Pkg.precompile()
Pkg.status()
