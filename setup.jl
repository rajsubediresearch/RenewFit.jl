# Run once:  julia --project=. setup.jl
using Pkg
Pkg.activate(@__DIR__)
old = get(ENV, "JULIA_PKG_PRECOMPILE_AUTO", nothing)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
try
    Pkg.add(["Distributions", "ForwardDiff", "Optim", "SpecialFunctions",
             "DelimitedFiles", "LinearAlgebra", "Printf", "Random",
             "Statistics", "Test"])
finally
    old === nothing ? delete!(ENV, "JULIA_PKG_PRECOMPILE_AUTO") :
                      (ENV["JULIA_PKG_PRECOMPILE_AUTO"] = old)
end
Pkg.precompile()
Pkg.status()
