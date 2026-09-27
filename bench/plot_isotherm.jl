# Draws bench/results/isotherm_co2_rubtak.png: loading against pressure (log-pressure axis),
# with error bars, from the committed isotherm run -- no benchmark runs here.
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/isotherm_co2_rubtak.png julia --project=bench bench/plot_isotherm.jl
using CairoMakie, JSON

resultsdir = joinpath(@__DIR__, "results")
path = joinpath(resultsdir, "pureadsorb_isotherm_co2_rubtak_neuromancer_cuda_f64_20260927_873c9ed.json")
d = JSON.parsefile(path)
pressure = Float64.(d["pressure"])
loading = Float64.(d["loading"])
loading_err = Float64.(d["loading_err"])
meta = d["meta"]

fig = Figure(size = (700, 500))
ax = Axis(
    fig[1, 1]; xlabel = "pressure (Pa)", ylabel = "loading (guests/framework)", xscale = log10,
    title = "CO2 in RUBTAK 3x3x3, 298.15 K: $(meta["npress"])-point isotherm x $(meta["nreplicas"]) replicas"
)
color = Makie.wong_colors()[1]
scatter!(ax, pressure, loading; color, markersize = 8)
errorbars!(ax, pressure, loading, loading_err; color, whiskerwidth = 6)
lines!(ax, pressure, loading; color, linewidth = 1, alpha = 0.5)

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "isotherm_co2_rubtak.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
