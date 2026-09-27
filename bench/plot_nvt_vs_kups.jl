# Draws bench/results/nvt_vs_kups.png: NVT cost per move against chain count, both
# precisions, against kUPS's own single-system rate as a horizontal reference — no benchmark
# runs here, only the committed JSON files named below.
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/nvt_vs_kups.png julia --project=bench bench/plot_nvt_vs_kups.jl
#
# kUPS's rate comes from the same OLS fit as plot_headtohead.jl: t = intercept + nmoves/rate
# over the median time at each nmoves, fit at nsys=1 (kUPS's peak on this card; batching does
# not raise it — see docs/src/benchmarks.md).
using CairoMakie, JSON, Statistics

resultsdir = joinpath(@__DIR__, "results")

function fit_rate(nmoves::AbstractVector, t::AbstractVector)
    xm, ym = mean(nmoves), mean(t)
    slope = sum((nmoves .- xm) .* (t .- ym)) / sum(abs2, nmoves .- xm)
    return 1 / slope
end

function per_move_us(path)
    results = JSON.parsefile(path)["results"]
    nsys = Int[r["nsys"] for r in results]
    us = Float64[r["per_move_s"] * 1.0e6 for r in results]
    perm = sortperm(nsys)
    return nsys[perm], us[perm]
end

nsys_f64, us_f64 = per_move_us(joinpath(resultsdir, "pureadsorb_mcstep_neuromancer4070_cuda_f64_20260927_5ff7620.json"))
nsys_f32, us_f32 = per_move_us(joinpath(resultsdir, "pureadsorb_mcstep_neuromancer4070_cuda_f32_20260927_5ff7620.json"))

kt = JSON.parsefile(joinpath(resultsdir, "kups_nvt_timing_neuromancer4070_f64_20260927.json"))
kt_samples = kt["samples"]
kups_rate = fit_rate(
    Float64[s["nmoves"] for s in kt_samples],
    Float64[median(Float64.(s["times_s"])) for s in kt_samples],
)
kups_us = 1.0e6 / kups_rate

fig = Figure(size = (800, 480))
ax = Axis(
    fig[1, 1]; xlabel = "chains (independent systems)", ylabel = "cost per move (µs)",
    xscale = log10, yscale = log10, title = "NVT cost per move vs chain count, RTX 4070",
)
scatterlines!(ax, nsys_f64, us_f64; label = "PureAdsorb Float64", color = Makie.wong_colors()[1], marker = :circle)
scatterlines!(ax, nsys_f32, us_f32; label = "PureAdsorb Float32", color = Makie.wong_colors()[2], marker = :utriangle)
hlines!(ax, [kups_us]; color = Makie.wong_colors()[6], linestyle = :dash, label = "kUPS, 1 system (peak on this card)")
axislegend(ax; position = :rt)

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "nvt_vs_kups.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
