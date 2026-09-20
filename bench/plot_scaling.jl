# Draws bench/results/widom_scaling.png: kernel insertions/s against the number of frameworks
# in one batch, from every bench/results/*_scaling_*.json — no benchmark runs here.
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/widom_scaling.png julia --project=bench bench/plot_scaling.jl
using CairoMakie, JSON, Statistics

resultsdir = joinpath(@__DIR__, "results")
paths = filter(p -> occursin("_scaling_", basename(p)) && endswith(p, ".json"), readdir(resultsdir; join = true))
isempty(paths) && error("no *_scaling_*.json files in $resultsdir to plot")

fig = Figure(size = (700, 500))
ax = Axis(
    fig[1, 1]; xlabel = "frameworks in batch", ylabel = "kernel insertions / s",
    xscale = log10, yscale = log10, title = "Widom kernel throughput vs batch size"
)
palette = Makie.wong_colors()

for (i, p) in enumerate(sort(paths))
    d = JSON.parsefile(p)
    label = "$(d["meta"]["host"])/$(d["meta"]["backend"])/$(d["meta"]["precision"])"
    samples = sort(d["samples"]; by = s -> s["nsys"])
    nsys = Float64[s["nsys"] for s in samples]
    ips = Float64[s["chunk"] / median(Float64.(s["times_s"])) for s in samples]
    color = palette[mod1(i, length(palette))]
    lines!(ax, nsys, ips; color, label)
    scatter!(ax, nsys, ips; color, markersize = 10)
end
axislegend(ax; position = :rb)

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "widom_scaling.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
