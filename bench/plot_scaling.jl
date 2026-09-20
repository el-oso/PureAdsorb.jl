# Draws bench/results/widom_scaling.png: kernel insertions/s against the number of frameworks
# in one batch, from every bench/results/*_scaling_*.json — no benchmark runs here.
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/widom_scaling.png julia --project=bench bench/plot_scaling.jl
# By default, only the most recent commit's file is plotted per host/backend/precision/run
# series (an older file for the same series draws an indistinguishable line otherwise);
# PA_PLOT_ALL=1 plots every file instead.
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

docs = [(p, JSON.parsefile(p)) for p in sort(paths)]
if get(ENV, "PA_PLOT_ALL", "0") != "1"
    series_key(d) = (d["meta"]["host"], d["meta"]["backend"], d["meta"]["precision"], get(d["meta"], "run_length", 1))
    latest = Dict{Any, Tuple{String, Any}}()
    for (p, d) in docs
        key = series_key(d)
        (!haskey(latest, key) || d["meta"]["date"] > latest[key][2]["meta"]["date"]) && (latest[key] = (p, d))
    end
    docs = collect(values(latest))
end
precisions = sort(unique(d["meta"]["precision"] for (_, d) in docs))
for (p, d) in docs
    precision = d["meta"]["precision"]
    run_length = get(d["meta"], "run_length", 1)
    commit = get(d["meta"], "commit", "unknown")
    label = "$(d["meta"]["host"])/$(d["meta"]["backend"])/$precision, run=$run_length, $commit"
    points = sort(d["samples"]; by = s -> s["nsys"])
    nsys = Float64[s["nsys"] for s in points]
    ips = Float64[s["chunk"] / median(Float64.(s["times_s"])) for s in points]
    color = palette[mod1(findfirst(==(precision), precisions), length(palette))]
    linestyle = run_length == 1 ? :dash : :solid
    lines!(ax, nsys, ips; color, label, linestyle)
    scatter!(ax, nsys, ips; color, markersize = 10)
end
axislegend(ax; position = :rb)

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "widom_scaling.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
