# Draws bench/results/widom_throughput.png from every bench/results/*.json — no benchmark
# runs here. Regenerate after adding a new results file with:
#   julia --project=bench bench/plot_widom.jl
using CairoMakie, JSON

resultsdir = joinpath(@__DIR__, "results")
paths = filter(p -> endswith(p, ".json"), readdir(resultsdir; join = true))
isempty(paths) && error("no *.json files in $resultsdir to plot")

records = NamedTuple[]
for p in paths
    d = JSON.parsefile(p)
    label = "$(d["meta"]["host"])/$(d["meta"]["backend"])"
    for s in d["samples"], t in s["times_s"]
        push!(records, (; nsys = s["nsys"], ninsert = s["ninsert"], label, ips = s["ninsert"] / t))
    end
end

nsys_values = sort(unique(r.nsys for r in records))
labels = sort(unique(r.label for r in records))
palette = Makie.wong_colors()

fig = Figure(size = (900, 380 * length(nsys_values)))
for (row, nsys) in enumerate(nsys_values)
    sub = filter(r -> r.nsys == nsys, records)
    ninserts = sort(unique(r.ninsert for r in sub))
    ax = Axis(
        fig[row, 1]; title = "nsys = $nsys", ylabel = "insertions / s", xlabel = "ninsert",
        yscale = log10, xticks = (eachindex(ninserts), string.(ninserts))
    )
    width = 0.8 / length(labels)
    for (li, label) in enumerate(labels)
        color = palette[mod1(li, length(palette))]
        offset = (li - (length(labels) + 1) / 2) * width
        xs = Float64[]
        ys = Float64[]
        for (ni, ninsert) in enumerate(ninserts)
            vals = [r.ips for r in sub if r.label == label && r.ninsert == ninsert]
            append!(xs, fill(ni + offset, length(vals)))
            append!(ys, vals)
        end
        isempty(xs) && continue
        violin!(ax, xs, ys; width = width * 0.9, color, label)
    end
    axislegend(ax; position = :rt, unique = true, merge = true)
end

mkpath(resultsdir)
outpath = joinpath(resultsdir, "widom_throughput.png")
save(outpath, fig)
println("wrote $outpath")
