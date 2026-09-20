# Draws bench/results/widom_throughput.png from every bench/results/*.json — no benchmark
# runs here. Regenerate after adding a new results file with:
#   julia --project=bench bench/plot_widom.jl
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/widom_throughput.png julia --project=bench bench/plot_widom.jl
#
# kUPS's times_s are whole-process wall time (Python/JAX startup, compilation, and the timed
# insertions); PureAdsorb's are Chairmarks samples, warm in-process. Plotting ninsert/t for both
# would put a per-process cost and a per-call one on the same axis, so the kUPS series instead
# uses its marginal rate: for each nsys, fit t = intercept + ninsert/rate by ordinary least
# squares over that nsys's median times across the ninsert grid, then plot
# ninsert/(t - intercept) per sample. PureAdsorb's series stays ninsert/t, labeled
# "warm in-process".
using CairoMakie, JSON, Statistics

resultsdir = joinpath(@__DIR__, "results")
paths = filter(p -> endswith(p, ".json") && !occursin("_scaling_", basename(p)), readdir(resultsdir; join = true))
isempty(paths) && error("no *.json files in $resultsdir to plot")
# Files without a "samples" array (e.g. pureadsorb_widom_processcost_*.json, a single
# whole-process wall-time measurement reported only in bench/results/README.md) carry no
# throughput series to plot. Scaling results (excluded above by name; kept here as a second
# guard) use a different sample schema — keyed by nsys and chunk, not ninsert.
filter!(paths) do p
    samples = get(JSON.parsefile(p), "samples", nothing)
    !isnothing(samples) && !isempty(samples) && all(haskey(s, "ninsert") for s in samples)
end

# Least-squares intercept of t = a + b*ninsert; the caller only needs `a`, since `ninsert - 0`
# needs no slope.
function ols_intercept(ninsert::AbstractVector, t::AbstractVector)
    length(ninsert) >= 2 || return zero(float(first(t)))
    xm, ym = mean(ninsert), mean(t)
    slope = sum((ninsert .- xm) .* (t .- ym)) / sum(abs2, ninsert .- xm)
    return ym - slope * xm
end

records = NamedTuple[]
for p in paths
    d = JSON.parsefile(p)
    # Older result files predate PA_PRECISION and are all Float64.
    precision = get(d["meta"], "precision", "f64")
    is_kups = d["meta"]["backend"] == "kups-jax"
    backend_label = is_kups ? "kUPS (JAX), marginal" : "$(d["meta"]["backend"]), warm in-process"
    label = "$(d["meta"]["host"])/$(backend_label)/$precision"
    samples = d["samples"]
    for nsys in unique(s["nsys"] for s in samples)
        sub = filter(s -> s["nsys"] == nsys, samples)
        intercept = if is_kups
            ols_intercept(
                Float64[s["ninsert"] for s in sub],
                Float64[median(Float64.(s["times_s"])) for s in sub]
            )
        else
            0.0
        end
        for s in sub, t in s["times_s"]
            push!(records, (; nsys, ninsert = s["ninsert"], label, ips = s["ninsert"] / (t - intercept)))
        end
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

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "widom_throughput.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
