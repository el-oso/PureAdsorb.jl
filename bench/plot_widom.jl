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
candidates = filter(p -> endswith(p, ".json") && !occursin("_scaling_", basename(p)), readdir(resultsdir; join = true))
isempty(candidates) && error("no *.json files in $resultsdir to plot")

# Files without a "samples" array (e.g. pureadsorb_widom_processcost_*.json, a single
# whole-process wall-time measurement reported only in bench/results/README.md) carry no
# throughput series to plot. Scaling results (excluded above by name; kept here as a second
# guard) use a different sample schema — keyed by nsys and chunk, not ninsert.
docs = Tuple{String, Dict}[]
for p in candidates
    d = JSON.parsefile(p)
    samples = get(d, "samples", nothing)
    (!isnothing(samples) && !isempty(samples) && all(haskey(s, "ninsert") for s in samples)) || continue
    push!(docs, (p, d))
end

# A head-to-head file splits one (host, backend, precision) series across several nsys/ninsert
# files by design, and each kUPS host contributes only one file, so both keep every point. For
# every other file, only the one with the latest `meta.date` for a given (host, backend,
# precision) is plotted — otherwise an old and a new benchmark run of the same series would be
# combined into one distribution.
is_headtohead(p) = occursin("headtohead", basename(p))
series_key(d) = (d["meta"]["host"], d["meta"]["backend"], get(d["meta"], "precision", "f64"))
latest = Dict{Tuple{String, String, String}, Tuple{String, String}}()   # key => (date, path)
for (p, d) in docs
    (is_headtohead(p) || d["meta"]["backend"] == "kups-jax") && continue
    key = series_key(d)
    date = d["meta"]["date"]
    if !haskey(latest, key) || date > latest[key][1]
        latest[key] = (date, p)
    end
end
keep_plain = Set(last.(values(latest)))
filter!(docs) do (p, d)
    is_headtohead(p) || d["meta"]["backend"] == "kups-jax" || p in keep_plain
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
for (p, d) in docs
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
        # `datalimits = extrema` clips the KDE to the sample range: without it, a group with few
        # samples (a slow GPU config that fits fewer Chairmarks reps in its time budget) can get
        # a density estimate that dips below zero, which errors on this log-scaled axis.
        violin!(ax, xs, ys; width = width * 0.9, color, label, datalimits = extrema)
    end
    axislegend(ax; position = :rt, unique = true, merge = true)
end

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "widom_throughput.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
