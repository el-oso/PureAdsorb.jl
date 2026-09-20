# Draws bench/results/widom_vs_kups.png: marginal insertion rate, kUPS vs PureAdsorb, on the
# same RTX 3050 — no benchmark runs here, only the three committed JSON files named below.
# PA_PLOT_OUT overrides the output path, e.g. to write the docs copy:
#   PA_PLOT_OUT=docs/src/assets/widom_vs_kups.png julia --project=bench bench/plot_headtohead.jl
#
# kUPS's times_s are whole-process wall time (Python/JAX startup, compilation, and the timed
# insertions); PureAdsorb's are Chairmarks samples, warm in-process. Comparing ninsert/t directly
# would put a per-process cost against a per-call one on the same footing, so both series use the
# same fit instead: for each nsys, t = intercept + ninsert/rate by ordinary least squares over
# the median time at each ninsert, and the bar is the fitted rate.
using CairoMakie, JSON, Statistics

resultsdir = joinpath(@__DIR__, "results")

function fit_rate(ninsert::AbstractVector, t::AbstractVector)
    xm, ym = mean(ninsert), mean(t)
    slope = sum((ninsert .- xm) .* (t .- ym)) / sum(abs2, ninsert .- xm)
    return 1 / slope
end

function rates_by_nsys(path)
    samples = JSON.parsefile(path)["samples"]
    out = Dict{Int, Float64}()
    for nsys in sort(unique(s["nsys"] for s in samples))
        sub = filter(s -> s["nsys"] == nsys, samples)
        ninsert = Float64[s["ninsert"] for s in sub]
        t = Float64[median(Float64.(s["times_s"])) for s in sub]
        out[nsys] = fit_rate(ninsert, t)
    end
    return out
end

kups = rates_by_nsys(joinpath(resultsdir, "kups_widom_timing_neuromancer_f64_20260919.json"))
pa_f64 = rates_by_nsys(joinpath(resultsdir, "pureadsorb_widom_neuromancer_cuda_f64_20260920_e903fac.json"))
pa_f32 = rates_by_nsys(joinpath(resultsdir, "pureadsorb_widom_neuromancer_cuda_f32_20260920_e903fac.json"))

bars = [
    ("kUPS Float64, 1 fw", kups[1], :kups),
    ("kUPS Float64, 4 fw", kups[4], :kups),
    ("PureAdsorb Float64, 1 fw", pa_f64[1], :pa64),
    ("PureAdsorb Float64, 64 fw", pa_f64[64], :pa64),
    ("PureAdsorb Float32, 1 fw", pa_f32[1], :pa32),
    ("PureAdsorb Float32, 64 fw", pa_f32[64], :pa32),
]

colors = Dict(:kups => Makie.wong_colors()[6], :pa64 => Makie.wong_colors()[1], :pa32 => Makie.wong_colors()[2])

fig = Figure(size = (800, 420))
ax = Axis(
    fig[1, 1]; xlabel = "marginal insertions / s (OLS fit)", xscale = log10,
    yticks = (eachindex(bars), first.(bars)), title = "Widom marginal insertion rate: PureAdsorb vs kUPS, RTX 3050"
)
rates = [b[2] for b in bars]
barplot!(ax, eachindex(bars), rates; direction = :x, color = [colors[b[3]] for b in bars])
for (i, r) in enumerate(rates)
    text!(ax, r, i; text = "$(round(Int, r))", align = (:left, :center), offset = (6, 0))
end
xlims!(ax, 1.0e3, maximum(rates) * 3)

outpath = get(ENV, "PA_PLOT_OUT", joinpath(resultsdir, "widom_vs_kups.png"))
mkpath(dirname(outpath))
save(outpath, fig)
println("wrote $outpath")
