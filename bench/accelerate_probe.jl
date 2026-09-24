# Task B, item 3: is Apple Accelerate's BLAS fast at the GEMM shape the all-pairs distance-matrix
# formulation would actually use? `‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b` turns pose-vs-framework-atom
# distances into a GEMM of shape (npose × 3) * (3 × natoms) — K=3, very thin. A large square GEMM
# is not that shape and is measured here only as a reference for how far a thin-K GEMM falls
# short of a BLAS's usual advertised throughput.
#
# Route to Accelerate: AppleAccelerate.jl v0.7.0 registers Accelerate as a second
# libblastrampoline (LBT) backend (`AppleAccelerate.load_accelerate()`); OpenBLAS is the default
# LBT target until then, so the same `A * B` call is benchmarked before and after that one call to
# compare backends without duplicating any GEMM-calling code. `BLAS.set_num_threads` pins OpenBLAS
# in the pre-load baseline; `AppleAccelerate.set_num_threads` toggles Accelerate's own threading
# (a single/multi toggle, not an exact count — Accelerate picks the actual thread count in
# multi-threaded mode) after the switch. Both are pinned to the 8-thread cap this benchmark run
# uses throughout, and reported explicitly below.
using LinearAlgebra, Chairmarks, Statistics, JSON, Dates
BLAS.set_num_threads(8)
println("BLAS config before loading Accelerate: ", BLAS.get_config())
flush(stdout)

natoms = 3078   # measured framework atom count, RUBTAK 3x3x3 (bench/widom_decompose.jl)
npose_grid = (1024, 16384, 65536)
square_n = 1024

function gemm_gflops(::Type{T}, M, K, N; seconds = 5, samples = 7) where {T}
    A = rand(T, M, K)
    B = rand(T, K, N)
    C = Matrix{T}(undef, M, N)
    mul!(C, A, B)   # warm-up: compile / first-call dispatch
    bm = @be mul!($C, $A, $B) seconds = seconds samples = samples evals = 1
    t = median([s.time for s in bm.samples])
    flops = 2.0 * M * N * K
    return t, flops / t / 1.0e9
end

results = Dict{String, Any}()

function run_suite(label)
    out = Dict{String, Any}()
    for T in (Float64, Float32)
        thin = Dict{String, Any}()
        for npose in npose_grid
            t, gf = gemm_gflops(T, npose, 3, natoms)
            thin[string(npose)] = (; t_s = t, gflops = gf)
            println("$label $T thin-K npose=$npose K=3 natoms=$natoms: $(round(t * 1000; digits = 3)) ms, $(round(gf; digits = 2)) GFLOP/s")
            flush(stdout)
        end
        t_sq, gf_sq = gemm_gflops(T, square_n, square_n, square_n)
        println("$label $T square n=$square_n: $(round(t_sq * 1000; digits = 3)) ms, $(round(gf_sq; digits = 2)) GFLOP/s")
        flush(stdout)
        out[string(T)] = (; thin, square = (; n = square_n, t_s = t_sq, gflops = gf_sq))
    end
    return out
end

println("=== OpenBLAS (default LBT backend) ===")
results["openblas"] = run_suite("openblas")

using AppleAccelerate
AppleAccelerate.load_accelerate()
AppleAccelerate.set_num_threads(8)
println()
println("BLAS config after loading Accelerate: ", BLAS.get_config())
println("AppleAccelerate.get_num_threads() = ", AppleAccelerate.get_num_threads())
flush(stdout)

println("=== Accelerate (LBT-forwarded) ===")
results["accelerate"] = run_suite("accelerate")

# Item 4 (partial): does Accelerate's vForce give the pair loop's transcendental work anywhere to
# go? `AppleAccelerate.jl`'s own `VMATH_COVERAGE` docstring enumerates every vForce.h entry it
# wraps (see `vmath.jl`): `erf`/`erfc` is not among them, so vForce has no vectorized erfc at
# all — our own inlined Chebyshev `pair_erfc_dev` (src/ewald.jl) has nothing to be measured
# against there; that half of item 4 is unmeasured because the routine does not exist, not because
# it was skipped. vForce's `exp` IS wrapped, so it is benchmarked here as a general proxy for what
# vForce buys on a transcendental it does cover (not a stand-in for the unmeasured erfc case).
function vecmath_gflops(::Type{T}, n) where {T}
    x = rand(T, n) .* T(5)
    y = similar(x)
    AppleAccelerate.exp!(y, x)   # warm-up
    y .= exp.(x)
    t_vv = median([s.time for s in (@be AppleAccelerate.exp!($y, $x) seconds = 5 samples = 7 evals = 1).samples])
    t_base = median([s.time for s in (@be $y .= exp.($x) seconds = 5 samples = 7 evals = 1).samples])
    return t_vv, t_base
end
vecmath = Dict{String, Any}()
n_vecmath = 1_000_000
for T in (Float64, Float32)
    t_vv, t_base = vecmath_gflops(T, n_vecmath)
    println(
        "vForce vvexp vs Base broadcast exp, $T, n=$n_vecmath: " *
            "vvexp=$(round(t_vv * 1000; digits = 3))ms Base=$(round(t_base * 1000; digits = 3))ms " *
            "speedup=$(round(t_base / t_vv; digits = 2))x"
    )
    vecmath[string(T)] = (; n = n_vecmath, t_vvexp_s = t_vv, t_base_exp_s = t_base, speedup = t_base / t_vv)
end
flush(stdout)

meta = (;
    host = "brutus", julia = string(VERSION), date = string(now()), natoms, npose_grid, square_n,
    blas_threads_pinned = 8, accelerate_threads_reported = AppleAccelerate.get_num_threads(),
    vforce_erfc = "not available: AppleAccelerate.jl's VMATH_COVERAGE lists every wrapped vForce.h " *
        "entry and erf/erfc is absent, so pair_erfc_dev has no vForce counterpart to benchmark against",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(@__DIR__, "results", "accelerate_gemm_brutus_$(Dates.format(now(), "yyyymmdd")).json")
open(outpath, "w") do io
    JSON.print(io, (; meta, results, vecmath), 2)
end
println("wrote $outpath")
