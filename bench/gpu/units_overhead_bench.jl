# Measures the absolute per-call overhead of `mc_exchange!`'s Unitful fugacity boundary
# (`src/units.jl`) on CUDA, RUBTAK 3x3x3 + CO2, nsys=1: the wrapped method strips
# `fugacity::AbstractVector{<:Unitful.Pressure}` to a bare `Vector{T}` (`T.(ustrip.(u"Pa",
# fugacity))`, one allocation) and forwards to the same bare-Float method the exchange workgroup
# fan-out already brought down to roughly 200-300 us/call at nsys=1 (`bench/results/
# pureadsorb_exchange_workgroup_*.json` measures `mc_insert!`/`mc_delete!` individually;
# `mc_exchange!`, timed here, adds one `rand(rng, Bool)` and calls whichever one that draw picks)
# -- a fixed per-call cost that was <0.2% of an 18 ms call could be a much larger fraction of a
# few-hundred-microsecond one, so this is measured directly rather than assumed small.
#
# Every call is issued from inside a FUNCTION (`run_one`), never from top-level script globals: a
# first attempt that closed over top-level (non-`const`) globals measured ~1.6-2.5 ms/call, since
# Julia cannot specialize a closure's dispatch on a non-`const` global's runtime type.
#
# `mc_exchange!` itself needed one more fix before it would measure cleanly: it draws a fresh coin
# and calls `mc_insert!` XOR `mc_delete!` per call, and those two paths compile SEPARATE
# `KernelAbstractions` kernels (including `apply_exchange_sk_kernel!`'s `Val(true)`/`Val(false)`
# specializations) that a single untimed "compile" call, which only exercises whichever side that
# one coin flip picks, does not both warm. Measured directly: with only `mc_exchange!`'s own coin
# flip to rely on, EVERY one of ten `@be` samples read 1.9-3.2 ms, uniformly -- not the "one slow
# outlier" a single missed compile would give -- because a 0.5 s wall-clock warm-up has, on
# average, only a few calls' worth of chances to land the coin flip the untimed compile call
# missed, and few enough that it can still be missing when `@be` starts timing. Explicitly calling
# `mc_insert!` and `mc_delete!` once each (untimed) before the warm-up loop fixed it completely:
# 167 us median, in line with the ~189/136 us `mc_insert!`/`mc_delete!` baseline
# (`bench/results/pureadsorb_exchange_workgroup_*.json`).
#
# Each of the four (precision, bare-or-unitful) combinations still runs in its own PROCESS
# (`run.sh` below): measuring two variants back-to-back in one process, even from inside a
# function with both paths pre-compiled, was tried first and gave an inflated first-measured
# reading, since the SECOND variant inherits clock-boost momentum from the first one's own 10 s of
# continuous benchmarking. A fresh process per variant removes that ordering artifact. Each still
# warms on WALL-CLOCK time (one call to compile, then ~0.5 s of further calls, never a fixed call
# count): the card idles at 210 MHz against a 3105 MHz boost and needs 100-150 sustained calls to
# reach it.
using PureAdsorb, StaticArrays, Chairmarks, CUDA, LinearAlgebra, Random, Statistics, Unitful, JSON, Dates
BLAS.set_num_threads(1)

function run_one(::Type{F}, variant::AbstractString) where {F}
    backend = CUDABackend()
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
    kT = F(PureAdsorb.KB * 298.15)

    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = SystemState(b, g, [10], ff; T = F(298.15), seed = 1, capacities = [40])
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_c = PureAdsorb.compact_guest(b, g)

    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    db = PureAdsorb.adapt(backend, b)
    dst = PureAdsorb.adapt(backend, st)
    dp = PureAdsorb.adapt(backend, p)
    dq = PureAdsorb.adapt(backend, q)
    ws = PureAdsorb.MoveWorkspace(F, 1, PureAdsorb.default_nblocks_per_chain(F, 1); backend)
    occ0 = PureAdsorb.adapt(backend, st.occupancy)   # snapshot: pins occupancy across every call below

    fugacity = variant == "unitful" ? F[2.0e4] .* u"Pa" : F[2.0e4]
    rng = Xoshiro(1)
    # Resets occupancy after every call, exactly as `exchange_workgroup_bench.jl`'s own `call()`
    # does: an unconstrained run of thousands of calls in 0.5 s can otherwise walk occupancy to
    # `capacity` and trip `mc_insert!`'s fail-fast `capacity_hits` throw partway through the loop.
    call() = begin
        r = PureAdsorb.mc_exchange!(rng, ws, db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
        copyto!(dst.occupancy, occ0)
        r
    end

    # Compiles BOTH of `mc_exchange!`'s branches before any wall-clock warm-up starts (see this
    # file's header): `fugacity`'s own bare-vs-Unitful element type never reaches either kernel
    # (both dispatch to the same bare-Float `mc_insert!`/`mc_delete!` methods), so compiling with
    # the bare vector here warms the identical kernels the `variant == "unitful"` call below runs.
    fugacity_bare = F[2.0e4]
    PureAdsorb.mc_insert!(ws, db, dst, guest_c, guest_types, dp, dq, fugacity_bare, kT; backend)
    copyto!(dst.occupancy, occ0)
    PureAdsorb.mc_delete!(ws, db, dst, guest_c, guest_types, dp, dq, fugacity_bare, kT; backend)
    copyto!(dst.occupancy, occ0)
    CUDA.synchronize()

    call()
    CUDA.synchronize()
    t_warmup = time()
    while time() - t_warmup < 0.5
        call()
        CUDA.synchronize()
    end

    bm = @be call() seconds = 10 samples = 10 evals = 1
    times_s = [s.time for s in bm.samples]
    per_call = median(times_s)
    println("$F/$variant nsys=1: $(per_call * 1.0e6) us/call (median of $(length(times_s)) samples)")
    return times_s, per_call
end

precision = get(ARGS, 1, "f64")
variant = get(ARGS, 2, "bare")
F = precision == "f32" ? Float32 : Float64
times_s, per_call = run_one(F, variant)

commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu = CUDA.name(CUDA.device()),
    backend = "cuda", precision, variant, nsys = 1, commit,
    description = "Job 1 (units rollout): mc_exchange!'s Unitful-fugacity boundary vs the bare-Float " *
        "call it forwards to, RUBTAK 3x3x3 + CO2, nsys=1, 10 initial guests, capacity 40. One process " *
        "per (precision, variant) -- see this file's header for why.",
)
mkpath(joinpath(@__DIR__, "..", "results"))
outpath = joinpath(@__DIR__, "..", "results", "pureadsorb_units_overhead_neuromancer4070_cuda_$(precision)_$(variant)_$(commit).json")
open(outpath, "w") do io
    JSON.print(io, (; meta, times_s, per_call_s = per_call))
end
println("wrote $outpath")
