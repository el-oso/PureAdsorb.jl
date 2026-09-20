# Widom insertion throughput, wall time only (no GPU event timers).
#
# Grid and repeat counts are smaller on CPU than on a GPU backend: assembling an `nsys=64`
# batch is expensive enough on CPU, and `ninsert=10^6` would run for minutes per sample there,
# that the CPU grid stays at nsys=1 with ninsert up to 10^5. The grid actually used is recorded
# in `meta` so a JSON file is self-describing regardless of which host produced it.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", or "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_GRID` restricts the timing grid to one "nsys:ninsert" point (e.g. "64:1000000") instead
# of the full sweep below, for a head-to-head run against a single kUPS config
# (bench/run_headtohead.sh); `PA_REPS` then sets how many Chairmarks samples that one point
# collects (default 5).
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cpu")
precision_name = get(ENV, "PA_PRECISION", "f64")
pa_grid = get(ENV, "PA_GRID", "")
F = precision_name == "f32" ? Float32 : Float64
if backend_name == "cuda"
    using CUDA
    backend = CUDABackend()
    gpu = CUDA.name(CUDA.device())
elseif backend_name == "rocm"
    using AMDGPU
    backend = ROCBackend()
    gpu = AMDGPU.HIP.name(AMDGPU.device())
else
    backend = KernelAbstractions.CPU()
    gpu = ""
end
is_gpu = backend_name in ("cuda", "rocm")

cellwidth = parse(Float64, get(ENV, "PA_CELLWIDTH", "2"))

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))

if isempty(pa_grid)
    nsys_grid = is_gpu ? (1, 64) : (1,)
    ninsert_grid = is_gpu ? (10^4, 10^5, 10^6) : (10^4, 10^5)
    bench_seconds = is_gpu ? 30 : 10
    bench_samples = is_gpu ? 10 : 5
    point_nsys, point_ninsert = nothing, nothing
else
    point_nsys, point_ninsert = parse.(Int, split(pa_grid, ":"))
    nsys_grid = (point_nsys,)
    ninsert_grid = (point_ninsert,)
    bench_samples = parse(Int, get(ENV, "PA_REPS", "5"))
    bench_seconds = 60 * bench_samples          # generous: never cut short by the time budget
end
kernel_chunk = 2^16

# Batch assembly (Ewald k-vector tables in particular) is expensive at nsys=64, so each nsys
# is built once and reused across every ninsert and by the kernel-only measurement below.
batches = Dict{Int, Any}()
get_batch!(nsys) = get!(() -> FrameworkBatch(fill(sc, nsys), ff, g, ewald; cellwidth), batches, nsys)

samples = []
for nsys in nsys_grid
    b = get_batch!(nsys)
    widom(b, g; T = F(298.15), ninsert = 200 * nsys, backend)          # warm-up: compile
    for ninsert in ninsert_grid
        nsys * 20 <= ninsert || continue
        bm = @be widom($b, $g; T = $(F(298.15)), ninsert = $ninsert, backend = $backend) seconds = bench_seconds samples = bench_samples evals = 1
        times = [s.time for s in bm.samples]
        push!(samples, (; nsys, ninsert, backend = backend_name, times_s = times))
        println("nsys=$nsys ninsert=$ninsert median=$(median(times)) s ips=$(ninsert / median(times))")
        flush(stdout)
    end
end

# Kernel-only: one prepared chunk of poses, timing just the kernel launches and their
# synchronization, separated from the per-chunk RNG fill and host<->device copies that
# `widom` also pays. Phase 0 (hardcore_kernel!) and phase 1 (widom_kernel!, over the survivors
# phase 0 finds) are timed separately, plus their sum as the kernel-path total. Skipped in
# PA_GRID mode: a head-to-head run only needs the end-to-end `widom` timing that is comparable
# to a kUPS invocation.
kernel_only_s = []
for nsys in (isempty(pa_grid) ? (1, 64) : ())
    b = get_batch!(nsys)
    N = length(g.sites)
    g_compact = PureAdsorb.Guest{F, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
    kT = F(PureAdsorb.KB * 298.15)
    rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, g_compact, kT)
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, kernel_chunk)
    rpos = Vector{SVector{3, F}}(undef, kernel_chunk)
    quat = Vector{SVector{4, F}}(undef, kernel_chunk)
    ΔU_h = Vector{F}(undef, kernel_chunk)
    flags = Vector{UInt8}(undef, kernel_chunk)
    survivor = Vector{Int32}(undef, kernel_chunk)
    run_length = PureAdsorb.default_run(kernel_chunk, nsys)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, run_length, nsys)
    dbatch = PureAdsorb.adapt(backend, b)
    dsys = PureAdsorb.adapt(backend, sys_of)
    drpos = PureAdsorb.adapt(backend, rpos)
    dquat = PureAdsorb.adapt(backend, quat)
    dΔU = PureAdsorb.adapt(backend, ΔU_h)
    dflags = PureAdsorb.adapt(backend, flags)
    dsurvivor = PureAdsorb.adapt(backend, survivor)
    drho2 = PureAdsorb.adapt(backend, rho2)
    dreach0 = PureAdsorb.adapt(backend, reach0)
    kern0 = PureAdsorb.hardcore_kernel!(backend)
    kern1 = PureAdsorb.widom_kernel!(backend)

    kern0(dflags, dsys, drpos, dquat, dbatch, g_compact, drho2, dreach0, Int32(ntypes); ndrange = kernel_chunk)
    KernelAbstractions.synchronize(backend)
    copyto!(flags, dflags)
    nsurv = 0
    for i in 1:kernel_chunk
        iszero(flags[i]) && (nsurv += 1; survivor[nsurv] = i)
    end
    copyto!(dsurvivor, survivor)
    kern1(dΔU, dsys, drpos, dquat, view(dsurvivor, 1:nsurv), dbatch, g_compact; ndrange = nsurv)  # warm-up: compile
    KernelAbstractions.synchronize(backend)

    bm0 = @be (
        kern0($dflags, $dsys, $drpos, $dquat, $dbatch, $g_compact, $drho2, $dreach0, $(Int32(ntypes)); ndrange = $kernel_chunk);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1
    bm1 = @be (
        kern1($dΔU, $dsys, $drpos, $dquat, $(view(dsurvivor, 1:nsurv)), $dbatch, $g_compact; ndrange = $nsurv);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1
    times0 = [s.time for s in bm0.samples]
    times1 = [s.time for s in bm1.samples]
    push!(
        kernel_only_s,
        (;
            nsys, chunk = kernel_chunk, nsurvivors = nsurv, run_length, backend = backend_name,
            phase0_times_s = times0, phase1_times_s = times1,
        )
    )
    rejected_frac = 1 - nsurv / kernel_chunk
    println(
        "kernel-only nsys=$nsys chunk=$kernel_chunk rejected=$(round(100 * rejected_frac; digits = 1))% " *
            "phase0_median=$(median(times0)) s phase1_median=$(median(times1)) s total_median=$(median(times0) + median(times1)) s"
    )
    flush(stdout)
end

# Remote hosts are synced without `.git`, so `git rev-parse` cannot find the commit there;
# `PA_COMMIT` lets the caller pass it in explicitly instead of falling back to "unknown".
commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end

meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nthreads = Threads.nthreads(), nsys_grid = collect(nsys_grid),
    ninsert_grid = collect(ninsert_grid), bench_seconds, bench_samples, kernel_chunk, cellwidth, commit,
)
mkpath(joinpath(@__DIR__, "results"))
outpath = if isempty(pa_grid)
    joinpath(
        @__DIR__, "results",
        "pureadsorb_widom_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
    )
else
    joinpath(
        @__DIR__, "results",
        "pureadsorb_widom_headtohead_$(meta.host)_$(precision_name)_nsys$(point_nsys)_ninsert$(point_ninsert)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
    )
end
open(outpath, "w") do io
    JSON.print(io, (; meta, samples, kernel_only_s), 2)
end
println("wrote $outpath")
