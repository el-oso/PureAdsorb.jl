# Widom insertion throughput, wall time only (no GPU event timers).
#
# Grid and repeat counts are smaller on CPU than on a GPU backend: `nsys=64` batches take
# ~9 s to assemble on this CPU and `ninsert=10^6` would run for minutes per sample, so the
# CPU grid stays at nsys=1 with ninsert up to 10^5. The grid actually used is recorded in
# `meta` so a JSON file is self-describing regardless of which host produced it.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", or "rocm".
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cpu")
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

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)

nsys_grid = is_gpu ? (1, 64) : (1,)
ninsert_grid = is_gpu ? (10^4, 10^5, 10^6) : (10^4, 10^5)
bench_seconds = is_gpu ? 30 : 10
bench_samples = is_gpu ? 10 : 5
kernel_chunk = 2^16

# Batch assembly (Ewald k-vector tables in particular) is expensive at nsys=64, so each nsys
# is built once and reused across every ninsert and by the kernel-only measurement below.
batches = Dict{Int, Any}()
get_batch!(nsys) = get!(() -> FrameworkBatch(fill(sc, nsys), ff, g, ewald), batches, nsys)

samples = []
for nsys in nsys_grid
    b = get_batch!(nsys)
    widom(b, g; T = 298.15, ninsert = 200 * nsys, backend)          # warm-up: compile
    for ninsert in ninsert_grid
        nsys * 20 <= ninsert || continue
        bm = @be widom($b, $g; T = 298.15, ninsert = $ninsert, backend = $backend) seconds = bench_seconds samples = bench_samples evals = 1
        times = [s.time for s in bm.samples]
        push!(samples, (; nsys, ninsert, backend = backend_name, times_s = times))
        println("nsys=$nsys ninsert=$ninsert median=$(median(times)) s ips=$(ninsert / median(times))")
        flush(stdout)
    end
end

# Kernel-only: one prepared chunk of poses, timing just the kernel launch and its
# synchronization, separated from the per-chunk RNG fill and host<->device copies that
# `widom` also pays.
kernel_only_s = []
for nsys in (1, 64)
    b = get_batch!(nsys)
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, kernel_chunk)
    rpos = Vector{SVector{3, Float64}}(undef, kernel_chunk)
    quat = Vector{SVector{4, Float64}}(undef, kernel_chunk)
    ΔU_h = Vector{Float64}(undef, kernel_chunk)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, nsys)
    dbatch = PureAdsorb.adapt(backend, b)
    dsys = PureAdsorb.adapt(backend, sys_of)
    drpos = PureAdsorb.adapt(backend, rpos)
    dquat = PureAdsorb.adapt(backend, quat)
    dΔU = PureAdsorb.adapt(backend, ΔU_h)
    kern = PureAdsorb.widom_kernel!(backend)
    kern(dΔU, dsys, drpos, dquat, dbatch, g; ndrange = kernel_chunk)  # warm-up: compile
    KernelAbstractions.synchronize(backend)
    bm = @be (
        kern($dΔU, $dsys, $drpos, $dquat, $dbatch, $g; ndrange = $kernel_chunk);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1
    times = [s.time for s in bm.samples]
    push!(kernel_only_s, (; nsys, chunk = kernel_chunk, backend = backend_name, times_s = times))
    println("kernel-only nsys=$nsys chunk=$kernel_chunk median=$(median(times)) s ips=$(kernel_chunk / median(times))")
    flush(stdout)
end

meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    nthreads = Threads.nthreads(), nsys_grid = collect(nsys_grid), ninsert_grid = collect(ninsert_grid),
    bench_seconds, bench_samples, kernel_chunk,
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results", "pureadsorb_widom_$(meta.host)_$(backend_name)_$(Dates.format(now(), "yyyymmdd")).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, samples, kernel_only_s), 2)
end
println("wrote $outpath")
