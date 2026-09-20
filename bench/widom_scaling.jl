# Kernel throughput of Widom insertion as a function of the number of frameworks in one batch,
# up to the largest batch the device holds.
#
# The batch holds `nsys` copies of RUBTAK 3×3×3 (3078 atoms each) with TraPPE CO2 as the guest.
# The single-framework tables are computed once on the CPU and tiled on the device, so neither
# the CPU-side table construction nor host memory limits the batch size. Timing covers the kernel
# launch and its synchronization for one chunk of poses; the batch stays on the device throughout.
#
# Usage: PA_BACKEND=rocm|cuda|cpu PA_PRECISION=f64|f32 [PA_NSYS="1 64 1024 ..."] \
#        julia --project=bench/gpu bench/widom_scaling.jl
# `PA_NSYS` must be increasing; the sweep stops at the first batch size that fails to allocate.
# Near the device's capacity, run one batch size per process: memory released by an earlier, smaller
# batch is not returned to the device within the process, and the next large allocation stalls
# instead of failing (observed on ROCm at 65536 frameworks after a 32768-framework batch).
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
using PureAdsorb: FrameworkBatch, adapt
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
    backend = CPU()
    gpu = ""
end
precision = get(ENV, "PA_PRECISION", "f64")
F = precision == "f32" ? Float32 : Float64
default_nsys = F === Float32 ?
    "1 64 1024 8192 32768 65536 131072 163840 180224 196608" :
    "1 64 1024 8192 32768 65536 81920 90112 98304"
nsys_grid = parse.(Int, split(get(ENV, "PA_NSYS", default_nsys)))
issorted(nsys_grid) || throw(ArgumentError("PA_NSYS must be increasing, got $nsys_grid"))
min_chunk = 2^18
# `PA_RUN` consecutive insertions go to the same framework before the assignment moves to the next
# one; 1 gives the same insertion-to-insertion interleaving `widom`'s default run length avoids.
run_length = parse(Int, get(ENV, "PA_RUN", "1"))
run_length >= 1 || throw(ArgumentError("PA_RUN must be ≥ 1, got $run_length"))

data = joinpath(pkgdir(PureAdsorb), "data")
fw = read_cif(joinpath(data, "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(data, "trappe.yaml"); T = F)
g = read_guest(joinpath(data, "co2.yaml"), ff; T = F)
b1 = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = F(12), precision = F(1.0e-6)))
natoms, nk = length(b1.positions), length(b1.ks)

# `n` consecutive copies of `v` in one device array. Each pass copies the filled prefix onto the
# next free range, so the number of device-to-device copies grows as log2(n).
function tile(backend, v::AbstractVector, n::Integer)
    m = length(v)
    d = KernelAbstractions.allocate(backend, eltype(v), m * n)
    iszero(m) && return d
    copyto!(d, 1, adapt(backend, collect(v)), 1, m)
    filled = m
    while filled < m * n
        len = min(filled, m * n - filled)
        copyto!(d, filled + 1, d, 1, len)
        filled += len
    end
    return d
end

function tiled_batch(backend, b::FrameworkBatch{T}, n::Integer) where {T}
    na, nkv = length(b.positions), length(b.ks)
    Int64(n) * max(na, nkv) <= typemax(Int32) || throw(ArgumentError("nsys=$n overflows the Int32 offsets"))
    offsets(len) = adapt(backend, Int32[Int32(i * len) for i in 0:n])
    rep(v) = adapt(backend, repeat(collect(v), n))
    return FrameworkBatch(
        tile(backend, b.positions, n), tile(backend, b.types, n), tile(backend, b.charges, n), offsets(na),
        rep(b.cells), rep(b.invcells), rep(b.volumes), rep(b.alphas),
        tile(backend, b.ks, n), tile(backend, b.kprefactor, n), tile(backend, b.Shost, n), offsets(nkv),
        rep(b.constant_offset), adapt(backend, b.sigma), adapt(backend, b.epsilon),
        b.cutoff, b.ewald_cutoff, Int(n),
    )
end

bytes_per_system = natoms * (sizeof(eltype(b1.positions)) + sizeof(Int32) + sizeof(F)) +
    nk * (sizeof(eltype(b1.ks)) + sizeof(F) + sizeof(Complex{F}))

samples = []
failed = nothing
kern = PureAdsorb.widom_kernel!(backend)
for nsys in nsys_grid
    chunk = max(min_chunk, nsys)
    local dbatch
    try
        dbatch = tiled_batch(backend, b1, nsys)
        KernelAbstractions.synchronize(backend)
    catch err
        global failed = (; nsys, bytes = nsys * bytes_per_system, error = first(sprint(showerror, err), 300))
        println("nsys=$nsys failed to allocate (~$(round(nsys * bytes_per_system / 2^30; digits = 2)) GiB): $(failed.error)")
        flush(stdout)
        break
    end
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, chunk)
    rpos = Vector{SVector{3, F}}(undef, chunk)
    quat = Vector{SVector{4, F}}(undef, chunk)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, run_length, nsys)
    dsys, drpos, dquat = adapt(backend, sys_of), adapt(backend, rpos), adapt(backend, quat)
    dΔU = KernelAbstractions.allocate(backend, F, chunk)
    kern(dΔU, dsys, drpos, dquat, dbatch, g; ndrange = chunk)      # warm-up: compile
    KernelAbstractions.synchronize(backend)
    all(isfinite, Array(dΔU)) || error("non-finite insertion energy at nsys=$nsys")
    bm = @be (
        kern($dΔU, $dsys, $drpos, $dquat, $dbatch, $g; ndrange = $chunk);
        KernelAbstractions.synchronize($backend)
    ) seconds = 20 samples = 10 evals = 1
    times = [s.time for s in bm.samples]
    push!(samples, (; nsys, chunk, bytes = nsys * bytes_per_system, times_s = times))
    println("run=$run_length nsys=$nsys chunk=$chunk batch=$(round(nsys * bytes_per_system / 2^30; digits = 2)) GiB median=$(median(times)) s ips=$(chunk / median(times))")
    flush(stdout)
    dbatch = nothing
    GC.gc(true)
end

meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision, nthreads = Threads.nthreads(), natoms_per_system = natoms, nk_per_system = nk,
    bytes_per_system, nsys_grid, min_chunk, run_length,
)
# A single-size run (one process per batch size, so each size starts from an empty device memory
# pool) writes its own file; `nsys` is part of the name.
tag = (length(nsys_grid) == 1 ? "_nsys$(only(nsys_grid))" : "") * (run_length == 1 ? "" : "_run$(run_length)")
out = joinpath(@__DIR__, "results", "pureadsorb_widom_scaling_$(meta.host)_$(backend_name)_$(precision)$(tag)_$(Dates.format(now(), "yyyymmdd")).json")
open(out, "w") do io
    JSON.print(io, (; meta, samples, failed), 2)
end
println("wrote $out")
