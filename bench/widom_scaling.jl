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

cellwidth = parse(Float64, get(ENV, "PA_CELLWIDTH", "2"))

data = joinpath(pkgdir(PureAdsorb), "data")
fw = read_cif(joinpath(data, "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(data, "trappe.yaml"); T = F)
g = read_guest(joinpath(data, "co2.yaml"), ff; T = F)
b1 = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = F(12), precision = F(1.0e-6)); cellwidth)
natoms, nk = length(b1.positions), length(b1.ks)
ncellgrid = length(b1.cell_offsets)   # one system's prod(ncells) + 1 local cell offsets
Nsites = length(g.sites)
ntypes = size(b1.sigma, 1)

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
    na, nkv, ncg = length(b.positions), length(b.ks), length(b.cell_offsets)
    Int64(n) * max(na, nkv, ncg) <= typemax(Int32) || throw(ArgumentError("nsys=$n overflows the Int32 offsets"))
    offsets(len) = adapt(backend, Int32[Int32(i * len) for i in 0:n])
    rep(v) = adapt(backend, repeat(collect(v), n))
    return FrameworkBatch(
        tile(backend, b.positions, n), tile(backend, b.types, n), tile(backend, b.charges, n), offsets(na),
        rep(b.cells), rep(b.invcells), rep(b.volumes), rep(b.alphas),
        tile(backend, b.ks, n), tile(backend, b.kprefactor, n), tile(backend, b.Shost, n), offsets(nkv),
        rep(b.constant_offset), rep(b.self_term_halfrange),
        rep(b.ncells), tile(backend, b.cell_offsets, n), offsets(ncg),
        adapt(backend, b.sigma), adapt(backend, b.epsilon),
        # compact_to_orig/guest_types/guest_types_orig/guest_sites_orig/guest_charges_orig are
        # batch-wide (sized by the number of compact types or guest sites, not per system), so
        # they are shared unchanged across every tiled copy rather than repeated.
        adapt(backend, b.compact_to_orig), adapt(backend, b.guest_types), adapt(backend, b.guest_types_orig),
        adapt(backend, b.guest_sites_orig), adapt(backend, b.guest_charges_orig),
        rep(b.bs), tile(backend, b.kmin, n),
        b.cutoff, b.ewald_cutoff, Int(n),
    )
end

bytes_per_system = natoms * (sizeof(eltype(b1.positions)) + sizeof(Int32) + sizeof(F)) +
    nk * (sizeof(eltype(b1.ks)) + sizeof(F) + sizeof(Complex{F})) +
    ncellgrid * sizeof(Int32) + 2 * sizeof(eltype(b1.ncells)) +
    sizeof(F) + Nsites * ntypes * sizeof(F)   # bs + kmin

samples = []
failed = nothing
kern0 = PureAdsorb.hardcore_kernel!(backend)
kern1 = PureAdsorb.widom_kernel!(backend)
kT = F(PureAdsorb.KB * 298.15)
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
    g_compact = PureAdsorb.Guest{F, Nsites}(g.sites, SVector{Nsites, Int}(b1.guest_types), g.charges, g.tc, g.pc, g.omega)
    # `bs`/`constant_offset` tile identically across every copy of the same framework, so the
    # rejection tables built from the tiled batch equal `b1`'s own tables tiled the same way.
    # `build_rejection_tables` only reads per-system scalars (`cells`, `ncells`, `bs`,
    # `constant_offset`, `kmin`) and atom/k-vector COUNTS via `atom_offsets`/`k_offsets`, never
    # `positions`/`types`/`charges`/`ks`/`kprefactor`/`Shost` themselves, so those stay
    # single-copy here; only the (cheap, `Int32`) offset arrays need every system's own entry.
    rb_batch = FrameworkBatch(
        b1.positions, b1.types, b1.charges, Int32[Int32(i * natoms) for i in 0:nsys],
        fill(b1.cells[1], nsys), fill(b1.invcells[1], nsys),
        fill(b1.volumes[1], nsys), fill(b1.alphas[1], nsys), b1.ks, b1.kprefactor, b1.Shost,
        Int32[Int32(i * nk) for i in 0:nsys],
        fill(b1.constant_offset[1], nsys), fill(b1.self_term_halfrange[1], nsys),
        fill(b1.ncells[1], nsys), b1.cell_offsets, b1.cellgrid_offsets, b1.sigma, b1.epsilon,
        b1.compact_to_orig, b1.guest_types, b1.guest_types_orig, b1.guest_sites_orig, b1.guest_charges_orig,
        fill(b1.bs[1], nsys), repeat(b1.kmin, nsys), b1.cutoff, b1.ewald_cutoff, nsys,
    )
    rho2, reach0, ntypes_ = PureAdsorb.build_rejection_tables(rb_batch, g_compact, kT)   # warm-up: compile
    rb_bm = @be PureAdsorb.build_rejection_tables($rb_batch, $g_compact, $kT) seconds = 10 samples = 5 evals = 1
    rejection_tables_s = median([s.time for s in rb_bm.samples])
    drho2, dreach0 = adapt(backend, rho2), adapt(backend, reach0)
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, chunk)
    rpos = Vector{SVector{3, F}}(undef, chunk)
    quat = Vector{SVector{4, F}}(undef, chunk)
    flags = Vector{UInt8}(undef, chunk)
    survivor = Vector{Int32}(undef, chunk)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, run_length, nsys)
    dsys, drpos, dquat = adapt(backend, sys_of), adapt(backend, rpos), adapt(backend, quat)
    dΔU = KernelAbstractions.allocate(backend, F, chunk)
    dflags, dsurvivor = adapt(backend, flags), adapt(backend, survivor)

    function kernel_path!()
        kern0(dflags, dsys, drpos, dquat, dbatch, g_compact, drho2, dreach0, Int32(ntypes_); ndrange = chunk)
        KernelAbstractions.synchronize(backend)
        copyto!(flags, dflags)
        nsurv = 0
        for i in 1:chunk
            iszero(flags[i]) && (nsurv += 1; survivor[nsurv] = i)
        end
        copyto!(dsurvivor, survivor)
        if nsurv > 0
            kern1(dΔU, dsys, drpos, dquat, view(dsurvivor, 1:nsurv), dbatch, g_compact; ndrange = nsurv)
            KernelAbstractions.synchronize(backend)
        end
        return nsurv
    end
    kernel_path!()   # warm-up: compile
    nsurv = kernel_path!()
    all(iszero, view(flags, 1:chunk)) || all(isfinite, view(Array(dΔU), view(survivor, 1:nsurv))) ||
        error("non-finite insertion energy at nsys=$nsys")
    bm = @be kernel_path!() seconds = 20 samples = 10 evals = 1
    times = [s.time for s in bm.samples]
    rejected_frac = 1 - nsurv / chunk
    push!(samples, (; nsys, chunk, bytes = nsys * bytes_per_system, rejected_frac, times_s = times, rejection_tables_s))
    println(
        "run=$run_length nsys=$nsys chunk=$chunk batch=$(round(nsys * bytes_per_system / 2^30; digits = 2)) GiB " *
            "rejected=$(round(100 * rejected_frac; digits = 1))% median=$(median(times)) s ips=$(chunk / median(times)) " *
            "rejection_tables_s=$rejection_tables_s"
    )
    flush(stdout)
    dbatch = nothing
    GC.gc(true)
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
    precision, nthreads = Threads.nthreads(), natoms_per_system = natoms, nk_per_system = nk,
    bytes_per_system, nsys_grid, min_chunk, run_length, cellwidth, commit,
)
# A single-size run (one process per batch size, so each size starts from an empty device memory
# pool) writes its own file; `nsys` is part of the name.
tag = (length(nsys_grid) == 1 ? "_nsys$(only(nsys_grid))" : "") * (run_length == 1 ? "" : "_run$(run_length)")
out = joinpath(
    @__DIR__, "results",
    "pureadsorb_widom_scaling_$(meta.host)_$(backend_name)_$(precision)$(tag)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(out, "w") do io
    JSON.print(io, (; meta, samples, failed), 2)
end
println("wrote $out")
