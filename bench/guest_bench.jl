# Per-move guest-guest energy throughput, split between real-space (Lennard-Jones + Ewald
# real-space, against the host and every other guest) and reciprocal-space (the running
# structure-factor update) work — the split the Milestone B throughput amendment requires
# before task 5 commits to a move-kernel structure. Wall time only (no GPU event timers),
# `Chairmarks.@be`, `evals = 1`, matching `bench/widom_bench.jl`'s convention.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NMOVES` sets the number of independent move proposals launched per kernel call (default
# 65536, matching `widom_bench.jl`'s chunk size).
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cpu")
precision_name = get(ENV, "PA_PRECISION", "f64")
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

nmoves = parse(Int, get(ENV, "PA_NMOVES", "65536"))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "50"))
bench_seconds = 20
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
nk = length(b.ks)

st = SystemState(b, g, [nguests], ff; T = F(298.15), seed = 1)
N = length(g.sites)
guest_types = SVector{N, Int}(b.guest_types)
guest_compact = PureAdsorb.Guest{F, N}(g.sites, guest_types, g.charges, g.tc, g.pc, g.omega)

# Every proposal moves a random guest in the (single) system by a random Gaussian displacement
# (matching R3's translation shape), fixed across the whole run: this is a cost measurement, not
# a correctness one, so one representative set of proposals suffices.
rng = Xoshiro(0)
gr = PureAdsorb.guest_range(st, 1)
sys_of = fill(Int32(1), nmoves)
gidx = Int32[rand(rng, gr) for _ in 1:nmoves]
oldpos = [st.refpoints[i] for i in gidx]
oldq = [st.orientations[i] for i in gidx]
newpos = [st.refpoints[i] + SVector{3, F}((rand(rng, F, 3) .- F(0.5)) .* F(2)) for i in gidx]
newq = [normalize(SVector{4, F}(rand(rng, F, 4) .- F(0.5))) for _ in 1:nmoves]

# Average number of host atoms plus other guests actually inside the LJ/Ewald real-space cutoff,
# over the same proposals, for context on the real-space side of the split.
function count_neighbors(batch, gidx, oldpos, rc2)
    total = 0
    a0 = batch.atom_offsets[1]
    natoms = batch.atom_offsets[2] - a0
    for m in eachindex(gidx)
        pos = oldpos[m]
        for j in (a0 + 1):(a0 + natoms)
            r2 = sum(abs2, PureAdsorb.minimum_image(batch.cells[1], batch.invcells[1], pos - batch.positions[j]))
            r2 < rc2 && (total += 1)
        end
    end
    return total / length(gidx)
end
avg_neighbors = count_neighbors(b, gidx, oldpos, max(b.cutoff, b.ewald_cutoff)^2)

backend === KernelAbstractions.CPU() || (
    global sys_of = PureAdsorb.adapt(backend, sys_of); global gidx = PureAdsorb.adapt(backend, gidx);
    global oldpos = PureAdsorb.adapt(backend, oldpos); global oldq = PureAdsorb.adapt(backend, oldq);
    global newpos = PureAdsorb.adapt(backend, newpos); global newq = PureAdsorb.adapt(backend, newq)
)
db = PureAdsorb.adapt(backend, b)
drefpoints = PureAdsorb.adapt(backend, st.refpoints)
dorientations = PureAdsorb.adapt(backend, st.orientations)
dguest_offsets = PureAdsorb.adapt(backend, st.guest_offsets)
dSk = PureAdsorb.adapt(backend, st.Sk)
dk_offsets = PureAdsorb.adapt(backend, st.k_offsets)
dΔU_real = PureAdsorb.adapt(backend, zeros(F, nmoves))
dΔU_recip = PureAdsorb.adapt(backend, zeros(F, nmoves))

kern_real = PureAdsorb.realspace_move_kernel!(backend)
kern_recip = PureAdsorb.recip_move_kernel!(backend)

kern_real(dΔU_real, sys_of, gidx, oldpos, oldq, newpos, newq, db, guest_compact, drefpoints, dorientations, dguest_offsets, guest_types; ndrange = nmoves)
KernelAbstractions.synchronize(backend)   # warm-up: compile
kern_recip(dΔU_recip, sys_of, oldpos, oldq, newpos, newq, db, g, dSk, dk_offsets; ndrange = nmoves)
KernelAbstractions.synchronize(backend)   # warm-up: compile

bm_real = @be (
    kern_real($dΔU_real, $sys_of, $gidx, $oldpos, $oldq, $newpos, $newq, $db, $guest_compact, $drefpoints, $dorientations, $dguest_offsets, $guest_types; ndrange = $nmoves);
    KernelAbstractions.synchronize($backend)
) seconds = bench_seconds samples = bench_samples evals = 1
bm_recip = @be (
    kern_recip($dΔU_recip, $sys_of, $oldpos, $oldq, $newpos, $newq, $db, $g, $dSk, $dk_offsets; ndrange = $nmoves);
    KernelAbstractions.synchronize($backend)
) seconds = bench_seconds samples = bench_samples evals = 1

realspace_times_s = [s.time for s in bm_real.samples]
recip_times_s = [s.time for s in bm_recip.samples]
real_per_move_s = median(realspace_times_s) / nmoves
recip_per_move_s = median(recip_times_s) / nmoves
println(
    "backend=$backend_name precision=$precision_name nk=$nk avg_neighbors=$(round(avg_neighbors; digits = 1)) " *
        "real=$(real_per_move_s * 1.0e9) ns/move recip=$(recip_per_move_s * 1.0e9) ns/move " *
        "ratio(recip/real)=$(recip_per_move_s / real_per_move_s)"
)
flush(stdout)

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nk, nguests, nmoves, avg_neighbors, cutoff = Float64(b.cutoff), ewald_cutoff = Float64(b.ewald_cutoff),
    bench_seconds, bench_samples, commit,
    description = "Per-move real-space (host-guest + guest-guest LJ/Ewald-real) vs reciprocal-space " *
        "(running structure-factor ΔU) cost, RUBTAK 3x3x3 with $nguests CO2 guests, $nmoves independent " *
        "move proposals per kernel launch.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_guestmove_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, realspace_times_s, recip_times_s, real_per_move_s, recip_per_move_s), 2)
end
println("wrote $outpath")
