# Decomposes the move kernel's real-space half (per P5.2) into distance/min-image "visits" versus
# in-cutoff pair arithmetic, by running `realspace_move_kernel!` (unmodified, from `src/guest.jl`)
# against three copies of the same batch that differ only in `cutoff`/`ewald_cutoff`: the real
# values, the Ewald cutoff collapsed to near zero (LJ arithmetic only), and both cutoffs collapsed
# to near zero (visits only, no pair arithmetic at all). The three runs share identical alpha,
# k-tables, sigma/epsilon and proposals, so differencing them attributes cost to exactly one
# cutoff's worth of pair arithmetic:
#   visit cost        = both-zero run
#   LJ pair arithmetic = ewald-zero run  - both-zero run
#   Ewald pair arithmetic = full run     - ewald-zero run
# This settles whether a neighbour list (not approved, not built here) would remove mostly visits
# or mostly arithmetic. Does not modify src/.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NMOVES` sets the number of independent move proposals launched per kernel call (default
# 65536, matching `guest_bench.jl`'s chunk size).
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

rng = Xoshiro(0)
gr = PureAdsorb.guest_range(st, 1)
sys_of = fill(Int32(1), nmoves)
gidx = Int32[rand(rng, gr) for _ in 1:nmoves]
oldpos = [st.refpoints[i] for i in gidx]
oldq = [st.orientations[i] for i in gidx]
newpos = [st.refpoints[i] + SVector{3, F}((rand(rng, F, 3) .- F(0.5)) .* F(2)) for i in gidx]
newq = [normalize(SVector{4, F}(rand(rng, F, 4) .- F(0.5))) for _ in 1:nmoves]

# A copy of `b` with different real-space cutoff RADII only: same alpha, k-table, sigma/epsilon,
# atoms and cells, so the only thing that changes between runs is which pairs pass the
# `r2 < rc_lj2`/`r2 < rc_ew2` branches inside `host_guest_realspace_energy`/
# `guest_pair_realspace_energy` — never the physics those branches gate.
with_cutoffs(batch::PureAdsorb.FrameworkBatch, cutoff, ewald_cutoff) = PureAdsorb.FrameworkBatch(
    batch.positions, batch.types, batch.charges, batch.atom_offsets, batch.cells, batch.invcells, batch.volumes, batch.alphas,
    batch.ks, batch.kprefactor, batch.Shost, batch.k_offsets, batch.constant_offset, batch.self_term_halfrange,
    batch.ncells, batch.cell_offsets, batch.cellgrid_offsets, batch.sigma, batch.epsilon, batch.compact_to_orig,
    batch.guest_types, batch.guest_types_orig, batch.guest_sites_orig, batch.guest_charges_orig, batch.bs, batch.kmin,
    cutoff, ewald_cutoff, batch.fullk, batch.nsys, batch.framework_of
)

# Effectively zero without literal zero: a distance below this never occurs between real atoms,
# so the cutoff branch is false on every visit while the branch's own arithmetic (`r2 < rc2`)
# stays well defined.
const NEAR_ZERO = 1.0e-6

b_full = b
b_ewald0 = with_cutoffs(b, b.cutoff, F(NEAR_ZERO))
b_both0 = with_cutoffs(b, F(NEAR_ZERO), F(NEAR_ZERO))

backend === KernelAbstractions.CPU() || (
    global sys_of = PureAdsorb.adapt(backend, sys_of); global gidx = PureAdsorb.adapt(backend, gidx);
    global oldpos = PureAdsorb.adapt(backend, oldpos); global oldq = PureAdsorb.adapt(backend, oldq);
    global newpos = PureAdsorb.adapt(backend, newpos); global newq = PureAdsorb.adapt(backend, newq)
)
drefpoints = PureAdsorb.adapt(backend, st.refpoints)
dorientations = PureAdsorb.adapt(backend, st.orientations)
dguest_offsets = PureAdsorb.adapt(backend, st.guest_offsets)
dΔU = PureAdsorb.adapt(backend, zeros(F, nmoves))

kern = PureAdsorb.realspace_move_kernel!(backend)

function run_case(db)
    kern(dΔU, sys_of, gidx, oldpos, oldq, newpos, newq, db, guest_compact, drefpoints, dorientations, dguest_offsets, guest_types; ndrange = nmoves)
    KernelAbstractions.synchronize(backend)   # warm-up: compile
    bm = @be (
        kern($dΔU, $sys_of, $gidx, $oldpos, $oldq, $newpos, $newq, $db, $guest_compact, $drefpoints, $dorientations, $dguest_offsets, $guest_types; ndrange = $nmoves);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1
    return [s.time for s in bm.samples]
end

full_times_s = run_case(PureAdsorb.adapt(backend, b_full))
ewald0_times_s = run_case(PureAdsorb.adapt(backend, b_ewald0))
both0_times_s = run_case(PureAdsorb.adapt(backend, b_both0))

full_per_move = median(full_times_s) / nmoves
ewald0_per_move = median(ewald0_times_s) / nmoves
both0_per_move = median(both0_times_s) / nmoves
visit_per_move = both0_per_move
lj_arith_per_move = ewald0_per_move - both0_per_move
ewald_arith_per_move = full_per_move - ewald0_per_move
total_arith_per_move = full_per_move - both0_per_move

println(
    "backend=$backend_name precision=$precision_name full=$(full_per_move * 1.0e9) ns/move " *
        "ewald0=$(ewald0_per_move * 1.0e9) ns/move both0(visit)=$(both0_per_move * 1.0e9) ns/move " *
        "lj_arith=$(lj_arith_per_move * 1.0e9) ns/move ewald_arith=$(ewald_arith_per_move * 1.0e9) ns/move " *
        "visit_fraction=$(visit_per_move / full_per_move)"
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
    precision = precision_name, nk, nguests, nmoves, cutoff = Float64(b.cutoff), ewald_cutoff = Float64(b.ewald_cutoff),
    bench_seconds, bench_samples, commit,
    description = "P5.2: realspace_move_kernel! per-move cost at the real cutoffs, at the Ewald cutoff collapsed " *
        "to near zero, and at both cutoffs collapsed to near zero, on RUBTAK 3x3x3 + 50 CO2. Differencing the three " *
        "isolates the visit (distance/min-image) cost from LJ and Ewald in-cutoff pair arithmetic.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_visitvspair_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(
        io, (;
            meta, full_times_s, ewald0_times_s, both0_times_s, full_per_move, ewald0_per_move, both0_per_move,
            visit_per_move, lj_arith_per_move, ewald_arith_per_move, total_arith_per_move,
        ), 2
    )
end
println("wrote $outpath")
