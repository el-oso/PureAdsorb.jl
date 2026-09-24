# Metal-vs-CPU agreement check, Float32 (Apple GPUs have no double precision, so this is the
# only precision Metal can run). Mirrors test/gpu_tests.jl's ":gpu" testitem but is a standalone
# script rather than a test: that item hard-requires CUDA or AMDGPU and asserts rtol=1e-10 in
# Float64, so it cannot gate Metal and must not be edited to try.
#
# Bit-exact agreement is not expected across backends in Float32 (different reduction order and
# transcendental implementations); this reports the actual relative differences in mu_ex, K_H,
# q_st against each run's own statistical error, and checks the phase-0 hard-core rejection
# flags for exact agreement (they are boolean/integer, so should not differ at all).
using PureAdsorb, StaticArrays, Random, KernelAbstractions, Metal

F = Float32
fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = F(12), precision = F(1.0e-6)))

rc = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4)[1]
rg = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4, backend = MetalBackend())[1]

relerr(a, b) = abs(a - b) / abs(a)
println("mu_ex: cpu=$(rc.mu_ex) +/- $(rc.mu_ex_err)  metal=$(rg.mu_ex) +/- $(rg.mu_ex_err)  relerr=$(relerr(rc.mu_ex, rg.mu_ex))")
println("K_H:   cpu=$(rc.K_H) +/- $(rc.K_H_err)  metal=$(rg.K_H) +/- $(rg.K_H_err)  relerr=$(relerr(rc.K_H, rg.K_H))")
println("q_st:  cpu=$(rc.q_st) +/- $(rc.q_st_err)  metal=$(rg.q_st) +/- $(rg.q_st_err)  relerr=$(relerr(rc.q_st, rg.q_st))")

# Phase 0 (hardcore_kernel!) on its own, directly compared: `widom`'s end-to-end agreement above
# could in principle mask a phase-0 discrepancy that phase 1 happens to compensate.
N = length(g.sites)
g_compact = PureAdsorb.Guest{F, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
kT = F(PureAdsorb.KB * 298.15)
rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, g_compact, kT)
nposes = 20_000
rng = Xoshiro(5)
sys_of = Vector{Int32}(undef, nposes)
rpos = Vector{SVector{3, F}}(undef, nposes)
quat = Vector{SVector{4, F}}(undef, nposes)
PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, PureAdsorb.default_run(nposes, b.nsys), b.nsys)
flags_cpu = zeros(UInt8, nposes)
PureAdsorb.hardcore_kernel!(CPU())(
    flags_cpu, sys_of, rpos, quat, b, g_compact, rho2, reach0, Int32(ntypes); ndrange = nposes
)
KernelAbstractions.synchronize(CPU())

backend = MetalBackend()
dbatch = PureAdsorb.adapt(backend, b)
dsys, drpos, dquat = PureAdsorb.adapt(backend, sys_of), PureAdsorb.adapt(backend, rpos), PureAdsorb.adapt(backend, quat)
drho2, dreach0 = PureAdsorb.adapt(backend, rho2), PureAdsorb.adapt(backend, reach0)
dflags = PureAdsorb.adapt(backend, zeros(UInt8, nposes))
PureAdsorb.hardcore_kernel!(backend)(
    dflags, dsys, drpos, dquat, dbatch, g_compact, drho2, dreach0, Int32(ntypes); ndrange = nposes
)
KernelAbstractions.synchronize(backend)
flags_metal = Array(dflags)
nmismatch = count(flags_cpu .!= flags_metal)
println("phase-0 rejection flags: $(nmismatch) mismatches out of $(nposes)")
