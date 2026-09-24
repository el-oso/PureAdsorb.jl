# Time decomposition of the CPU `widom` kernel and the cell-list sparsity ratio, for the standard
# RUBTAK 3x3x3 + CO2 case (LJ cutoff 12 Å, Ewald cutoff 12 Å, cellwidth 2 Å). Read-only
# measurement: this script adds no dependency and does not modify src/.
#
# Decomposition method: `Profile.@profile` over a real `widom()` call segfaults on this machine
# (see the note above `decompose`), so the decomposition instead calls the real, unmodified
# `PureAdsorb.insertion_energy` four times over the same real survivor poses with its own
# `cutoff`/`ewald_cutoff`/`ks` arguments set to disable one term at a time.
#
# Sparsity ratio: re-runs `hardcore_kernel!`'s own cell-list traversal (same helper functions from
# src/cell.jl, called qualified since they are not exported) as a plain host loop over a sample of
# real random poses, counting atoms visited per guest site against `natoms`.
using PureAdsorb, StaticArrays, KernelAbstractions, LinearAlgebra, Random, Statistics, Chairmarks
BLAS.set_num_threads(1)

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
g0 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
cellwidth = 2.0

function build(::Type{F}) where {F}
    fwF = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
    ffF = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    gF = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ffF; T = F)
    scF = replicate(fwF, (3, 3, 3))
    ewaldF = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
    b = FrameworkBatch([scF], ffF, gF, ewaldF; cellwidth = F(cellwidth))
    return b, gF
end

println("=== natoms (framework atom count, RUBTAK 3x3x3) ===")
b64, g64 = build(Float64)
natoms = b64.atom_offsets[end] - b64.atom_offsets[1]
println("natoms = ", natoms)
flush(stdout)

# --- Item 2: cell-list sparsity ratio (hardcore_kernel!'s own traversal, item 2) ---
function count_visits(batch::PureAdsorb.FrameworkBatch{F}, guest_compact, rho2, reach0, ntypes, sys_of, rpos, quat) where {F}
    N = length(guest_compact.sites)
    total_visits = 0
    total_flagged = 0
    npose = length(sys_of)
    for i in 1:npose
        s = sys_of[i]
        a0 = batch.atom_offsets[s]
        g0_ = batch.cellgrid_offsets[s] + 1
        g1_ = batch.cellgrid_offsets[s + 1]
        A = batch.cells[s]
        invA = batch.invcells[s]
        pos = A * rpos[i]
        gsites = map(sv -> PureAdsorb.rotate(quat[i], sv), guest_compact.sites)
        cell_offsets = view(batch.cell_offsets, g0_:g1_)
        n = batch.ncells[s]
        m = reach0[s]
        n1 = n[1]; n2 = n[2]; n3 = n[3]
        m1 = m[1]; m2 = m[2]; m3 = m[3]
        base = (s - 1) * N * ntypes
        flagged = false
        for a in 1:N
            site_pos = pos + gsites[a]
            f = invA * site_pos
            h1 = PureAdsorb.home_cell_dev(PureAdsorb.wrap_frac(f[1]), n1)
            h2 = PureAdsorb.home_cell_dev(PureAdsorb.wrap_frac(f[2]), n2)
            h3 = PureAdsorb.home_cell_dev(PureAdsorb.wrap_frac(f[3]), n3)
            start1, count1 = PureAdsorb.stencil_start_count(h1, m1, n1)
            start2, count2 = PureAdsorb.stencil_start_count(h2, m2, n2)
            start3, count3 = PureAdsorb.stencil_start_count(h3, m3, n3)
            for t3 in Int32(0):(count3 - Int32(1))
                c3 = PureAdsorb.wrap_cell(start3 + t3, n3)
                for t2 in Int32(0):(count2 - Int32(1))
                    c2 = PureAdsorb.wrap_cell(start2 + t2, n2)
                    for t1 in Int32(0):(count1 - Int32(1))
                        c1 = PureAdsorb.wrap_cell(start1 + t1, n1)
                        c = PureAdsorb.cell_linear(c1, c2, c3, n1, n2)
                        j0 = a0 + cell_offsets[c + 1] + 1
                        j1 = a0 + cell_offsets[c + 2]
                        for j in j0:j1
                            total_visits += 1
                            Δ = PureAdsorb.minimum_image(A, invA, site_pos - batch.positions[j])
                            ht = batch.types[j]
                            r2 = dot(Δ, Δ)
                            idx = base + (a - 1) * ntypes + ht
                            r2 < rho2[idx] && (flagged = true)
                        end
                    end
                end
            end
        end
        total_flagged += flagged
    end
    return total_visits, total_flagged, npose * N
end

npose_sample = 20_000
N = length(g64.sites)
guest_compact = PureAdsorb.Guest{Float64, N}(g64.sites, SVector{N, Int}(b64.guest_types), g64.charges, g64.tc, g64.pc, g64.omega)
kT = Float64(PureAdsorb.KB * 298.15)
rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b64, guest_compact, kT)
rng = Xoshiro(1)
sys_of = Vector{Int32}(undef, npose_sample)
rpos = Vector{SVector{3, Float64}}(undef, npose_sample)
quat = Vector{SVector{4, Float64}}(undef, npose_sample)
PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, PureAdsorb.default_run(npose_sample, 1), 1)
total_visits, total_flagged, total_sites = count_visits(b64, guest_compact, rho2, reach0, ntypes, sys_of, rpos, quat)
println("=== item 2: cell-list sparsity (phase-0 rejection stencil) ===")
println("poses sampled = ", npose_sample, "  guest sites/pose = ", N)
println("mean atoms visited per guest site = ", total_visits / total_sites)
println("natoms (total framework atoms)    = ", natoms)
println("sparsity ratio (visited/natoms)   = ", (total_visits / total_sites) / natoms)
println("fraction of poses flagged (rejected) = ", total_flagged / npose_sample)
println(
    "NOTE: phase-1 (widom_kernel!/insertion_energy, the dominant kernel cost, see item 1) does " *
        "NOT use this cell list — it loops over all $natoms framework atoms per surviving pose, " *
        "gated only by a cutoff distance test, not a cell-list stencil."
)
flush(stdout)

# --- Item 1: time decomposition of the CPU kernel, per precision ---
# `Profile.@profile` over a real `widom()` call segfaults on this machine (libunwind
# "stepWithCompactEncoding - invalid compact unwind encoding", SIGABRT) — a macOS/Apple Silicon
# libunwind issue with Julia's sampling profiler, not specific to this code. Reported as-is; no
# workaround attempted, per instructions to stop a failing probe rather than route around it.
#
# Fallback: call the REAL, unmodified `PureAdsorb.insertion_energy` four times over the same real
# survivor poses, with `cutoff`/`ewald_cutoff`/`ks` (its own arguments, not a reimplementation)
# set so each call does a strict superset of the work of the one before it — no duplicated loop
# body, so no risk of the new code compiling differently from the original:
#   v0: cutoff=0, ewald_cutoff=0, ks=∅         -> pair loop + distance/cutoff test only
#   v1: cutoff=real, ewald_cutoff=0, ks=∅      -> + Lennard-Jones
#   v2: cutoff=real, ewald_cutoff=real, ks=∅   -> + real-space screened Coulomb
#   v3: cutoff=real, ewald_cutoff=real, ks=real (unmodified) -> + Ewald reciprocal = full
# Successive differences isolate each term's marginal cost. Since `cutoff=0`/`ewald_cutoff=0`
# make the corresponding branch's condition always false rather than skipping it, v0 still pays
# the branch-not-taken cost of the eliminated terms, so v0's time is an upper bound on "distance
# and cutoff test alone", not a lower one.
function decompose(::Type{F}) where {F}
    b, gF = build(F)
    backend = KernelAbstractions.CPU()
    N = length(gF.sites)
    guest_compact = PureAdsorb.Guest{F, N}(gF.sites, SVector{N, Int}(b.guest_types), gF.charges, gF.tc, gF.pc, gF.omega)
    kT = F(PureAdsorb.KB * 298.15)
    rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, guest_compact, kT)

    # Real survivor sample, exactly as widom_kernel! receives it: run phase 0 once on a real chunk.
    chunk = 65536
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, chunk)
    rpos = Vector{SVector{3, F}}(undef, chunk)
    quat = Vector{SVector{4, F}}(undef, chunk)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, PureAdsorb.default_run(chunk, 1), 1)
    dbatch = PureAdsorb.adapt(backend, b)
    dsys = PureAdsorb.adapt(backend, sys_of)
    drpos = PureAdsorb.adapt(backend, rpos)
    dquat = PureAdsorb.adapt(backend, quat)
    flags = Vector{UInt8}(undef, chunk)
    dflags = PureAdsorb.adapt(backend, flags)
    drho2 = PureAdsorb.adapt(backend, rho2)
    dreach0 = PureAdsorb.adapt(backend, reach0)
    kern0 = PureAdsorb.hardcore_kernel!(backend)
    kern0(dflags, dsys, drpos, dquat, dbatch, guest_compact, drho2, dreach0, Int32(ntypes); ndrange = chunk)
    KernelAbstractions.synchronize(backend)
    copyto!(flags, dflags)
    survivor = Int32[i for i in 1:chunk if iszero(flags[i])]
    nsurv = length(survivor)
    println("F=$F: chunk=$chunk survivors=$nsurv ($(round(100nsurv / chunk; digits = 1))%)")

    a0 = b.atom_offsets[1]
    natoms_s = b.atom_offsets[2] - b.atom_offsets[1]
    A = b.cells[1]; invA = b.invcells[1]
    alpha = b.alphas[1]
    ks = view(b.ks, (b.k_offsets[1] + 1):b.k_offsets[2])
    kprefactor = view(b.kprefactor, (b.k_offsets[1] + 1):b.k_offsets[2])
    Shost = view(b.Shost, (b.k_offsets[1] + 1):b.k_offsets[2])
    ks_empty = view(b.ks, 1:0)
    kprefactor_empty = view(b.kprefactor, 1:0)
    Shost_empty = view(b.Shost, 1:0)

    function drive(cutoff, ewald_cutoff, ks_, kprefactor_, Shost_)
        acc = zero(F)
        for i in survivor
            pos = A * rpos[i]
            acc += PureAdsorb.insertion_energy(
                pos, quat[i], guest_compact, b.sigma, b.epsilon, cutoff, ewald_cutoff,
                b.positions, b.types, b.charges, a0, natoms_s, A, invA, alpha, ks_, kprefactor_, Shost_
            )
        end
        return acc
    end

    v0() = drive(zero(F), zero(F), ks_empty, kprefactor_empty, Shost_empty)
    v1() = drive(b.cutoff, zero(F), ks_empty, kprefactor_empty, Shost_empty)
    v2() = drive(b.cutoff, b.ewald_cutoff, ks_empty, kprefactor_empty, Shost_empty)
    v3() = drive(b.cutoff, b.ewald_cutoff, ks, kprefactor, Shost)   # full, unmodified

    v0(); v1(); v2(); v3()   # warm-up: compile

    t0 = median([s.time for s in (@be v0() seconds = 8 samples = 7 evals = 1).samples])
    t1 = median([s.time for s in (@be v1() seconds = 8 samples = 7 evals = 1).samples])
    t2 = median([s.time for s in (@be v2() seconds = 8 samples = 7 evals = 1).samples])
    t3 = median([s.time for s in (@be v3() seconds = 8 samples = 7 evals = 1).samples])

    dist_s = t0
    lj_s = t1 - t0
    coul_s = t2 - t1
    ewald_s = t3 - t2

    println("F=$F  v0(dist)=$(t0)s  v1(+lj)=$(t1)s  v2(+coulomb)=$(t2)s  v3(+ewald,full)=$(t3)s")
    println(
        "F=$F  decomposition of insertion_energy time over $nsurv survivors: " *
            "distance+cutoff<=$(round(100dist_s / t3; digits = 1))%  " *
            "LJ=$(round(100lj_s / t3; digits = 1))%  " *
            "real-Coulomb=$(round(100coul_s / t3; digits = 1))%  " *
            "Ewald-recip=$(round(100ewald_s / t3; digits = 1))%"
    )
    flush(stdout)
    return (; t0, t1, t2, t3, nsurv, chunk)
end

println("=== item 1: time decomposition (real insertion_energy, cutoffs/ks zeroed per term) ===")
for F in (Float64, Float32)
    decompose(F)
end
