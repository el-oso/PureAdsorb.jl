# Gate 4 of the ewald_gg oracle tests (test/guest_tests.jl) asks that `total_energy` match an
# independently-decomposed brute-force oracle to 1e-10 relative; two `@test_broken` items record
# that this does not hold reliably (measured 5e-10 to 2.4e-9). The oracle there always requests a
# FIXED precision (1e-8) for its own k-space cutoff, using the host's own alpha throughout, while
# the production path's guest-guest terms use a separate `alpha_gg`/`ewald_cutoff_gg` decomposition
# — two independently-truncated Ewald sums being compared, rather than the same decomposition
# compared to itself. If the residual is pure truncation error (not a formula bug), it must scale
# with `ewald_gg.precision`, and it must stop scaling once that precision passes the oracle's own
# fixed resolution, at which point the oracle's own truncation becomes the limiting term.
#
# This script isolates `ewald_gg.precision`'s effect using the SAME oracle helper the tests use
# (`oracle_total_energy`, duplicated here since it lives in a `@testsnippet`), against two
# different oracle resolutions: 1e-12 (tight enough to be a clean reference across the whole
# swept range) and 1e-8 (the resolution the actual gate-4 tests use, to show why they show a
# floor). A second sweep holds `ewald_gg.precision` fixed and varies `ewald_gg.cutoff` instead,
# which changes the guest-guest k-vector count without changing the target accuracy.
using PureAdsorb, StaticArrays, LinearAlgebra, Random, JSON, Dates

function oracle_total_energy(batch, state, guest::PureAdsorb.Guest{T, N}, ff, n::Integer, precision) where {T, N}
    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    hpos = batch.positions[(a0 + 1):(a0 + natoms)]
    hq = batch.charges[(a0 + 1):(a0 + natoms)]
    htype = batch.compact_to_orig[batch.types[(a0 + 1):(a0 + natoms)]]
    gr = PureAdsorb.guest_range(state, n)
    Ng = length(gr)
    alpha = batch.alphas[fw]; ewald_cutoff = batch.ewald_cutoff
    ks, w, = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(alpha, precision))

    gpos = SVector{3, T}[]; gq = T[]; gmol = Int[]; gtype_orig = Int[]
    for (idx, i) in enumerate(gr), s in 1:N
        push!(gpos, state.refpoints[i] + PureAdsorb.rotate(state.orientations[i], guest.sites[s]))
        push!(gq, guest.charges[s]); push!(gmol, natoms + idx); push!(gtype_orig, guest.types[s])
    end
    allpos = vcat(hpos, gpos); allq = vcat(hq, gq); allmol = vcat(collect(1:natoms), gmol)
    Ecoul = PureAdsorb.ewald_energy(A, allpos, allq, allmol, alpha, ewald_cutoff, ks, w) -
        PureAdsorb.ewald_energy(A, hpos, hq, collect(1:natoms), alpha, ewald_cutoff, ks, w)

    rc2 = ff.cutoff^2
    Elj = zero(T)
    for a in eachindex(gpos)
        for h in eachindex(hpos)
            r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - hpos[h]))
            r * r < rc2 || continue
            sig = ff.sigma[gtype_orig[a], htype[h]]; eps = ff.epsilon[gtype_orig[a], htype[h]]
            x = (sig / r)^6
            Elj += 4 * eps * (x^2 - x)
        end
        for bb in eachindex(gpos)
            bb > a || continue
            gmol[a] == gmol[bb] && continue
            r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - gpos[bb]))
            r * r < rc2 || continue
            sig = ff.sigma[gtype_orig[a], gtype_orig[bb]]; eps = ff.epsilon[gtype_orig[a], gtype_orig[bb]]
            x = (sig / r)^6
            Elj += 4 * eps * (x^2 - x)
        end
    end

    ntypes_ff = length(ff.names)
    gcounts = zeros(Int, ntypes_ff)
    for t in guest.types
        gcounts[t] += 1
    end
    host_counts = zeros(Int, ntypes_ff)
    for t in htype
        host_counts[t] += 1
    end
    Etail = PureAdsorb.tail_delta(ff, host_counts, Ng .* gcounts, batch.volumes[fw])
    return Elj + Ecoul + Etail
end

function empty_box_setup(::Type{T}) where {T}
    ff0 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = T)
    names = vcat(ff0.names, ["X1_"])
    sigma = vcat([ff0.sigma[i, i] for i in eachindex(ff0.names)], T(1))
    epsilon = vcat([ff0.epsilon[i, i] for i in eachindex(ff0.names)], T(0))
    ff = ForceField(names, sigma, epsilon; cutoff = T(12), tail = true)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = T)
    A = SMatrix{3, 3}(T(30), T(0), T(0), T(0), T(30), T(0), T(0), T(0), T(30))
    fw = Framework{T}(A, [SVector(T(0), T(0), T(0))], ["X1"], ["X1"], [T(0)])
    return ff, g, fw
end

function relerr_at(b, st, g, ff, n, oracle_precision)
    ref = oracle_total_energy(b, st, g, ff, n, oracle_precision)
    prod = PureAdsorb.total_energy(b, st, g, ff, n)
    return abs(prod - ref) / max(abs(ref), abs(prod))
end

precisions = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 3.0e-9]

# --- RUBTAK 3x3x3 with guests, matching the gate-4 test's exact configuration ---
fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-8)

function precision_sweep(sc, ff, g, ewald, cutoff_gg, precisions, oracle_precision)
    rows = NamedTuple[]
    for p in precisions
        ewald_gg = EwaldParams(cutoff = cutoff_gg, precision = p)
        b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = false, ewald_gg)
        fwidx = b.framework_of[2]
        st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15, seed = 42)
        push!(
            rows, (;
                precision = p, alpha_gg = b.alphas_gg[fwidx],
                n_k_gg = length(PureAdsorb.batch_kvec_gg_range(b, 2)),
                relerr = relerr_at(b, st, g, ff, 2, oracle_precision),
            )
        )
    end
    return rows
end

sweep_tight_oracle = precision_sweep(sc, ff, g, ewald, 13.0, precisions, 1.0e-12)
sweep_matched_oracle = precision_sweep(sc, ff, g, ewald, 13.0, precisions, 1.0e-8)
sweep_tight_oracle_c16 = precision_sweep(sc, ff, g, ewald, 16.0, precisions, 1.0e-12)

cutoffs_gg = [13.0, 14.0, 15.0, 16.0, 16.5]
cutoff_sweep = NamedTuple[]
for c in cutoffs_gg
    ewald_gg = EwaldParams(cutoff = c, precision = 1.0e-6)
    b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = false, ewald_gg)
    fwidx = b.framework_of[2]
    st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15, seed = 42)
    push!(
        cutoff_sweep, (;
            cutoff_gg = c, alpha_gg = b.alphas_gg[fwidx],
            n_k_gg = length(PureAdsorb.batch_kvec_gg_range(b, 2)),
            relerr = relerr_at(b, st, g, ff, 2, 1.0e-12),
        )
    )
end

# --- pure CO2 in an empty box, the second broken oracle test ---
ffe, ge, fwe = empty_box_setup(Float64)
ewalde = EwaldParams(cutoff = 12.0, precision = 1.0e-8)
sweep_empty_box = NamedTuple[]
for p in precisions
    ewald_gge = EwaldParams(cutoff = 13.5, precision = p)
    be = FrameworkBatch([fwe], ffe, ge, ewalde; fullk = false, ewald_gg = ewald_gge)
    st = PureAdsorb.SystemState(be, ge, [50], ffe; T = 298.15, seed = 42)
    push!(
        sweep_empty_box, (;
            precision = p, alpha_gg = be.alphas_gg[be.framework_of[1]],
            n_k_gg = length(PureAdsorb.batch_kvec_gg_range(be, 1)),
            relerr = relerr_at(be, st, ge, ffe, 1, 1.0e-12),
        )
    )
end

for (name, rows) in (
        "RUBTAK 3x3x3, cutoff_gg=13.0, oracle precision=1e-12" => sweep_tight_oracle,
        "RUBTAK 3x3x3, cutoff_gg=13.0, oracle precision=1e-8 (matches the gate-4 test)" => sweep_matched_oracle,
        "RUBTAK 3x3x3, cutoff_gg=16.0, oracle precision=1e-12" => sweep_tight_oracle_c16,
        "RUBTAK 3x3x3, precision_gg=1e-6, cutoff_gg swept, oracle precision=1e-12" => cutoff_sweep,
        "empty box, cutoff_gg=13.5, oracle precision=1e-12" => sweep_empty_box,
    )
    println(name)
    for r in rows
        println("  ", r)
    end
end

out = Dict(
    "meta" => Dict(
        "description" => "Gate-4 diagnosis check: does the ewald_gg split-vs-oracle discrepancy scale with Ewald precision (pure truncation error) or plateau at a fixed floor (a formula bug)?",
        "date" => string(today()),
        "julia_version" => string(VERSION),
    ),
    "rubtak_cutoff13_oracle1e-12" => sweep_tight_oracle,
    "rubtak_cutoff13_oracle1e-8" => sweep_matched_oracle,
    "rubtak_cutoff16_oracle1e-12" => sweep_tight_oracle_c16,
    "rubtak_cutoff_sweep_precision1e-6_oracle1e-12" => cutoff_sweep,
    "empty_box_cutoff13.5_oracle1e-12" => sweep_empty_box,
)
outpath = joinpath(pkgdir(PureAdsorb), "bench", "results", "ewald_split_precision_sweep.json")
open(outpath, "w") do io
    JSON.print(io, out, 2)
end
println("wrote ", outpath)
