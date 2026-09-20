@testitem "insertion_energy agrees with the full-sum reference within the self-term half-range" begin
    using StaticArrays, LinearAlgebra, Random
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g_co2 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    # SPC-like three-site polar guest, reusing CO2's LJ types (C_co2, O_co2, O_co2).
    g_polar = PureAdsorb.Guest(
        SVector(SVector(0.0, 0.0, 0.0), SVector(0.8165, 0.5774, 0.0), SVector(-0.8165, 0.5774, 0.0)),
        g_co2.types, SVector(-0.82, 0.41, 0.41), g_co2.tc, g_co2.pc, g_co2.omega
    )
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    hpos = PureAdsorb.cartesian(sc)
    htype = Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols]
    hq = sc.charges
    A = sc.cell; invA = inv(A)
    α = PureAdsorb.ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = PureAdsorb.ewald_kmax(α, ewald.precision)
    ks_full, kpref_full, Sh_full, _ = PureAdsorb.full_ktables(A, hpos, hq, α, kmax)

    for g in (g_co2, g_polar)
        b = FrameworkBatch([sc], ff, g, ewald)
        # `b.types` uses the batch's compact type index (E3); map back through `compact_to_orig`
        # to index `ff.sigma`/`ff.epsilon` directly, as below.
        btype = b.compact_to_orig[b.types]
        natoms = b.atom_offsets[2] - b.atom_offsets[1]
        # `constant_offset` without the orientation-averaged guest self-term mean it folds in:
        # the same tail/self/exclusion/net terms `insertion_energy_reference` does not include
        # (its E_lr already carries the guest self term, computed per pose).
        self = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
        excl = -PureAdsorb.KE * sum(
            g.charges[a] * g.charges[c] * (1 - PureAdsorb.erfc_dev(α * norm(g.sites[a] - g.sites[c]))) / norm(g.sites[a] - g.sites[c])
                for a in 1:3 for c in (a + 1):3
        )
        counts = [count(==(t), btype) for t in eachindex(ff.names)]
        gcounts = [count(==(t), g.types) for t in eachindex(ff.names)]
        tail = PureAdsorb.tail_delta(ff, counts, gcounts, PureAdsorb.volume(A))
        old_offset = self + excl + tail   # CO2 and the polar guest are both neutral: no net-charge term

        rng = Xoshiro(21)
        for _ in 1:200
            pos = A * rand(rng, SVector{3, Float64})
            q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
            prod = PureAdsorb.insertion_energy(
                pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff,
                b.positions, btype, b.charges, b.atom_offsets[1], natoms,
                A, invA, α, b.ks, b.kprefactor, b.Shost
            ) + b.constant_offset[1]
            ref = PureAdsorb.insertion_energy_reference(
                pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff, hpos, htype, hq, A, invA, α,
                ks_full, kpref_full, Sh_full
            ) + old_offset
            # `self_term_halfrange` is the spread found by 64 *discrete* fixed orientations; an
            # arbitrary continuous pose's guest self term can fall slightly outside that
            # discrete range (measured worst case over 10 000 random poses: 1.33x), so the
            # bound carries a 2x margin on top of the half-range itself.
            @test abs(prod - ref) <= 2 * b.self_term_halfrange[1] + 1.0e-12 * abs(ref)
        end
    end
end

@testitem "RUBTAK CO2 matches kUPS within combined error" tags = [:slow] begin
    using JSON
    ref = JSON.parsefile(joinpath(pkgdir(PureAdsorb), "test", "reference", "rubtak_co2_kups.json"))
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    r = widom(b, g; T = 298.15, ninsert = 1_000_000, seed = 42, nblocks = 20)[1]
    for (ours, err, key) in ((r.mu_ex, r.mu_ex_err, "mu_ex"), (r.K_H, r.K_H_err, "K_H"), (r.q_st, r.q_st_err, "q_st"))
        m, s = ref[key]["mean"], ref[key]["sem"]
        @test abs(ours - m) < 3 * hypot(err, s)
    end
end
