@testitem "theta_F is the exact underflow boundary" begin
    for T in (Float64, Float32)
        θ = PureAdsorb.theta_F(T)
        @test exp(-θ) == zero(T)
        @test exp(-prevfloat(θ)) != zero(T)
    end
    # Recorded orders of magnitude (not hard-coded into the implementation): about 745.13 for
    # Float64, about 103.97 for Float32.
    @test 745 < PureAdsorb.theta_F(Float64) < 746
    @test 103 < PureAdsorb.theta_F(Float32) < 105
end

@testitem "find_r0 brackets a genuine sign change of the pair energy" begin
    using Random
    rng = Xoshiro(1)
    for _ in 1:200
        σ = 2.0 + 3rand(rng)
        ε = 0.0005 + 0.02rand(rng)
        K = -20rand(rng)
        α = 0.2 + 0.3rand(rng)
        r0 = PureAdsorb.find_r0(σ, ε, K, α, Inf)
        u0 = PureAdsorb.pair_energy(r0, σ, ε, K, α)
        @test abs(u0) < 1.0e-9 * ε
        @test PureAdsorb.pair_energy(prevfloat(r0) - 1.0e-8r0, σ, ε, K, α) > 0
        @test PureAdsorb.pair_energy(r0 * 1.001, σ, ε, K, α) < 0
    end
end

@testitem "find_r0 rejects a non-attractive or well-less pair" begin
    @test_throws "must be positive" PureAdsorb.find_r0(3.0, 0.0, -1.0, 0.2, Inf)
    @test_throws "must be negative" PureAdsorb.find_r0(3.0, 0.01, 1.0, 0.2, Inf)
end

# All fixtures below index `sigma`/`epsilon`/`types` by the FORCE FIELD's own type index
# (`ff.sigma`, `htype = b.compact_to_orig[b.types]`, `g.types`), not `FrameworkBatch`'s compact
# index: `hardcore_bound`/`kmin_table`/`find_rho2` are generic over whichever index space their
# `sigma`/`epsilon`/`types` arguments share, and the force field's own space avoids needing a
# separately-built compact guest here.
@testsnippet RejectFixture begin
    using StaticArrays, LinearAlgebra, Random

    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald)
    htype = b.compact_to_orig[b.types]

    # The most negative single guest-site/host-atom pair energy at a pose, over `cutoff`: stands
    # in for "the trigger pair" in the bound test below. `hardcore_bound`'s per-pair envelope is
    # a valid lower bound on every pair's energy over its whole domain, including whichever pair
    # is picked out here, so subtracting even more than that one pair's own envelope contribution
    # (the full B_s) only makes the inequality more conservative, never wrong.
    function min_pair_energy(g, ff, b, htype, pos, gsites, α, atoms, A, invA)
        u_trigger = Inf
        for a in eachindex(g.sites), h in atoms
            Δ = PureAdsorb.minimum_image(A, invA, pos - b.positions[h]) + gsites[a]
            r = norm(Δ)
            r < ff.cutoff || continue
            σ = ff.sigma[g.types[a], htype[h]]
            ε = ff.epsilon[g.types[a], htype[h]]
            K = PureAdsorb.KE * g.charges[a] * b.charges[h]
            u_trigger = min(u_trigger, PureAdsorb.pair_energy(r, σ, ε, K, α))
        end
        return u_trigger
    end

    # Runs the rejection-vs-full-energy comparison for `nposes` random poses in a genuine
    # function body (a `@testitem`'s top-level scope does not specialize on types the way a
    # function does, and this loop touches every host atom per pose).
    function count_falsepositives(g, ff, b, htype, rho2, atoms, natoms, α, kT, ::Type{T}; nposes) where {T}
        A = b.cells[1]; invA = b.invcells[1]
        rng = Xoshiro(2026)
        n_rejected = 0
        for _ in 1:nposes
            pos = A * rand(rng, SVector{3, T})
            q = PureAdsorb.shoemake_quaternion(rng, T)
            gsites = map(s -> PureAdsorb.rotate(q, s), g.sites)
            rejected = false
            for a in eachindex(g.sites), h in atoms
                Δ = PureAdsorb.minimum_image(A, invA, pos - b.positions[h]) + gsites[a]
                if sum(abs2, Δ) < rho2[a, htype[h]]
                    rejected = true
                    break
                end
            end
            rejected || continue
            n_rejected += 1
            e = PureAdsorb.insertion_energy(
                pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, b.ewald_cutoff,
                b.positions, htype, b.charges, b.atom_offsets[1], natoms,
                A, invA, α, b.ks, b.kprefactor, b.Shost
            )
            ΔU = e + b.constant_offset[1]
            w = exp(-ΔU / kT)
            iszero(w) || return n_rejected, false
        end
        return n_rejected, true
    end
end

@testitem "hardcore_bound holds against a random-pose brute-force check" setup = [RejectFixture] begin
    α = b.alphas[1]
    atoms = (b.atom_offsets[1] + 1):b.atom_offsets[2]
    natoms = b.atom_offsets[2] - b.atom_offsets[1]
    Bs = PureAdsorb.hardcore_bound(g, ff.sigma, ff.epsilon, b.positions, htype, b.charges, atoms, ff.cutoff, α, b.kprefactor, b.Shost)
    @test isfinite(Bs)
    @test Bs > 0

    A = b.cells[1]; invA = b.invcells[1]; cs = b.constant_offset[1]
    rng = Xoshiro(3)
    for _ in 1:2000
        pos = A * rand(rng, SVector{3, Float64})
        q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
        gsites = [PureAdsorb.rotate(q, s) for s in g.sites]
        u_trigger = min_pair_energy(g, ff, b, htype, pos, gsites, α, atoms, A, invA)
        isfinite(u_trigger) || continue
        e = PureAdsorb.insertion_energy(
            pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, b.ewald_cutoff,
            b.positions, htype, b.charges, b.atom_offsets[1], natoms,
            A, invA, α, b.ks, b.kprefactor, b.Shost
        )
        ΔU = e + cs
        @test ΔU - u_trigger >= -Bs + cs - 1.0e-6 * Bs
    end
end

@testitem "find_rho2 rejection never has a nonzero Boltzmann weight" setup = [RejectFixture] begin
    for T in (Float64, Float32)
        fw2 = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T)
        ff2 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T)
        g2 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff2; T)
        sc2 = replicate(fw2, (3, 3, 3))
        ewald2 = EwaldParams(cutoff = T(12), precision = T(1.0e-6))
        b2 = FrameworkBatch([sc2], ff2, g2, ewald2; cellwidth = T(3))
        htype2 = b2.compact_to_orig[b2.types]
        α = b2.alphas[1]
        atoms = (b2.atom_offsets[1] + 1):b2.atom_offsets[2]
        natoms = b2.atom_offsets[2] - b2.atom_offsets[1]
        ntypes = maximum(view(htype2, atoms))
        Bs = PureAdsorb.hardcore_bound(g2, ff2.sigma, ff2.epsilon, b2.positions, htype2, b2.charges, atoms, ff2.cutoff, α, b2.kprefactor, b2.Shost)
        @test isfinite(Bs)
        kmin = PureAdsorb.kmin_table(g2, htype2, b2.charges, atoms, ntypes)
        kT = T(PureAdsorb.KB) * T(298.15)
        θ = PureAdsorb.theta_F(T)
        # The safety term bounds the floating-point summation error of the actual computed ΔU
        # (2*n*eps(T)*(Bs+|cs|), n counting every summed term) plus pair_erfc_dev's own
        # approximation error (4e-6*Bs), on top of the +Bs that bounds the ideal (exact-formula)
        # energy -- the same margin build_rejection_tables computes.
        cs = b2.constant_offset[1]
        n = length(g2.sites) * natoms + length(b2.ks) + 8
        safety = 2 * n * eps(T) * (Bs + abs(cs)) + T(4.0e-6) * Bs
        margin = (θ + 2) * kT + safety + Bs - cs
        rho2 = [
            PureAdsorb.find_rho2(ff2.sigma[g2.types[a], t], ff2.epsilon[g2.types[a], t], kmin[a, t], margin, ff2.cutoff)
                for a in eachindex(g2.sites), t in 1:ntypes
        ]
        n_rejected, all_zero = count_falsepositives(g2, ff2, b2, htype2, rho2, atoms, natoms, α, kT, T; nposes = 100_000)
        @test all_zero
        @test n_rejected > 0
    end
end

@testitem "find_rho2 guards a non-positive margin and a zero epsilon" begin
    # A non-positive margin makes the bound's target unreachable (see the docstring); the guard
    # throws immediately instead of the upper-bracket search looping forever (unbounded as
    # margin <= 0 makes f(r) -> -margin >= 0 for large r).
    @test_throws "margin=0.0 must be positive" PureAdsorb.find_rho2(3.0, 0.01, -14.0, 0.0, Inf)
    @test_throws "margin=-5.0 must be positive" PureAdsorb.find_rho2(3.0, 0.01, -14.0, -5.0, Inf)

    # A zero epsilon (no Lennard-Jones repulsion) returns rho^2 = 0 directly, without searching:
    # with kmin = 0 too, f(r) has no root at all, so a search would halve r toward zero for
    # about 1080 iterations before exiting through a NaN.
    @test iszero(PureAdsorb.find_rho2(3.0, 0.0, 0.0, 25.0, Inf))
    @test iszero(PureAdsorb.find_rho2(3.0, 0.0, -14.0, 25.0, Inf))
end

@testitem "build_rejection_tables rejects a non-positive margin, naming the system" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    N = length(g.sites)
    g_compact = PureAdsorb.Guest{Float64, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
    # A large negative kT drives margin negative through the (θ+2)*kT term; the guard must
    # name the offending system rather than letting the search loop run unbounded.
    @test_throws "system 1" PureAdsorb.build_rejection_tables(b, g_compact, -1.0e30)
end

@testitem "find_r0 and find_rho2 clamp to the LJ cutoff" begin
    σ, ε = 3.0, 0.01
    α = PureAdsorb.ewald_alpha(12.0, 1.0e-6)
    K = -14.399645351950548

    r0_unclamped = PureAdsorb.find_r0(σ, ε, K, α, Inf)
    @test PureAdsorb.find_r0(σ, ε, K, α, 1.9) == 1.9                 # r_lj below the true root: clamped
    @test PureAdsorb.find_r0(σ, ε, K, α, 3.0) == r0_unclamped        # r_lj above the true root: unaffected

    margin = 25.0
    rho_unclamped = sqrt(PureAdsorb.find_rho2(σ, ε, K, margin, Inf))
    @test sqrt(PureAdsorb.find_rho2(σ, ε, K, margin, 1.0)) == 1.0    # r_lj below the true root: clamped
    @test sqrt(PureAdsorb.find_rho2(σ, ε, K, margin, 3.0)) == rho_unclamped   # r_lj above: unaffected
end

# A cubic 30 Å cell with one host atom (type B, q=+1) at the center, a one-site guest (type G,
# q=-1), σ=ε=0.01 for both types and a Lennard-Jones cutoff far shorter than the pair's natural
# length scale (`ρ_at`/`r0` both land beyond it).
@testsnippet C1Fixture begin
    using StaticArrays
    A_c1 = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw_c1 = Framework{Float64}(A_c1, [SVector(0.5, 0.5, 0.5)], ["B"], ["B"], [1.0])
    g_c1 = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(-1.0), 1.0, 1.0, 0.0)
    ewald_c1 = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    atom_pos_c1 = A_c1 * SVector(0.5, 0.5, 0.5)
end

@testitem "a Lennard-Jones cutoff shorter than the core radius" setup = [C1Fixture] begin
    using StaticArrays
    ff_c1 = ForceField(["G_", "B_"], [3.0, 3.0], [0.01, 0.01]; cutoff = 1.0, tail = false)
    b_c1 = FrameworkBatch([fw_c1], ff_c1, g_c1, ewald_c1)
    kT = PureAdsorb.KB * 298.15
    g_compact = PureAdsorb.Guest{Float64, 1}(g_c1.sites, SVector{1, Int}(b_c1.guest_types), g_c1.charges, g_c1.tc, g_c1.pc, g_c1.omega)
    rho2, = PureAdsorb.build_rejection_tables(b_c1, g_compact, kT)
    @test all(<=(ff_c1.cutoff^2), rho2)   # every rejection radius stays within the LJ cutoff

    # r = 1.2 Å is beyond the LJ cutoff (1.0 Å) but was inside the unclamped bug radius (1.74 Å):
    # the true energy there is purely attractive Coulomb, giving a huge Boltzmann weight, so a
    # pose there must not be rejected.
    q = SVector(0.0, 0.0, 0.0, 1.0)
    pos = atom_pos_c1 + SVector(1.2, 0.0, 0.0)
    natoms = b_c1.atom_offsets[2] - b_c1.atom_offsets[1]
    e = PureAdsorb.insertion_energy(
        pos, q, g_c1, b_c1.sigma, b_c1.epsilon, ff_c1.cutoff, b_c1.ewald_cutoff,
        b_c1.positions, b_c1.types, b_c1.charges, b_c1.atom_offsets[1], natoms,
        b_c1.cells[1], b_c1.invcells[1], b_c1.alphas[1], b_c1.ks, b_c1.kprefactor, b_c1.Shost
    )
    W = exp(-(e + b_c1.constant_offset[1]) / kT)
    @test W > 1.0e100

    r2 = widom(b_c1, g_c1; T = 298.15, ninsert = 500_000, seed = 17, nblocks = 4)
    r1 = PureAdsorb.widom_singlephase(b_c1, g_c1; T = 298.15, ninsert = 500_000, seed = 17, nblocks = 4)
    @test r1 == r2
end

@testitem "a pair zero beyond the Lennard-Jones cutoff" setup = [C1Fixture] begin
    using StaticArrays
    ff_c1b = ForceField(["G_", "B_"], [3.0, 3.0], [0.01, 0.01]; cutoff = 1.9, tail = false)
    b_c1b = FrameworkBatch([fw_c1], ff_c1b, g_c1, ewald_c1)
    α = b_c1b.alphas[1]
    K = PureAdsorb.KE * g_c1.charges[1] * fw_c1.charges[1]
    r0_unclamped = PureAdsorb.find_r0(3.0, 0.01, K, α, Inf)
    @test r0_unclamped > ff_c1b.cutoff   # this pair's true root lies beyond the LJ cutoff
    @test PureAdsorb.find_r0(3.0, 0.01, K, α, ff_c1b.cutoff) == ff_c1b.cutoff

    # At r = 2.0 (between the LJ cutoff 1.9 and the unclamped root 2.07) the Lennard-Jones term
    # is truncated away, so the true real-space pair term is purely the attractive Coulomb part --
    # strongly negative, not >= 0 as the unclamped envelope's "(0, r0]" claim would assert.
    r = 2.0
    E_lj = r^2 < ff_c1b.cutoff^2 ? PureAdsorb.lj_pair_energy(r^2, 3.0, 0.01) : 0.0
    E_sr = g_c1.charges[1] * fw_c1.charges[1] * PureAdsorb.pair_erfc_dev(α * r) / r
    u_ah_true = E_lj + PureAdsorb.KE * E_sr
    @test u_ah_true < -1.0

    r0_clamped = PureAdsorb.find_r0(3.0, 0.01, K, α, ff_c1b.cutoff)
    bound = -0.01 - abs(K) * PureAdsorb.pair_erfc_dev(α * r0_clamped) / r0_clamped
    @test u_ah_true >= bound   # hardcore_bound's per-pair envelope holds at the critical separation
end
