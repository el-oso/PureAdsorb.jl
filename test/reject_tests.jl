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
        r0 = PureAdsorb.find_r0(σ, ε, K, α)
        u0 = PureAdsorb.pair_energy(r0, σ, ε, K, α)
        @test abs(u0) < 1.0e-9 * ε
        @test PureAdsorb.pair_energy(prevfloat(r0) - 1.0e-8r0, σ, ε, K, α) > 0
        @test PureAdsorb.pair_energy(r0 * 1.001, σ, ε, K, α) < 0
    end
end

@testitem "find_r0 rejects a non-attractive or well-less pair" begin
    @test_throws "must be positive" PureAdsorb.find_r0(3.0, 0.0, -1.0, 0.2)
    @test_throws "must be negative" PureAdsorb.find_r0(3.0, 0.01, 1.0, 0.2)
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
    Bs = PureAdsorb.hardcore_bound(g, ff.sigma, ff.epsilon, b.positions, htype, b.charges, atoms, α, b.kprefactor, b.Shost)
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
        Bs = PureAdsorb.hardcore_bound(g2, ff2.sigma, ff2.epsilon, b2.positions, htype2, b2.charges, atoms, α, b2.kprefactor, b2.Shost)
        @test isfinite(Bs)
        kmin = PureAdsorb.kmin_table(g2, htype2, b2.charges, atoms, ntypes)
        kT = T(PureAdsorb.KB) * T(298.15)
        θ = PureAdsorb.theta_F(T)
        margin = (θ + 2) * kT + Bs - b2.constant_offset[1]
        rho2 = [
            PureAdsorb.find_rho2(ff2.sigma[g2.types[a], t], ff2.epsilon[g2.types[a], t], kmin[a, t], margin)
                for a in eachindex(g2.sites), t in 1:ntypes
        ]
        n_rejected, all_zero = count_falsepositives(g2, ff2, b2, htype2, rho2, atoms, natoms, α, kT, T; nposes = 100_000)
        @test all_zero
        @test n_rejected > 0
    end
end
