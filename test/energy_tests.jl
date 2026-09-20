@testsnippet CellListOracle begin
    using StaticArrays, LinearAlgebra, Random

    # Brute-force stand-in for `insertion_energy`'s real-space sum (loop over every host atom, no
    # cell list) plus the same reciprocal cross term over the SAME (already coupled) k-table, so
    # a comparison against `insertion_energy` isolates the cell list's correctness from E1's
    # k-vector truncation: both sides then agree to floating-point roundoff alone.
    function brute_insertion_energy(
            pos::SVector{3, T}, q::SVector{4, T}, guest::PureAdsorb.Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
            hpos, htype, hq, A, invA, alpha, ks, kprefactor, Shost
        ) where {T, N}
        rc_lj2 = cutoff * cutoff; rc_ew2 = ewald_cutoff * ewald_cutoff
        gsites = map(s -> PureAdsorb.rotate(q, s), guest.sites)
        E_lj = zero(T); E_sr = zero(T)
        for j in eachindex(hpos)
            Δ0 = PureAdsorb.minimum_image(A, invA, pos - hpos[j])
            for s in 1:N
                Δ = Δ0 + gsites[s]
                r2 = dot(Δ, Δ)
                (r2 < rc_lj2 || r2 < rc_ew2) || continue
                gt = guest.types[s]; gq = guest.charges[s]
                if r2 < rc_lj2
                    σ = sigma[gt, htype[j]]; ε = epsilon[gt, htype[j]]
                    x = (σ * σ / r2)^3
                    E_lj += 4 * ε * (x * x - x)
                end
                if r2 < rc_ew2
                    r = sqrt(r2)
                    E_sr += gq * hq[j] * PureAdsorb.erfc_dev(alpha * r) / r
                end
            end
        end
        E_lr = zero(T)
        for i in eachindex(ks)
            k = ks[i]
            Sg = zero(Complex{T})
            for s in 1:N
                Sg += guest.charges[s] * cis(dot(k, pos + gsites[s]))
            end
            E_lr += kprefactor[i] * 2 * real(conj(Shost[i]) * Sg)
        end
        return E_lj + T(PureAdsorb.KE) * (E_sr + E_lr)
    end

    # Test-only mirror of `insertion_energy`'s stencil that returns the visited (global, 1-based)
    # atom indices for one pose instead of an energy, for checking the visited set directly.
    function visited_atoms(pos::SVector{3}, atom_base::Integer, ncells::SVector{3, Int32}, reach::SVector{3, Int32}, cell_offsets, invA)
        f = invA * pos
        n1, n2, n3 = ncells[1], ncells[2], ncells[3]
        m1, m2, m3 = reach[1], reach[2], reach[3]
        h1 = PureAdsorb.home_cell_dev(f[1], n1); h2 = PureAdsorb.home_cell_dev(f[2], n2); h3 = PureAdsorb.home_cell_dev(f[3], n3)
        start1, count1 = PureAdsorb.stencil_start_count(h1, m1, n1)
        start2, count2 = PureAdsorb.stencil_start_count(h2, m2, n2)
        start3, count3 = PureAdsorb.stencil_start_count(h3, m3, n3)
        visited = Int[]
        for t3 in 0:(count3 - 1)
            c3 = PureAdsorb.wrap_cell(start3 + Int32(t3), n3)
            for t2 in 0:(count2 - 1)
                c2 = PureAdsorb.wrap_cell(start2 + Int32(t2), n2)
                for t1 in 0:(count1 - 1)
                    c1 = PureAdsorb.wrap_cell(start1 + Int32(t1), n1)
                    c = PureAdsorb.cell_linear(c1, c2, c3, n1, n2)
                    a0 = atom_base + cell_offsets[c + 1] + 1
                    a1 = atom_base + cell_offsets[c + 2]
                    for j in a0:a1
                        push!(visited, j)
                    end
                end
            end
        end
        return visited
    end
end

@testitem "insertion_energy matches a brute-force real-space sum" setup = [CellListOracle] begin
    # E3 replaced insertion_energy's cell-list stencil with a linear loop over every atom (the
    # cell list now serves only the hard-core rejection stage's phase-0 kernel), so this oracle
    # comparison no longer varies `cellwidth`: `insertion_energy` does not consume it any more.
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    sc_tri = replicate(fw, (3, 3, 3))
    L = 30.0
    cubic = Framework{Float64}(SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L), [SVector(0.5, 0.5, 0.5)], ["Zr"], ["Zr"], [1.0])
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)

    for (name, sc) in (("triclinic", sc_tri), ("cubic", cubic))
        b = FrameworkBatch([sc], ff, g, ewald)
        A = b.cells[1]; invA = b.invcells[1]; α = b.alphas[1]
        htype = b.compact_to_orig[b.types]   # map back to the force field's own type index
        natoms = b.atom_offsets[2] - b.atom_offsets[1]
        rng = Xoshiro(hash(name))
        # A face pose (exactly on the x=0 cell boundary) picked away from (0.5,0.5,0.5), which is
        # the cubic test framework's one host atom — a guest reference point exactly on top of a
        # host atom gives an (expected) infinite LJ repulsion in both `insertion_energy` and the
        # brute-force reference, which a NaN ≈ NaN test comparison would then wrongly fail.
        poses = SVector{3, Float64}[
            SVector(0.0, 0.0, 0.0), SVector(prevfloat(1.0), prevfloat(1.0), prevfloat(1.0)), SVector(0.0, 0.37, 0.61),
        ]
        for _ in 1:500
            push!(poses, rand(rng, SVector{3, Float64}))
        end
        for fpos in poses
            pos = A * fpos
            q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
            prod = PureAdsorb.insertion_energy(
                pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff,
                b.positions, htype, b.charges, b.atom_offsets[1], natoms,
                A, invA, α, b.ks, b.kprefactor, b.Shost
            )
            ref = brute_insertion_energy(
                pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff,
                b.positions, htype, b.charges, A, invA, α, b.ks, b.kprefactor, b.Shost
            )
            @test prod ≈ ref rtol = 1.0e-12
        end
    end
end

@testitem "insertion_energy matches brute force in Float32" setup = [CellListOracle] begin
    # An unreplicated (replication == (1,1,1)) framework, so this exercises Float32 without going
    # through `verify_replication`'s uncoupled-k-vector check, whose fixed 1e-8 absolute-charge
    # threshold is calibrated for Float64 roundoff and is a separate, pre-existing issue from E1,
    # not addressed here.
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    rng = Xoshiro(13)
    L = 30.0f0
    A0 = SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L)
    natoms = 40
    frac = [SVector{3, Float32}(rand(rng, Float32), rand(rng, Float32), rand(rng, Float32)) for _ in 1:natoms]
    charges = Float32[isodd(i) ? 0.3f0 : -0.3f0 for i in 1:natoms]
    fw = Framework{Float32}(A0, frac, fill("Zr", natoms), fill("Zr", natoms), charges)
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([fw], ff, g, ewald)
    A = b.cells[1]; invA = b.invcells[1]; α = b.alphas[1]
    htype = b.compact_to_orig[b.types]
    for _ in 1:200
        pos = A * rand(rng, SVector{3, Float32})
        q = normalize(rand(rng, SVector{4, Float32}) .- 0.5f0)
        prod = PureAdsorb.insertion_energy(
            pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff,
            b.positions, htype, b.charges, b.atom_offsets[1], natoms,
            A, invA, α, b.ks, b.kprefactor, b.Shost
        )
        ref = brute_insertion_energy(
            pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff,
            b.positions, htype, b.charges, A, invA, α, b.ks, b.kprefactor, b.Shost
        )
        @test prod ≈ ref rtol = 1.0f-4
    end
end

@testitem "rotation preserves length and the unit quaternion is the identity" begin
    using StaticArrays, LinearAlgebra
    v = SVector(1.16, 0.0, 0.0)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, 0.0, 1.0), v) ≈ v
    q = normalize(SVector(0.3, -0.5, 0.7, 0.2))
    @test norm(PureAdsorb.rotate(q, v)) ≈ norm(v)
    # 90° about z maps x to y: q = (0, 0, sin45°, cos45°)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, sqrt(0.5), sqrt(0.5)), SVector(1.0, 0.0, 0.0)) ≈ SVector(0.0, 1.0, 0.0) atol = 1.0e-12
end

@testitem "insertion_energy_reference equals the full-system energy difference" begin
    using StaticArrays, LinearAlgebra, Random
    # `insertion_energy` itself computes only the reciprocal-space cross term over the
    # host-coupled k-vectors and omits the guest self term, so this full-contract check — every
    # k-vector, cross plus self — runs against the oracle `insertion_energy_reference` instead.
    ff = ForceField(
        ["Zr_", "H_", "C_", "O_", "C", "O"], [2.78, 2.57, 3.43, 3.12, 2.8, 3.05],
        [0.003, 0.0019, 0.0046, 0.0026, 0.0023, 0.0068]; cutoff = 10.0, tail = false
    )
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    g0 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml")))
    g = PureAdsorb.Guest(g0.sites, SVector(5, 6, 6), g0.charges, g0.tc, g0.pc, g0.omega)
    sc = replicate(fw, PureAdsorb.min_multiplicity(fw.cell, 10.0))
    A = sc.cell; invA = inv(A); V = PureAdsorb.volume(A)
    hpos = PureAdsorb.cartesian(sc); hq = sc.charges
    htype = Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols]
    α = PureAdsorb.ewald_alpha(10.0, 1.0e-7)
    ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1.0e-7))
    Sh = PureAdsorb.structure_factor(ks, hpos, hq)
    kpre = [w[i] * PureAdsorb.pk(dot(ks[i], ks[i]), α, V) for i in eachindex(ks, w)]
    rng = Xoshiro(7)
    pos = A * rand(rng, SVector{3, Float64}); q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
    ΔU = PureAdsorb.insertion_energy_reference(pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, 10.0, hpos, htype, hq, A, invA, α, ks, kpre, Sh)
    gpos = [pos + PureAdsorb.rotate(q, s) for s in g.sites]
    mol_h = collect(1:length(hpos)); mol_g = fill(0, 3) # noidiom: hpos is a freshly built Vector, always one-based
    Ecoul = PureAdsorb.ewald_energy(A, vcat(hpos, gpos), vcat(hq, collect(g.charges)), vcat(mol_h, mol_g), α, 10.0, ks, w) -
        PureAdsorb.ewald_energy(A, hpos, hq, mol_h, α, 10.0, ks, w)
    Elj = let Elj = 0.0
        for (s, t) in zip(gpos, g.types), (h, ht) in zip(hpos, htype)
            r = norm(PureAdsorb.minimum_image(A, invA, s - h))
            r < 10.0 || continue
            x = (ff.sigma[t, ht] / r)^6
            Elj += 4ff.epsilon[t, ht] * (x^2 - x)
        end
        Elj
    end
    Eself = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    # ewald_energy's exclusion term subtracts only the erf part of an intramolecular pair
    # (see its docstring), so the reference decomposition must match that convention.
    Eexcl = -PureAdsorb.KE * sum(g.charges[a] * g.charges[b] * (1 - PureAdsorb.erfc_dev(α * norm(gpos[a] - gpos[b]))) / norm(gpos[a] - gpos[b]) for a in 1:3 for b in (a + 1):3)
    Enet = -PureAdsorb.KE * π / (2V * α^2) * ((sum(hq) + sum(g.charges))^2 - sum(hq)^2)
    @test ΔU ≈ Elj + Ecoul - Eself - Eexcl - Enet rtol = 1.0e-8
end

@testitem "insertion_energy_reference uses independent LJ and Ewald cutoffs" begin
    using StaticArrays, LinearAlgebra, Random
    # See the comment in "insertion_energy_reference equals the full-system energy difference"
    # above: this is the oracle's full-contract check, not `insertion_energy`'s own.
    ff = ForceField(
        ["Zr_", "H_", "C_", "O_", "C", "O"], [2.78, 2.57, 3.43, 3.12, 2.8, 3.05],
        [0.003, 0.0019, 0.0046, 0.0026, 0.0023, 0.0068]; cutoff = 6.0, tail = false
    )
    ewald_cutoff = 12.0
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    g0 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml")))
    g = PureAdsorb.Guest(g0.sites, SVector(5, 6, 6), g0.charges, g0.tc, g0.pc, g0.omega)
    sc = replicate(fw, (3, 3, 3))
    A = sc.cell; invA = inv(A); V = PureAdsorb.volume(A)
    hpos = PureAdsorb.cartesian(sc); hq = sc.charges
    htype = Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols]
    α = PureAdsorb.ewald_alpha(ewald_cutoff, 1.0e-7)
    ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1.0e-7))
    Sh = PureAdsorb.structure_factor(ks, hpos, hq)
    kpre = [w[i] * PureAdsorb.pk(dot(ks[i], ks[i]), α, V) for i in eachindex(ks, w)]
    rng = Xoshiro(11)
    pos = A * rand(rng, SVector{3, Float64}); q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
    ΔU = PureAdsorb.insertion_energy_reference(pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald_cutoff, hpos, htype, hq, A, invA, α, ks, kpre, Sh)
    gpos = [pos + PureAdsorb.rotate(q, s) for s in g.sites]
    mol_h = collect(1:length(hpos)); mol_g = fill(0, 3) # noidiom: hpos is a freshly built Vector, always one-based
    Ecoul = PureAdsorb.ewald_energy(A, vcat(hpos, gpos), vcat(hq, collect(g.charges)), vcat(mol_h, mol_g), α, ewald_cutoff, ks, w) -
        PureAdsorb.ewald_energy(A, hpos, hq, mol_h, α, ewald_cutoff, ks, w)
    Elj = let Elj = 0.0
        for (s, t) in zip(gpos, g.types), (h, ht) in zip(hpos, htype)
            r = norm(PureAdsorb.minimum_image(A, invA, s - h))
            r < ff.cutoff || continue
            x = (ff.sigma[t, ht] / r)^6
            Elj += 4ff.epsilon[t, ht] * (x^2 - x)
        end
        Elj
    end
    Eself = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    Eexcl = -PureAdsorb.KE * sum(g.charges[a] * g.charges[b] * (1 - PureAdsorb.erfc_dev(α * norm(gpos[a] - gpos[b]))) / norm(gpos[a] - gpos[b]) for a in 1:3 for b in (a + 1):3)
    Enet = -PureAdsorb.KE * π / (2V * α^2) * ((sum(hq) + sum(g.charges))^2 - sum(hq)^2)
    @test ΔU ≈ Elj + Ecoul - Eself - Eexcl - Enet rtol = 1.0e-8
end
