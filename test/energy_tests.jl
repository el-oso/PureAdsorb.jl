@testitem "rotation preserves length and the unit quaternion is the identity" begin
    using StaticArrays, LinearAlgebra
    v = SVector(1.16, 0.0, 0.0)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, 0.0, 1.0), v) ≈ v
    q = normalize(SVector(0.3, -0.5, 0.7, 0.2))
    @test norm(PureAdsorb.rotate(q, v)) ≈ norm(v)
    # 90° about z maps x to y: q = (0, 0, sin45°, cos45°)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, sqrt(0.5), sqrt(0.5)), SVector(1.0, 0.0, 0.0)) ≈ SVector(0.0, 1.0, 0.0) atol = 1.0e-12
end

@testitem "insertion energy equals the full-system energy difference" begin
    using StaticArrays, LinearAlgebra, Random
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
    ΔU = PureAdsorb.insertion_energy(pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, 10.0, hpos, htype, hq, A, invA, α, ks, kpre, Sh)
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

@testitem "insertion energy uses independent LJ and Ewald cutoffs" begin
    using StaticArrays, LinearAlgebra, Random
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
    ΔU = PureAdsorb.insertion_energy(pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, ewald_cutoff, hpos, htype, hq, A, invA, α, ks, kpre, Sh)
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
