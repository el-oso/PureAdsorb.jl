@testitem "batch layout and offsets" begin
    using LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    @test b.nsys == 2
    @test b.atom_offsets == Int32[0, 3078, 6156]
    @test length(b.ks) == 2 * (b.k_offsets[2] - b.k_offsets[1])
    @test b.constant_offset[1] == b.constant_offset[2]
    # Atoms are stored sorted by cell (E2), not CIF order, so position 1 is no longer
    # necessarily the CIF's first atom (Zr) — only the per-system multiset of types survives.
    @test sort(b.types[1:3078]) == sort(Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols])
    @test b.ewald_cutoff == 12.0
    i = 1
    k = b.ks[i]
    # `kvectors` weights n1 == 0 by 1 and n1 > 0 by 2; n1 is k's component along the first
    # reciprocal lattice vector, recovered by solving k = B * (n1, n2, n3).
    n1 = round(Int, (PureAdsorb.reciprocal_basis(b.cells[1]) \ k)[1])
    w = iszero(n1) ? 1.0 : 2.0
    @test b.kprefactor[i] ≈ w * PureAdsorb.pk(dot(k, k), b.alphas[1], b.volumes[1])
end

@testitem "batch rejects a cell too small for the cutoff" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    @test_throws "replicate" FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
end

@testitem "an uncharged guest builds no reciprocal-space table" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g0 = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    gq = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(1.0), 1.0, 1.0, 0.0)
    nsys = 2
    b0 = FrameworkBatch(fill(fw, nsys), ff, g0, EwaldParams(cutoff = 12.0))
    bq = FrameworkBatch(fill(fw, nsys), ff, gq, EwaldParams(cutoff = 12.0))
    @test isempty(b0.ks)
    @test b0.k_offsets == zeros(Int32, nsys + 1)
    @test !isempty(bq.ks)
    @test all(iszero, b0.self_term_halfrange)   # no charge, no orientation dependence
end

@testitem "FrameworkBatch keeps only the k-vectors coupled to the replication" begin
    using LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald)
    @test length(b.ks) == 190

    α = PureAdsorb.ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = PureAdsorb.ewald_kmax(α, ewald.precision)
    hpos = PureAdsorb.cartesian(sc)
    ks_full, kpref_full, Sh_full, coeffs = PureAdsorb.full_ktables(sc.cell, hpos, sc.charges, α, kmax)
    dropped = [i for i in eachindex(coeffs) if !all(iszero, mod.(coeffs[i], sc.replication))]
    @test length(dropped) == length(ks_full) - 190
    @test all(i -> abs(Sh_full[i]) < 1.0e-9, dropped)
end

@testitem "an unreplicated framework keeps every k-vector" begin
    using StaticArrays
    L = 30.0
    A = SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L)
    fw = Framework{Float64}(A, [SVector(0.5, 0.5, 0.5)], ["Zr"], ["Zr"], [1.0])
    @test fw.replication == (1, 1, 1)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([fw], ff, g, ewald)
    α = PureAdsorb.ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = PureAdsorb.ewald_kmax(α, ewald.precision)
    ks_full, = PureAdsorb.full_ktables(fw.cell, PureAdsorb.cartesian(fw), fw.charges, α, kmax)
    @test length(b.ks) == length(ks_full)
end

@testitem "FrameworkBatch verifies a claimed replication against the host structure factor" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    gq = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(1.0), 1.0, 1.0, 0.0)
    # Two atoms that are NOT translational copies of each other under (2,1,1): a genuine
    # `replicate` output would place a matching atom at frac (x+0.5, y, z) with the same charge.
    frac = [SVector(0.1, 0.1, 0.1), SVector(0.6, 0.35, 0.7)]
    fw = Framework{Float64}(A, frac, ["X", "X"], ["X", "X"], [1.0, -1.0], (2, 1, 1))
    @test_throws "replication" FrameworkBatch([fw], ff, gq, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
end

@testitem "a genuine replicate output passes the replication check" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    @test b.nsys == 1
end

@testitem "self-term half-range guard trips for a strongly polar guest in a small cell" begin
    using StaticArrays
    cutoff = 3.0
    r_guest = 0.3
    L = 2 * (cutoff + r_guest)   # smallest cell satisfying min_multiplicity(cell, cutoff+r_guest) == (1,1,1)
    A = SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L)
    fw = Framework{Float64}(A, [SVector(0.5, 0.5, 0.5)], ["Zr"], ["Zr"], [1.0])
    ff = ForceField(["Zr_"], [2.78], [0.003]; cutoff = cutoff, tail = false)
    # A strongly polar two-site guest: charges ±20, 0.3 Å apart (same dipole moment, 6 e·Å, as a
    # ±2-charge, 3 Å pair, but a much smaller r_guest so the cell above stays small).
    g = PureAdsorb.Guest(
        SVector(SVector(0.0, 0.0, 0.0), SVector(r_guest, 0.0, 0.0)),
        SVector(1, 1), SVector(20.0, -20.0), 1.0, 1.0, 0.0
    )
    @test_throws "self-term half-range" FrameworkBatch([fw], ff, g, EwaldParams(cutoff = cutoff, precision = 1.0e-6))
end

@testitem "cell-list construction places atoms by their wrapped fractional cell" begin
    using StaticArrays, LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); cellwidth = 3.0)
    n = b.ncells[1]
    ncell = Int(n[1]) * Int(n[2]) * Int(n[3])
    @test length(b.cell_offsets) == ncell + 1
    @test iszero(b.cell_offsets[1])
    @test b.cell_offsets[end] == length(b.positions)
    @test issorted(b.cell_offsets)
    invA = b.invcells[1]
    for c in 0:(ncell - 1)
        a0, a1 = b.cell_offsets[c + 1] + 1, b.cell_offsets[c + 2]
        for j in a0:a1
            f = invA * b.positions[j]
            @test PureAdsorb.cell_linear(PureAdsorb.cell_coord(f, n), n) == c
        end
    end
end

@testitem "cell sort is a permutation of the original atoms" begin
    using LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    orig_pos = PureAdsorb.cartesian(sc)
    orig_type = Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols]
    key(p, t, q) = (p[1], p[2], p[3], Int(t), q)
    orig_keys = sort([key(orig_pos[i], orig_type[i], sc.charges[i]) for i in eachindex(orig_pos)])
    sorted_keys = sort([key(b.positions[i], b.types[i], b.charges[i]) for i in eachindex(b.positions)])
    @test orig_keys == sorted_keys
end

@testitem "cell-list ragged concatenation for two different frameworks" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    L = 30.0
    cubic = Framework{Float64}(
        SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L),
        [SVector(0.25, 0.25, 0.25), SVector(0.75, 0.75, 0.75)], ["Zr", "Zr"], ["Zr", "Zr"], [1.0, -1.0]
    )
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc, cubic], ff, g, ewald; cellwidth = 3.0)
    @test b.nsys == 2
    n1 = b.ncells[1]; n2 = b.ncells[2]
    @test n1 == PureAdsorb.grid_dims(PureAdsorb.perpendicular_lengths(sc.cell), 3.0)
    @test n2 == PureAdsorb.grid_dims(PureAdsorb.perpendicular_lengths(cubic.cell), 3.0)
    ncell1 = Int(n1[1]) * Int(n1[2]) * Int(n1[3])
    ncell2 = Int(n2[1]) * Int(n2[2]) * Int(n2[3])
    @test b.cellgrid_offsets == Int32[0, ncell1 + 1, ncell1 + 1 + ncell2 + 1]
    @test length(b.cell_offsets) == b.cellgrid_offsets[end]
    natoms1 = b.atom_offsets[2] - b.atom_offsets[1]
    natoms2 = b.atom_offsets[3] - b.atom_offsets[2]
    @test b.cell_offsets[b.cellgrid_offsets[2]] == natoms1
    @test b.cell_offsets[b.cellgrid_offsets[3]] == natoms2
end

@testitem "constant offset matches the pose-independent terms" begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald)
    α = b.alphas[1]; V = b.volumes[1]
    self = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    excl = -PureAdsorb.KE * sum(
        g.charges[a] * g.charges[c] * (1 - PureAdsorb.erfc_dev(α * norm(g.sites[a] - g.sites[c]))) / norm(g.sites[a] - g.sites[c])
            for a in 1:3 for c in (a + 1):3
    )
    counts = [count(==(t), b.types) for t in eachindex(ff.names)]
    gcounts = [count(==(t), g.types) for t in eachindex(ff.names)]
    tail = PureAdsorb.tail_delta(ff, counts, gcounts, V)

    # Guest self term over the full k set, at the same 64 fixed orientations `FrameworkBatch`
    # draws from `Xoshiro(0x5e1f)`, computed directly from `full_ktables` rather than through
    # `FrameworkBatch`'s internals.
    kmax = PureAdsorb.ewald_kmax(α, ewald.precision)
    hpos = PureAdsorb.cartesian(sc)
    ks_full, kpref_full, = PureAdsorb.full_ktables(sc.cell, hpos, sc.charges, α, kmax)
    rng = Xoshiro(0x5e1f)
    samples = Float64[]
    for _ in 1:64
        q = PureAdsorb.shoemake_quaternion(rng, Float64)
        gsites = [PureAdsorb.rotate(q, s) for s in g.sites]
        acc = 0.0
        for i in eachindex(ks_full, kpref_full)
            Sg = sum(g.charges[s] * cis(dot(ks_full[i], gsites[s])) for s in eachindex(gsites, g.charges))
            acc += kpref_full[i] * abs2(Sg)
        end
        push!(samples, PureAdsorb.KE * acc)
    end
    self_mean = sum(samples) / 64
    @test b.self_term_halfrange[1] ≈ (maximum(samples) - minimum(samples)) / 2
    @test b.constant_offset[1] ≈ self + excl + tail + self_mean     # CO2 is neutral: no net-charge term
end
