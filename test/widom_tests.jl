@testitem "empty box gives ideal-gas statistics" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    r = widom(b, g; T = 300.0, ninsert = 10_000, seed = 1)[1]
    kT = PureAdsorb.KB * 300.0
    @test r.mu_ex ≈ 0 atol = 1.0e-12
    @test r.K_H ≈ 30.0^3 / kT
    @test r.q_st ≈ kT
    @test r.nsamples == 10_000
end

@testitem "empty box in Float32 stays Float32 throughout" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0f0, 0, 0, 0, 30.0f0, 0, 0, 0, 30.0f0)
    fw = Framework{Float32}(A, SVector{3, Float32}[], String[], String[], Float32[])
    ff = ForceField(["X_"], Float32[3.0], Float32[0.001]; cutoff = 12.0f0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0f0, 0.0f0, 0.0f0)), SVector(1), SVector(0.0f0), 1.0f0, 1.0f0, 0.0f0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0f0, precision = 1.0f-6))
    r = widom(b, g; T = 300.0f0, ninsert = 10_000, seed = 1)[1]
    kT = Float32(PureAdsorb.KB) * 300.0f0
    for field in (:mu_ex, :mu_ex_err, :K_H, :K_H_err, :q_st, :q_st_err)
        @test getfield(r, field) isa Float32
    end
    @test r.K_H ≈ 30.0f0^3 / kT
end

@testitem "a remainder-sized insertion count leaves no block empty" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    r = widom(b, g; T = 300.0, ninsert = 81, nblocks = 10, seed = 4)[1]
    @test isfinite(r.mu_ex_err) && isfinite(r.K_H_err)
    @test r.nsamples == 81
end

@testitem "a remainder-sized insertion count leaves no block empty across systems" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw, fw], ff, g, EwaldParams(cutoff = 12.0))
    rs = widom(b, g; T = 300.0, ninsert = 45, nblocks = 10, seed = 5)
    @test all(r -> isfinite(r.mu_ex_err) && isfinite(r.K_H_err), rs)
    @test sum(r -> r.nsamples, rs) == 45
end

@testitem "single LJ atom matches the radial integral" begin
    using StaticArrays, QuadGK
    L = 40.0
    A = SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L)
    fw = Framework{Float64}(A, [SVector(0.5, 0.5, 0.5)], ["X"], ["X"], [0.0])
    σ, ε, rc = 3.4, 0.0103, 12.0
    ff = ForceField(["X_"], [σ], [ε]; cutoff = rc, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = rc))
    Tk = 300.0; β = 1 / (PureAdsorb.KB * Tk)
    r = widom(b, g; T = Tk, ninsert = 4_000_000, seed = 2, nblocks = 20)[1]
    u(x) = 4ε * ((σ / x)^12 - (σ / x)^6)
    integral, _ = quadgk(x -> (1 - exp(-β * u(x))) * x^2, 1.0e-3, rc; rtol = 1.0e-10)
    expected = 1 - 4π * integral / L^3
    meanW = r.K_H * PureAdsorb.KB * Tk / L^3
    errW = r.K_H_err * PureAdsorb.KB * Tk / L^3
    @test abs(meanW - expected) < 4 * errW
end

@testitem "RUBTAK CO2 runs and is finite" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    r = widom(b, g; T = 298.15, ninsert = 2_000, seed = 3, nblocks = 4)[1]
    @test isfinite(r.mu_ex) && isfinite(r.K_H) && isfinite(r.q_st)
    @test r.K_H > 0
end

@testitem "widom rejects a batch with mismatched array axes" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    bad = PureAdsorb.FrameworkBatch(
        b.positions, push!(copy(b.types), Int32(1)), b.charges, b.atom_offsets, b.cells, b.invcells,
        b.volumes, b.alphas, b.ks, b.kprefactor, b.Shost, b.k_offsets, b.constant_offset, b.self_term_halfrange,
        b.ncells, b.reach, b.cell_offsets, b.cellgrid_offsets,
        b.sigma, b.epsilon, b.cutoff, b.ewald_cutoff, b.nsys
    )
    @test_throws DimensionMismatch widom(bad, g; T = 300.0, ninsert = 100)
end

@testitem "widom rejects too few insertions, too few blocks, and a zero chunk" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    @test_throws "ninsert" widom(b, g; T = 300.0, ninsert = 5, nblocks = 10)
    @test_throws "nblocks" widom(b, g; T = 300.0, ninsert = 100, nblocks = 1)
    @test_throws "chunk" widom(b, g; T = 300.0, ninsert = 100, chunk = 0)
end

@testitem "run keyword is validated against 1:per" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    @test_throws "run=0 must be ≥ 1" widom(b, g; T = 300.0, ninsert = 100, run = 0)
    @test_throws "run=1000 must be ≤ per=100" widom(b, g; T = 300.0, ninsert = 100, run = 1000)
end

@testitem "sys_of_index matches a brute-force per-system count" begin
    for (ninsert, nsys, run) in ((37, 4, 3), (100, 7, 5), (11, 3, 2), (1, 1, 1), (500, 13, 1), (45, 2, 5))
        counts = zeros(Int, nsys)
        for g in 1:ninsert
            s = PureAdsorb.sys_of_index(g, run, nsys)
            @test 1 <= s <= nsys
            counts[s] += 1
        end
        @test counts == PureAdsorb.system_counts(ninsert, nsys, run)
    end
end

@testitem "run-based assignment properties over a parameter grid" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    # Chunk sizes below are not multiples of run or nsys, ninsert values are prime or otherwise
    # not evenly divisible by nblocks/nsys/run/chunk, and one case has nsys > chunk.
    grid = (
        (ninsert = 1009, nsys = 3, nblocks = 2, chunk = 37, run = nothing),
        (ninsert = 770, nsys = 11, nblocks = 3, chunk = 23, run = 6),
        (ninsert = 97, nsys = 5, nblocks = 2, chunk = 6, run = 4),
        (ninsert = 53, nsys = 10, nblocks = 2, chunk = 3, run = 2),
        (ninsert = 257, nsys = 7, nblocks = 3, chunk = 100, run = nothing),
    )
    for p in grid
        b = FrameworkBatch(fill(fw, p.nsys), ff, g, EwaldParams(cutoff = 12.0))
        runlen = isnothing(p.run) ? PureAdsorb.default_run(p.ninsert, p.nsys) : p.run
        rs = isnothing(p.run) ?
            widom(b, g; T = 300.0, ninsert = p.ninsert, nblocks = p.nblocks, chunk = p.chunk, seed = 7) :
            widom(b, g; T = 300.0, ninsert = p.ninsert, nblocks = p.nblocks, chunk = p.chunk, seed = 7, run = p.run)
        ns = [r.nsamples for r in rs]
        @test sum(ns) == p.ninsert
        @test maximum(ns) - minimum(ns) <= runlen
        @test all(r -> isfinite(r.mu_ex) && isfinite(r.mu_ex_err) && isfinite(r.K_H_err), rs)
    end
end

@testitem "widom results do not depend on the chunk size" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector{1}(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch(fill(fw, 4), ff, g, EwaldParams(cutoff = 12.0))
    # The pose RNG stream and the per-insertion system/block assignment are both functions of
    # the global insertion index alone, so chunking cannot change the accumulated sums.
    chunks = (7, 13, 64, 512, 1009)
    ref = widom(b, g; T = 300.0, ninsert = 1009, nblocks = 3, run = 5, seed = 9, chunk = chunks[1])
    for c in chunks[2:end]
        rs = widom(b, g; T = 300.0, ninsert = 1009, nblocks = 3, run = 5, seed = 9, chunk = c)
        for (r1, r2) in zip(ref, rs)
            @test r1.mu_ex == r2.mu_ex
            @test r1.mu_ex_err == r2.mu_ex_err
            @test r1.K_H == r2.K_H
            @test r1.K_H_err == r2.K_H_err
            @test r1.q_st == r2.q_st
            @test r1.q_st_err == r2.q_st_err
            @test r1.nsamples == r2.nsamples
        end
    end
end

@testitem "widom results for a single system match a recorded reference within tolerance" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    r = widom(b, g; T = 298.15, ninsert = 2_000, seed = 3, nblocks = 4)[1]
    # `insertion_energy` approximates the guest self term by its orientation average, folded
    # into `constant_offset`; the literals below were computed under the exact per-insertion
    # formula, so `mu_ex`/`q_st` (energies) differ from them by at most `2·self_term_halfrange`:
    # `self_term_halfrange` is a spread estimate from 64 discrete orientations, and a continuous
    # orientation can exceed it by up to about 1.3x, so the bound carries the same 2x margin used
    # in "insertion_energy agrees with the full-sum reference within the self-term half-range".
    # `K_H` and every standard error are built from the Boltzmann weight `exp(-ΔU/kT)`, so they
    # differ by a relative amount of order `self_term_halfrange/kT`; the 5x margin below covers
    # the block-statistics propagation on top of that leading-order estimate.
    halfrange = b.self_term_halfrange[1]
    kT = PureAdsorb.KB * 298.15
    rtol = 5 * expm1(halfrange / kT)
    @test abs(r.mu_ex - (-0.14435951741921793)) <= 2 * halfrange + 1.0e-12 * abs(r.mu_ex)
    @test abs(r.q_st - 0.2563139447257997) <= 2 * halfrange + 1.0e-12 * abs(r.q_st)
    @test r.mu_ex_err ≈ 0.004098060610114027 rtol = rtol
    @test r.K_H ≈ 6.590899117050657e8 rtol = rtol
    @test r.K_H_err ≈ 1.0512729413942294e8 rtol = rtol
    @test r.q_st_err ≈ 0.0025536418864197745 rtol = rtol
end

@testitem "widom credits samples to the correct framework in a mixed batch" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    empty_A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    empty_fw = Framework{Float64}(empty_A, SVector{3, Float64}[], String[], String[], Float64[])
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)

    r_full = widom(FrameworkBatch([sc], ff, g, ewald), g; T = 298.15, ninsert = 2_000, seed = 11, nblocks = 4)[1]
    r_empty = widom(FrameworkBatch([empty_fw], ff, g, ewald), g; T = 298.15, ninsert = 2_000, seed = 12, nblocks = 4)[1]
    r_mixed = widom(FrameworkBatch([sc, empty_fw], ff, g, ewald), g; T = 298.15, ninsert = 4_000, seed = 13, nblocks = 4)

    se(a, b) = 3 * hypot(a, b)
    @test abs(r_mixed[1].mu_ex - r_full.mu_ex) < se(r_mixed[1].mu_ex_err, r_full.mu_ex_err)
    @test abs(r_mixed[1].K_H - r_full.K_H) < se(r_mixed[1].K_H_err, r_full.K_H_err)
    # An empty framework has no host atoms, so `insertion_energy`'s cross term is always zero
    # and ΔU is the same constant for every insertion, giving zero block-to-block variance and
    # hence `se == 0`; the two runs must then match exactly rather than within a nonzero
    # statistical margin.
    @test abs(r_mixed[2].mu_ex - r_empty.mu_ex) <= se(r_mixed[2].mu_ex_err, r_empty.mu_ex_err)
    @test abs(r_mixed[2].K_H - r_empty.K_H) <= se(r_mixed[2].K_H_err, r_empty.K_H_err)
    # RUBTAK and the empty box give very different physics, so a system/sample mix-up would show
    # up as a false pass above; this confirms the two references are actually distinguishable.
    @test abs(r_full.mu_ex - r_empty.mu_ex) > se(r_full.mu_ex_err, r_empty.mu_ex_err)
end
