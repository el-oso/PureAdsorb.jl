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
