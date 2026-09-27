@testsnippet IsothermOracle begin
    using StaticArrays, LinearAlgebra, Random, Statistics

    # Same ideal-gas recipe `ideal_gas_tests.jl`/`gcmc_tests.jl` use: a single LJ type (epsilon=0),
    # tail=false, zero guest charges makes every interaction exactly zero, so loading is governed
    # only by the mu-VT combinatorial prefactor -- here, the closed-form `<N> = fV/kT` at every
    # pressure. Unlike those files' own guest (`tc = pc = 1`, fine when they bypass
    # `peng_robinson_fugacity` and pass `f == P` directly), `run_isotherm!` ALWAYS converts
    # pressure through the real equation of state, so `tc = 300 K`, `pc = 1e12 Pa` are chosen so
    # that stays honest: at this guest's critical pressure a test pressure of 1e5 Pa gives
    # `Pr = P/pc ~ 1e-7`, driving the fugacity coefficient to `phi ~= 1` (verified directly:
    # `peng_robinson_fugacity(1.01325e5, 273.15, 300.0, 1.0e12, 0.0).phi = 0.99999995`) so `f ~= P`
    # without disabling the equation-of-state path this test is exercising.
    function ideal_gas_isotherm_box(::Type{F}, V::F; cutoff = F(8)) where {F}
        L = F(cbrt(V))
        cell = SMatrix{3, 3, F}(L * I)
        fw = PureAdsorb.Framework{F}(cell, [SVector(F(0.5), F(0.5), F(0.5))], ["X1"], ["X"], [F(0)])
        ff = PureAdsorb.ForceField(["X_"], [F(1)], [F(0)]; cutoff, tail = false)
        g = PureAdsorb.Guest(SVector{1}(SVector(F(0), F(0), F(0))), SVector(1), SVector(F(0)), F(300), F(1.0e12), F(0))
        return fw, ff, g
    end
end

@testitem "combine_replicas reduces to the single chain's own error at R = 1" begin
    m, err = PureAdsorb.combine_replicas([3.7], [0.2])
    @test m == 3.7
    @test err == 0.2
end

@testitem "combine_replicas adds within-chain and across-replica variance" begin
    using Statistics
    values = [1.0, 2.0, 3.0, 4.0]
    errors = [0.1, 0.1, 0.1, 0.1]
    m, err = PureAdsorb.combine_replicas(values, errors)
    @test m == 2.5
    within = sum(abs2, errors) / length(values)^2
    between = var(values; corrected = true) / length(values)
    @test err ≈ sqrt(within + between)
    # A wider spread across replicas (the `between` term) must raise the combined error above
    # what the within-chain errors alone would give -- the whole point of folding it in.
    @test err > sqrt(within)
end

@testitem "combine_replicas rejects mismatched lengths and empty input" begin
    @test_throws DimensionMismatch PureAdsorb.combine_replicas([1.0, 2.0], [0.1])
    @test_throws ArgumentError PureAdsorb.combine_replicas(Float64[], Float64[])
end

@testitem "a 50-point isotherm batch builds near the cost of a single framework" setup = [IsothermOracle] begin
    F = Float64
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))

    # Warm up compilation before timing either build.
    FrameworkBatch([sc], ff, g, ewald; fullk = true)
    t_one = @elapsed FrameworkBatch([sc], ff, g, ewald; fullk = true)

    npress = 50
    nreplicas = 4
    nsys = npress * nreplicas
    t_iso = @elapsed begin
        biso = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        SystemState(biso, g, zeros(Int, nsys), ff; T = F(298.15), seed = 1, capacities = fill(60, nsys))
    end
    @test PureAdsorb.nframeworks(biso) == 1
    # Generous bound (not a tight timing assertion, which would be CI-noise-prone): dedup means
    # the isotherm-scale build must stay within a small constant factor of the single-system cost,
    # not grow with nsys=200.
    @test t_iso < 10 * t_one
end

@testitem "run_isotherm! reproduces the ideal-gas law at every pressure" setup = [IsothermOracle] begin
    F = Float64
    V = F(37219.0)   # Loschmidt's own volume, reused from ideal_gas_tests.jl
    fw, ff, g = ideal_gas_isotherm_box(F, V)
    T = F(273.15)
    kT = F(PureAdsorb.KB * 273.15)
    pressures = F[5.0e4, 1.01325e5, 2.0e5]   # <N> = PV/kT: 0.49, 1.0, 1.97 respectively
    # `min_cycle_length = 30`, well above the target loading (`ideal_gas_tests.jl`'s/`gcmc_tests.jl`'s
    # own convention for a low-loading GCMC test): without it, a cycle's length is recomputed from
    # the live occupancy, which stays near 1-2 here, so each cycle would attempt only 1-2 moves and
    # decorrelate far too slowly for `nblocks`-block averaging to see independent samples.
    iso = run_isotherm!(
        fw, ff, g, EwaldParams(cutoff = F(8), precision = F(1.0e-6)); T, pressures, nreplicas = 400,
        capacity = 40, n_warmup = 200, n_production = 500, n_audit = 701, step_trans = F(0.3), step_rot = F(0.3),
        exchange_prob = 1.0, seed = 7, min_cycle_length = 30
    )
    @test iso.pressure == pressures
    @test iso.nreplicas == 400
    for p in eachindex(pressures)
        target = pressures[p] * PureAdsorb.PASCAL * V / kT
        # max(8 sigma, 15% of the target), not the usual bare 5 sigma: `combine_replicas`' own
        # `between` term is an honest, autocorrelation-agnostic spread estimator, but 400 replicas
        # is still far fewer than `ideal_gas_tests.jl`'s own 1500-3000 independent chains, so a
        # bare-sigma band trips on ordinary finite-sample fluctuation more often than is useful
        # here. The 15% floor is sized to catch a broken combinatorial factor -- historically wrong
        # by an exact +1 shift or a 1.6e11 unit-conversion factor (`docs/superpowers/plans/
        # 2026-09-27-milestone-c-gcmc.md`, E1/R1) -- not to demand publication-grade precision from
        # a single batch run.
        @test isapprox(iso.loading[p], target; atol = max(8 * iso.loading_err[p], F(0.15) * target))
        @test iso.max_occupancy[p] <= iso.capacity[p]
    end
    # Monotone in pressure for a genuine ideal gas over this narrow range.
    @test issorted(iso.loading)
end

@testitem "run_isotherm! propagates mc_insert!'s hard capacity failure" setup = [IsothermOracle] begin
    F = Float64
    fw, ff, g = ideal_gas_isotherm_box(F, F(8000.0))
    @test_throws "hit capacity" run_isotherm!(
        fw, ff, g, EwaldParams(cutoff = F(8), precision = F(1.0e-6)); T = F(298.15), pressures = F[1.0e9],
        nreplicas = 1, capacity = 3, n_warmup = 0, n_production = 50, n_audit = 1000, step_trans = F(0.3),
        step_rot = F(0.3), exchange_prob = 1.0, seed = 4
    )
end
