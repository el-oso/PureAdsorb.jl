@testsnippet GCMCOracle begin
    using StaticArrays, LinearAlgebra, Random, Statistics, KernelAbstractions

    function gcmc_rubtak_setup(::Type{F}; ncounts, capacities = ncounts, seed = 3) where {F}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
        b = FrameworkBatch(fill(sc, length(ncounts)), ff, g, ewald; fullk = true)
        st = SystemState(b, g, ncounts, ff; T = F(298.15), seed, capacities)
        return b, st, g, ff
    end

    # A single LJ type (epsilon=0), tail=false, zero guest charges: every interaction is exactly
    # zero (`ideal_gas_tests.jl`'s own "interactions are genuinely all zero" pin applies here
    # identically, since this is the same recipe), so a chain's loading is governed only by the
    # μVT combinatorial prefactor.
    function ideal_gas_gcmc_box(::Type{F}, V::F; cutoff = F(8)) where {F}
        L = F(cbrt(V))
        cell = SMatrix{3, 3, F}(L * I)
        fw = PureAdsorb.Framework{F}(cell, [SVector(F(0.5), F(0.5), F(0.5))], ["X1"], ["X"], [F(0)])
        ff = PureAdsorb.ForceField(["X_"], [F(1)], [F(0)]; cutoff, tail = false)
        g = PureAdsorb.Guest(SVector{1}(SVector(F(0), F(0), F(0))), SVector(1), SVector(F(0)), F(1), F(1), F(0))
        ewald = EwaldParams(cutoff = cutoff, precision = F(1.0e-6))
        b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
        return b, ff, g
    end

    # Sample mean/variance of `Ns` alongside their Poisson sampling errors at the TARGET λ, as
    # `ideal_gas_tests.jl`'s own `ideal_gas_stats` does (duplicated here since TestItems.jl setups
    # are file-scoped).
    function ideal_gas_stats(Ns, target::Real)
        m = mean(Ns)
        v = var(Ns; corrected = false)
        se_mean = sqrt(target / length(Ns))
        se_var = sqrt((target + 2 * target^2) / length(Ns))
        return m, v, se_mean, se_var
    end

    # K independent single-system replicas, each run through `run_gcmc!` at `exchange_prob = 1`
    # (every cycle is an exchange attempt) rather than a raw `mc_exchange!` loop -- the same
    # "one system per replica" design `ideal_gas_tests.jl`'s own `ideal_gas_replicas` uses and for
    # the same reason (`mc_exchange!`'s one-coin-per-launch draw correlates every system in a
    # shared batch). At `λ = ⟨N⟩` near 1, P2's live cycle length stays near 1 while occupancy is
    # low, so a `min_cycle_length` of just 1 makes `ncycles` cycles far fewer total attempts than
    # `ideal_gas_tests.jl`'s own fixed `nattempts`; `min_cycle_length` here is instead set well
    # above the target loading so every cycle attempts a comparable number of moves regardless of
    # the instantaneous occupancy. `nblocks_per_chain = 1`, for the same reason
    # `ideal_gas_tests.jl`'s own `ideal_gas_replicas` sets it: `K` replicas of many cheap,
    # single-guest-site, non-interacting attempts is a CPU microbenchmark shape, not the GPU-scale
    # workload the exchange kernels' workgroup fan-out is for.
    function gcmc_ideal_gas_replicas(
            b::PureAdsorb.FrameworkBatch{F}, g::PureAdsorb.Guest{F}, ff, fugacity_pa::F, T::F;
            capacity::Integer, K::Integer, n_warmup::Integer, n_production::Integer, min_cycle_length::Integer,
            seed0::Integer = 5000
        ) where {F}
        Ns = Vector{Int}(undef, K)
        for k in 1:K
            st = SystemState(b, g, [0], ff; T = F(1), seed = seed0 + k, capacities = [capacity])
            run_gcmc!(
                b, st, g, ff; T, n_warmup, n_production, n_audit = n_warmup + n_production + 1,
                step_trans = [F(0.3)], step_rot = [F(0.3)], fugacity = [fugacity_pa], exchange_prob = F(1),
                seed = seed0 + 7 * k + 3, nblocks = 4, min_cycle_length, nblocks_per_chain = 1
            )
            Ns[k] = st.occupancy[1]
        end
        return Ns
    end
end

@testitem "disabling exchange reproduces Milestone B exactly for a fixed seed" setup = [GCMCOracle] begin
    b, st0, g, ff = gcmc_rubtak_setup(Float64; ncounts = [5, 3], seed = 7)
    st_nvt = deepcopy(st0)
    st_gcmc = deepcopy(st0)

    common = (
        T = 298.15, n_warmup = 5, n_production = 20, n_audit = 100, step_trans = [0.3, 0.3],
        step_rot = [0.3, 0.3], seed = 11, nblocks = 4,
    )
    nvt_results = run_nvt!(b, st_nvt, g, ff; common..., n_widom_per_cycle = 3, widom_seed = 999)
    gcmc_results = run_gcmc!(b, st_gcmc, g, ff; common..., fugacity = [1.0, 1.0], exchange_prob = 0.0)

    # `exchange_prob = 0` never draws the exchange coin (`&&`'s short-circuit) and the cycle
    # length -- recomputed every cycle from occupancy that here never changes -- equals
    # `run_nvt!`'s own fixed value, so the two drivers' move sequences coincide exactly: compare
    # VALUES of the shared `SystemState` fields, which stay the same `SVector{NMOVETYPES}` type in
    # both paths since exchange attempts here never touch `accepted`/`attempted` at all.
    @test st_nvt.refpoints == st_gcmc.refpoints
    @test st_nvt.orientations == st_gcmc.orientations
    @test st_nvt.Sk == st_gcmc.Sk
    @test st_nvt.energy == st_gcmc.energy
    @test st_nvt.accepted == st_gcmc.accepted
    @test st_nvt.attempted == st_gcmc.attempted
    @test st_gcmc.occupancy == st0.occupancy   # exchange never ran: occupancy unchanged
    for n in eachindex(nvt_results, gcmc_results)
        @test nvt_results[n].energy == gcmc_results[n].energy
        @test nvt_results[n].energy_err == gcmc_results[n].energy_err
    end
end

@testitem "the audit passes over a long mixed run and still catches an injected corruption" setup = [
    GCMCOracle,
] begin
    b, st, g, ff = gcmc_rubtak_setup(Float64; ncounts = [5], capacities = [40], seed = 21)
    results = run_gcmc!(
        b, st, g, ff; T = 298.15, n_warmup = 20, n_production = 200, n_audit = 25, step_trans = [0.3],
        step_rot = [0.3], fugacity = [2.0e4], exchange_prob = 0.5, seed = 5, nblocks = 5
    )   # must not throw
    @test results[1].max_occupancy <= results[1].capacity
    # `state.energy[1]` is the RUNNING value accumulated since the LAST internal audit, not
    # necessarily the very last production cycle -- unlike `exchange_tests.jl`'s own audit test,
    # which calls `audit_energy!` immediately after the chain stops, `run_gcmc!` may run several
    # more accepted moves after its last internal audit before returning. Compare within that
    # audit's own tolerance rather than expecting bit-identical equality.
    recomputed = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test isapprox(st.energy[1], recomputed; atol = 1.0e-8 * max(abs(recomputed), 1.0))

    st.energy[1] += 1.0
    @test_throws "energy audit failed" run_gcmc!(
        b, st, g, ff; T = 298.15, n_warmup = 0, n_production = 30, n_audit = 5, step_trans = [0.3],
        step_rot = [0.3], fugacity = [2.0e4], exchange_prob = 0.5, seed = 6, nblocks = 3
    )
end

@testitem "run_gcmc! propagates mc_insert!'s hard capacity failure" setup = [GCMCOracle] begin
    b, st, g, ff = gcmc_rubtak_setup(Float64; ncounts = [0], capacities = [3], seed = 9)
    @test_throws "hit capacity" run_gcmc!(
        b, st, g, ff; T = 298.15, n_warmup = 0, n_production = 50, n_audit = 1000, step_trans = [0.3],
        step_rot = [0.3], fugacity = [1.0e12], exchange_prob = 1.0, seed = 4, nblocks = 5
    )
end

@testitem "run_gcmc! reports loading, energy, q_st and corr(U,N) with sane shapes" setup = [GCMCOracle] begin
    b, st, g, ff = gcmc_rubtak_setup(Float64; ncounts = [3], capacities = [30], seed = 15)
    results = run_gcmc!(
        b, st, g, ff; T = 298.15, n_warmup = 20, n_production = 200, n_audit = 50, step_trans = [0.3],
        step_rot = [0.3], fugacity = [2.0e4], exchange_prob = 0.5, seed = 16, nblocks = 5
    )
    r = results[1]
    @test r.loading >= 0
    @test r.loading_err >= 0
    @test isfinite(r.energy)
    @test r.energy_err >= 0
    @test (-1 <= r.corr_UN <= 1) || isnan(r.corr_UN)
    @test r.max_occupancy <= r.capacity
    @test r.ncycles == 200
end

@testitem "the ideal-gas gate through the driver matches Loschmidt's number" setup = [GCMCOracle] begin
    F = Float64
    V = F(37219.0)
    b, ff, g = ideal_gas_gcmc_box(F, V)
    T = F(273.15)
    P_1atm = F(101325.0)   # f == P exactly for a genuine ideal gas
    Ns = gcmc_ideal_gas_replicas(
        b, g, ff, P_1atm, T; capacity = 50, K = 1500, n_warmup = 50, n_production = 200, min_cycle_length = 20
    )
    m, v, se_mean, se_var = ideal_gas_stats(Ns, 1.0)
    @test isapprox(m, 1.0; atol = 5 * se_mean)
    @test isapprox(v, 1.0; atol = 5 * se_var)
end
