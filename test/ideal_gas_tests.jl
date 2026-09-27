@testsnippet IdealGasOracle begin
    using StaticArrays, LinearAlgebra, Random, Statistics, KernelAbstractions

    # A single LJ type (epsilon=0), tail=false, and zero guest charges: every host-guest and
    # guest-guest term -- Lennard-Jones, real-space Coulomb, the reciprocal cross and self terms,
    # `guest_self_terms`' exclusion correction, the tail correction and the net-charge correction
    # -- is then exactly zero (pinned directly by "ideal-gas setup's interactions are genuinely
    # all zero" below), so a chain's loading is governed only by the μVT combinatorial prefactor:
    # a true ideal gas. A cubic box of volume `V` stands in for a real framework since host
    # geometry plays no role once every interaction is zero.
    function ideal_gas_box(::Type{F}, V::F; cutoff = F(8)) where {F}
        L = F(cbrt(V))
        cell = SMatrix{3, 3, F}(L * I)
        fw = PureAdsorb.Framework{F}(cell, [SVector(F(0.5), F(0.5), F(0.5))], ["X1"], ["X"], [F(0)])
        ff = PureAdsorb.ForceField(["X_"], [F(1)], [F(0)]; cutoff, tail = false)
        g = PureAdsorb.Guest(SVector{1}(SVector(F(0), F(0), F(0))), SVector(1), SVector(F(0)), F(1), F(1), F(0))
        ewald = EwaldParams(cutoff = cutoff, precision = F(1.0e-6))
        b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
        return b, ff, g
    end

    # K independent replicas of an interaction-free μVT chain, each with its OWN SystemState seed
    # and OWN coin-flip RNG. `mc_exchange!` draws ONE fair coin per LAUNCH, shared across every
    # chain of a batch (R2/E2: needed for detailed balance), so a single batch of K systems would
    # give K CORRELATED snapshots of one shared coin-flip sequence at any one time -- verified
    # directly during this test's construction: at fV/kT close to an integer, insertion and
    # deletion acceptance both saturate near 0 or 1, so a shared-batch snapshot can show EXACTLY
    # zero cross-chain variance even after hundreds of attempts. Running each replica as its own
    # batch of one system, with its own independent coin sequence, avoids that correlation
    # entirely. Each replica runs `nattempts` mc_exchange! attempts from N=0 and reports its final
    # occupancy, treated by `ideal_gas_stats` below as one iid draw of the stationary distribution.
    # `nblocks_per_chain = 1` throughout: this replica loop makes millions of exchange attempts on
    # a single-guest-site, neutral, non-interacting system (`ideal_gas_box`'s own construction), so
    # every attempt's real- and reciprocal-space loops are already trivial -- `mc_step!`'s workgroup
    # fan-out exists to spread a GPU's SM-count of parallel work across a genuinely expensive
    # k-vector sum (`src/moves.jl`'s own comment on the μVT exchange moves), which buys nothing here
    # and only adds per-attempt CPU dispatch overhead at the default `target_blocks = 256`.
    function ideal_gas_replicas(
            b::PureAdsorb.FrameworkBatch{F}, g::PureAdsorb.Guest{F}, ff, fugacity_pa::F, kT::F;
            capacity::Integer, K::Integer, nattempts::Integer, seed0::Integer = 1000
        ) where {F}
        gc = PureAdsorb.compact_guest(b, g)
        N = length(g.sites)
        gt = SVector{N, Int}(b.guest_types)
        p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
        Ns = Vector{Int}(undef, K)
        for k in 1:K
            st = SystemState(b, g, [0], ff; T = F(1), seed = seed0 + k, capacities = [capacity])
            ws = PureAdsorb.MoveWorkspace(F, st.nsys, 1)
            rng = Xoshiro(seed0 + 7 * k + 3)
            for _ in 1:nattempts
                PureAdsorb.mc_exchange!(rng, ws, b, st, gc, gt, p, q, [fugacity_pa], kT; nblocks_per_chain = 1)
            end
            Ns[k] = st.occupancy[1]
        end
        return Ns
    end

    # Sample mean/variance of `Ns` alongside their Poisson sampling errors, evaluated at the
    # TARGET λ (the physically anchored value under test) rather than at the sample itself: for K
    # iid draws of a Poisson(λ) variable, Var(sample mean) = λ/K exactly, and Var(sample variance)
    # ≈ (μ4 - μ2²)/K = (λ + 2λ²)/K for large K, since a Poisson's 2nd and 4th central moments are
    # λ and λ + 3λ².
    function ideal_gas_stats(Ns, target::Real)
        m = mean(Ns)
        v = var(Ns; corrected = false)
        se_mean = sqrt(target / length(Ns))
        se_var = sqrt((target + 2 * target^2) / length(Ns))
        return m, v, se_mean, se_var
    end
end

# R1's own anchors, checked directly against physical constants before use below (the numbers
# a reader can re-derive without running any code): at 1 atm (101,325 Pa) and 273.15 K, an ideal
# gas occupies 22.414 L/mol -- one molecule per 37,219 Å³, Loschmidt's number. At 1e4 Pa and
# 298.15 K (kUPS's own example state point, `examples/mcmc_rigid.yaml`), RUBTAK-3x3x3's cell
# volume is 61,457 Å³ (`replicate(read_cif("RUBTAK.cif")), (3,3,3))`, measured directly), giving
# kT = 0.025693 eV and ⟨N⟩ = fV/kT = 0.1493. Both are used here as literals,
# NOT recomputed from the code's own f/V/kT (R1's own warning: that self-referential form pins
# the combinatorial factors but is blind to a uniform scale error in f -- exactly the class of bug
# E1 was, a missing Pa->eV/Å³ conversion worth a factor of 1.6e11).
@testitem "ideal-gas limit matches Loschmidt's number (1 atm, 273.15 K, 37219 Å³ gives ⟨N⟩=1)" setup = [
    IdealGasOracle,
] begin
    F = Float64
    V = F(37219.0)
    b, ff, g = ideal_gas_box(F, V)
    kT = F(PureAdsorb.KB * 273.15)
    P_1atm = F(101325.0)   # f == P exactly for a genuine ideal gas (φ = 1 by definition)
    Ns = ideal_gas_replicas(b, g, ff, P_1atm, kT; capacity = 50, K = 3000, nattempts = 2000)
    m, v, se_mean, se_var = ideal_gas_stats(Ns, 1.0)
    # 5*SE keeps this test's own false-positive rate low (a Gaussian tail beyond 5σ is ~3e-7)
    # while staying far inside the ~39σ gap the perturbed-factor run below measures.
    @test isapprox(m, 1.0; atol = 5 * se_mean)
    @test isapprox(v, 1.0; atol = 5 * se_var)   # Poisson: Var(N) == ⟨N⟩
end

# A wrong combinatorial factor produces a stable, energy-audit-passing chain that converges to the
# wrong loading (the design's own warning) -- demonstrated directly, once, rather than shipped as
# a permanent test path: redefining `log_insertion_prefactor` in a scratch session to use N where
# N+1 belongs (`log(f*V) - log(N) - log(kT)` instead of `... - log(N+1) - ...`) and rerunning the
# Loschmidt case above with identical seeds (K=1500, nattempts=1000) shifts the mean by an EXACT
# +1.0 (0.9393 -> 1.9393, a ~39σ effect against se_mean=0.0258) while leaving the variance
# bit-for-bit unchanged (0.848986 both times): the bug is algebraically equivalent to evaluating
# the correct prefactor one guest count too low, which for ΔU≡0 and identical RNG draws is exactly
# the correct trajectory offset by a permanent, energy-free extra guest -- an exact shift in the
# mean with no change in shape, not a vague "gets worse". `src/moves.jl` was never edited to
# obtain this; the substitution lives only in the throwaway session that measured it.

@testitem "ideal-gas limit matches kUPS's example (1e4 Pa, 298.15 K, RUBTAK-3x3x3 volume gives ⟨N⟩=0.1493)" setup = [
    IdealGasOracle,
] begin
    F = Float64
    V = F(61457.0)
    b, ff, g = ideal_gas_box(F, V)
    kT = F(PureAdsorb.KB * 298.15)
    @test isapprox(kT, 0.025693; atol = 1.0e-5)   # pins the literal against R1's own stated number
    P = F(1.0e4)
    Ns = ideal_gas_replicas(b, g, ff, P, kT; capacity = 25, K = 3000, nattempts = 1500)
    m, v, se_mean, se_var = ideal_gas_stats(Ns, 0.1493)
    @test isapprox(m, 0.1493; atol = 5 * se_mean)
    @test isapprox(v, 0.1493; atol = 5 * se_var)
end

@testitem "ideal-gas setup's interactions are genuinely all zero, not just the pair potentials" setup = [
    IdealGasOracle,
] begin
    F = Float64
    b, ff, g = ideal_gas_box(F, F(8000.0))
    gself, gexcl = PureAdsorb.guest_self_terms(g, b.alphas[1], b.ewald_cutoff)
    @test iszero(gself)
    @test iszero(gexcl)
    for Ng in 0:10
        # tail correction + net-charge correction + self/exclusion, exactly what
        # `mc_insert_kernel!`/`mc_delete_kernel!` add via `exchange_constant_coeffs`
        @test iszero(PureAdsorb.exchange_constant_term(ff, b, g, 1, Ng))
    end

    # Run a real mc_exchange!-driven chain and confirm the running energy total is EXACTLY zero
    # throughout -- not merely small -- which any nonzero residual pair, self, tail or net-charge
    # term would perturb away from.
    st = SystemState(b, g, [0], ff; T = F(1), seed = 99, capacities = [40])
    gc = PureAdsorb.compact_guest(b, g)
    gt = SVector{1, Int}(b.guest_types)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
    ws = PureAdsorb.MoveWorkspace(F, st.nsys, 1)
    rng = Xoshiro(42)
    maxocc = Ref(0)
    for _ in 1:5000
        PureAdsorb.mc_exchange!(rng, ws, b, st, gc, gt, p, q, [1.0e4], F(PureAdsorb.KB * 298.15); nblocks_per_chain = 1)
        maxocc[] = max(maxocc[], st.occupancy[1])
        @test st.energy[1] === 0.0
    end
    maxocc[] > 0 || error("test setup never inserted a guest; the energy check above would be vacuous")
end

# R3: nothing else in the ladder exercises the fugacity path -- every other exchange test runs at
# a pressure low enough that φ ≈ 1, so a caller passing raw pressure instead of `peng_robinson_fugacity`'s
# `f` would pass unnoticed. CO2 at 298 K, 5e6 Pa has φ well below 1 (Peng-Robinson, task 1), so the
# ideal (raw-P) and real (φP) predictions for ⟨N⟩ differ by ~24%, a many-σ discriminator.
@testitem "the fugacity path: CO2 at 5e6 Pa, 298 K gives ⟨N⟩ = φPV/kT, not PV/kT" setup = [
    IdealGasOracle,
] begin
    F = Float64
    ff_real = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    g_co2 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff_real; T = F)
    V = F(61457.0)   # RUBTAK-3x3x3's own volume, reused as a plain box (interactions are off)
    L = F(cbrt(V))
    cell = SMatrix{3, 3, F}(L * I)
    fw = PureAdsorb.Framework{F}(cell, [SVector(F(0.5), F(0.5), F(0.5))], ["C1"], ["C"], [F(0)])
    ff0 = PureAdsorb.ForceField(ff_real.names, ff_real.sigma, zero(ff_real.epsilon), ff_real.cutoff, false)
    # A single-site stand-in carrying CO2's real tc/pc/omega: geometry plays no role once every
    # interaction is zero, and one site keeps the guest-guest loop cheap at the ~56-guest loading
    # this state point equilibrates to.
    g0 = PureAdsorb.Guest(SVector{1}(SVector(F(0), F(0), F(0))), SVector(g_co2.types[1]), SVector(F(0)), g_co2.tc, g_co2.pc, g_co2.omega)
    ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
    b = FrameworkBatch([fw], ff0, g0, ewald; fullk = true)

    T = F(298.0)
    P = F(5.0e6)
    kT = F(PureAdsorb.KB * T)
    res = peng_robinson_fugacity(P, T, g0)
    @test res.phi < F(0.8)   # well below 1: this state point genuinely exercises the equation of state
    N_real = res.f * PureAdsorb.PASCAL * b.volumes[1] / kT
    N_ideal_wrong = P * PureAdsorb.PASCAL * b.volumes[1] / kT   # what passing raw P instead of f would give
    @test (N_ideal_wrong - N_real) / N_ideal_wrong > 0.2

    Ns = ideal_gas_replicas(b, g0, ff0, F(res.f), kT; capacity = 150, K = 1500, nattempts = 1200)
    m, _, se_mean, _ = ideal_gas_stats(Ns, N_real)
    @test isapprox(m, N_real; atol = 5 * se_mean)
    @test abs(m - N_ideal_wrong) > 5 * se_mean   # the "used raw P" alternative is a many-σ miss
end
