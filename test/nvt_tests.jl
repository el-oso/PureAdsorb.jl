@testsnippet NVTOracle begin
    using StaticArrays

    # kUPS's own `host/empty.cif`: a 30 Å cubic P1 cell with one non-interacting dummy site
    # (sigma=1, epsilon=0, charge=0), giving a pure-CO2-in-vacuum-with-PBC system whose host
    # contributes identically zero to every energy term (same recipe as `guest_tests.jl`'s
    # `GuestOracle.empty_box_setup`).
    function empty_box_setup(::Type{T}) where {T}
        ff0 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = T)
        names = vcat(ff0.names, ["X1_"])
        sigma = vcat([ff0.sigma[i, i] for i in eachindex(ff0.names)], T(1))
        epsilon = vcat([ff0.epsilon[i, i] for i in eachindex(ff0.names)], T(0))
        ff = ForceField(names, sigma, epsilon; cutoff = T(12), tail = true)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = T)
        A = SMatrix{3, 3}(T(30), T(0), T(0), T(0), T(30), T(0), T(0), T(0), T(30))
        fw = Framework{T}(A, [SVector(T(0), T(0), T(0))], ["X1"], ["X1"], [T(0)])
        return ff, g, fw
    end

    function rubtak_setup(::Type{T}; ncounts, seed = 3) where {T}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = T)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = T)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = T)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = T(12), precision = T(1.0e-6))
        b = FrameworkBatch(fill(sc, length(ncounts)), ff, g, ewald; fullk = true)
        st = SystemState(b, g, ncounts, ff; T = T(298.15), seed)
        return b, st, g, ff
    end
end

@testsnippet WidomChainOracle begin
    using StaticArrays, LinearAlgebra, Random

    # Independent, non-incremental total-energy oracle for a system with guests already present:
    # a literal Ewald sum over every host-plus-guest position (`ewald_energy`, a general primitive
    # `widom_chain_kernel!`'s incremental `Sk`-based formula never calls) minus the host-only
    # Ewald energy, plus a hand-written direct-pair Lennard-Jones loop over every host-guest and
    # guest-guest pair (no cell list, no shortcuts). Mirrors `guest_tests.jl`'s
    # `GuestOracle.oracle_total_energy`, duplicated here since TestItems.jl setups are
    # file-scoped: this validates `widom_chain_kernel!`'s test particle correctly seeing every
    # OTHER guest, not only the host, which neither B0 nor B1's kUPS comparison exercises (R5).
    function oracle_total_energy_with_guests(
            batch, refpoints, orientations, gr, guest::PureAdsorb.Guest{T, N}, ff, n::Integer, precision
        ) where {T, N}
        fw = batch.framework_of[n]
        A = batch.cells[fw]; invA = batch.invcells[fw]
        a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
        hpos = batch.positions[(a0 + 1):(a0 + natoms)]
        hq = batch.charges[(a0 + 1):(a0 + natoms)]
        htype = batch.compact_to_orig[batch.types[(a0 + 1):(a0 + natoms)]]
        Ng = length(gr)
        alpha = batch.alphas[fw]; ewald_cutoff = batch.ewald_cutoff
        ks, w, = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(alpha, precision))

        gpos = SVector{3, T}[]; gq = T[]; gmol = Int[]; gtype_orig = Int[]
        for (idx, i) in enumerate(gr), s in 1:N
            push!(gpos, refpoints[i] + PureAdsorb.rotate(orientations[i], guest.sites[s]))
            push!(gq, guest.charges[s]); push!(gmol, natoms + idx); push!(gtype_orig, guest.types[s])
        end
        allpos = vcat(hpos, gpos); allq = vcat(hq, gq); allmol = vcat(collect(1:natoms), gmol)
        Ecoul = PureAdsorb.ewald_energy(A, allpos, allq, allmol, alpha, ewald_cutoff, ks, w) -
            PureAdsorb.ewald_energy(A, hpos, hq, collect(1:natoms), alpha, ewald_cutoff, ks, w)

        rc2 = ff.cutoff^2
        Elj = zero(T)
        for a in eachindex(gpos)
            for h in eachindex(hpos)
                r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - hpos[h]))
                r * r < rc2 || continue
                sig = ff.sigma[gtype_orig[a], htype[h]]; eps = ff.epsilon[gtype_orig[a], htype[h]]
                x = (sig / r)^6
                Elj += 4 * eps * (x^2 - x)
            end
            for bb in eachindex(gpos)
                bb > a || continue
                gmol[a] == gmol[bb] && continue
                r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - gpos[bb]))
                r * r < rc2 || continue
                sig = ff.sigma[gtype_orig[a], gtype_orig[bb]]; eps = ff.epsilon[gtype_orig[a], gtype_orig[bb]]
                x = (sig / r)^6
                Elj += 4 * eps * (x^2 - x)
            end
        end

        ntypes_ff = length(ff.names)
        gcounts = zeros(Int, ntypes_ff)
        for t in guest.types
            gcounts[t] += 1
        end
        host_counts = zeros(Int, ntypes_ff)
        for t in htype
            host_counts[t] += 1
        end
        Etail = PureAdsorb.tail_delta(ff, host_counts, Ng .* gcounts, batch.volumes[fw])
        return Elj + Ecoul + Etail
    end

    # `widom_chain_kernel!`'s `ΔU` for a single fixed pose does NOT match
    # `oracle_total_energy_with_guests(after) - oracle_total_energy_with_guests(before)` exactly:
    # `insertion_constant_term`/`FrameworkBatch.constant_offset` folds in an
    # ORIENTATION-AVERAGED reciprocal self term (`self_mean`, `FrameworkBatch`'s own docstring),
    # not the exact per-orientation value, so any single insertion carries a real, orientation-
    # dependent error bounded by `self_term_halfrange` — a deliberate Milestone A cost tradeoff,
    # not a bug, and the reason Widom itself is validated by an AVERAGE over many insertions
    # rather than a per-pose comparison (`widom_tests.jl`). `self_mean` is defined as exactly the
    # average of the true per-orientation value, so the per-pose error has zero mean and washes
    # out in a Boltzmann-weighted average over enough random poses — the same quantity Widom's own
    # μ_ex estimator accumulates. This checks that average, for `nposes` random insertions into
    # `state`'s CURRENT configuration (host and every existing guest), against the SAME poses
    # evaluated through the independent oracle, within `nsem` combined standard errors.
    function check_widom_chain_oracle(b, st, g, ff, n::Integer, kT, nposes::Integer, precision, nsem, rng)
        gr = PureAdsorb.guest_range(st, n)
        N = length(g.sites)
        guest_c = PureAdsorb.compact_guest(b, g)
        guest_types = SVector{N, Int}(b.guest_types)
        const_term = PureAdsorb.insertion_constant_term(ff, b, g, n, PureAdsorb.nguests(st, n))
        E_before = oracle_total_energy_with_guests(b, st.refpoints, st.orientations, gr, g, ff, n, precision)
        w_ours = zeros(Float64, nposes)
        w_oracle = zeros(Float64, nposes)
        for m in 1:nposes
            fw = b.framework_of[n]
            pos = b.cells[fw] * rand(rng, SVector{3, Float64})
            q = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
            rpos = inv(b.cells[fw]) * pos
            ΔU = zeros(Float64, 1)
            PureAdsorb.widom_chain_kernel!(CPU())(
                ΔU, Int32[n], [rpos], [q], b, guest_c, guest_types, st.refpoints, st.orientations,
                st.guest_offsets, st.Sk, st.k_offsets, [const_term]; ndrange = 1
            )
            w_ours[m] = exp(-ΔU[1] / kT)
            refpoints2 = vcat(st.refpoints, [pos])
            orientations2 = vcat(st.orientations, [q])
            gr2 = gr.start:(gr.stop + 1)
            E_after = oracle_total_energy_with_guests(b, refpoints2, orientations2, gr2, g, ff, n, precision)
            w_oracle[m] = exp(-(E_after - E_before) / kT)
        end
        m_ours = sum(w_ours) / nposes
        m_oracle = sum(w_oracle) / nposes
        sem_ours = sqrt(sum(abs2, w_ours .- m_ours) / (nposes - 1) / nposes)
        sem_oracle = sqrt(sum(abs2, w_oracle .- m_oracle) / (nposes - 1) / nposes)
        combined_se = sqrt(sem_ours^2 + sem_oracle^2)
        @test abs(m_ours - m_oracle) <= nsem * combined_se
        return nothing
    end
end

@testitem "widom_chain_kernel! matches an independent oracle with other guests present: empty box (B1-style)" setup = [
    NVTOracle, WidomChainOracle,
] begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions
    ff, g, fw = empty_box_setup(Float64)
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-8)
    b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [10], ff; T = 298.15, seed = 31)
    rng = Xoshiro(41)
    kT = PureAdsorb.KB * 298.15
    check_widom_chain_oracle(b, st, g, ff, 1, kT, 2000, 1.0e-8, 5.0, rng)
end

@testitem "widom_chain_kernel! matches an independent oracle with other guests present: RUBTAK (B2-style)" tags = [
    :slow,
] setup = [NVTOracle, WidomChainOracle] begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions
    # precision=1e-8 (not kUPS's usual 1e-6, matching guest_tests.jl's `GuestOracle` convention)
    # so the batch's own k-vector table is converged enough to compare against the oracle at
    # matching alpha/kmax.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-8)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [10], ff; T = 298.15, seed = 31)
    rng = Xoshiro(41)
    kT = PureAdsorb.KB * 298.15
    # RUBTAK's oracle call is ~1 s each (a literal, non-incremental Ewald sum over all 3078+ host
    # atoms plus guests, no cell list): 150 poses keeps this tractable as a :slow item.
    check_widom_chain_oracle(b, st, g, ff, 1, kT, 150, 1.0e-8, 5.0, rng)
end

@testitem "N=0 reproduces Milestone A's widom exactly (to floating-point rounding)" setup = [NVTOracle] begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    nsys = 2
    b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, fill(0, nsys), ff; T = 298.15, seed = 42)

    n_widom_per_cycle = 25
    n_production = 8
    results = run_nvt!(
        b, deepcopy(st), g, ff; T = 298.15, n_warmup = 3, n_production, n_widom_per_cycle, n_audit = 2,
        step_trans = fill(0.3, nsys), step_rot = fill(0.3, nsys), seed = 42, nblocks = 4
    )

    ninsert = nsys * n_widom_per_cycle * n_production
    reference = PureAdsorb.widom_singlephase(b, g; T = 298.15, ninsert, seed = 42, run = 1, nblocks = 10)
    for (r, rf) in zip(results, reference)
        # Both estimators sum the SAME per-insertion Boltzmann weights in the SAME order, but into
        # differently-shaped block partitions (cycles here, insertions there), so their totals
        # agree to floating-point rounding rather than bit for bit -- the same standard this
        # project already applies to independently-grouped floating-point sums (e.g.
        # `moves_tests.jl`'s GPU-vs-CPU comparison).
        @test r.mu_ex ≈ rf.mu_ex rtol = 1.0e-10
        @test r.K_H ≈ rf.K_H rtol = 1.0e-10
        @test r.q_st ≈ rf.q_st rtol = 1.0e-10
        @test iszero(r.energy)
        @test all(iszero, r.acceptance)
    end
end

@testitem "cycle length matches kUPS: max(particle_count, min_cycle_length), one scalar across the batch" setup = [
    NVTOracle,
] begin
    # `nsteps_per_cycle` is not part of `run_nvt!`'s return value; this checks it indirectly via
    # `state.attempted`, which `run_nvt!` syncs from the device before returning.
    b, st, g, ff = rubtak_setup(Float64; ncounts = [3, 7], seed = 1)
    # `n_audit` exceeds `n_production` so no audit ever runs: this test checks the cycle-length
    # arithmetic only, at a move count far too small for `sk_audit_tolerance`'s bound (sized for
    # thousands of accepted moves) to be meaningful.
    run_nvt!(
        b, st, g, ff; T = 298.15, n_warmup = 0, n_production = 4, n_widom_per_cycle = 2, n_audit = 1_000_000,
        step_trans = fill(0.3, 2), step_rot = fill(0.3, 2), seed = 1, widom_seed = 1001, nblocks = 2
    )
    # nsteps_per_cycle = max(maximum(ncounts), min_cycle_length=1) = 7, shared across the batch
    # even though the two systems' own guest counts (3 and 7) differ: both systems get 7*4=28
    # move attempts, not 3*4=12 for the first.
    @test sum(st.attempted[1]) == 28
    @test sum(st.attempted[2]) == 28
end

@testitem "min_cycle_length floors the cycle length when every system is lightly loaded" setup = [NVTOracle] begin
    b, st, g, ff = rubtak_setup(Float64; ncounts = [1, 1], seed = 1)
    run_nvt!(
        b, st, g, ff; T = 298.15, n_warmup = 0, n_production = 4, n_widom_per_cycle = 1, n_audit = 1_000_000,
        step_trans = fill(0.3, 2), step_rot = fill(0.3, 2), min_cycle_length = 5, seed = 1, widom_seed = 1001, nblocks = 2
    )
    # max(maximum([1, 1]), min_cycle_length=5) = 5 attempts/cycle * 4 cycles = 20, not 1*4=4.
    @test sum(st.attempted[1]) == 20
    @test sum(st.attempted[2]) == 20
end

@testitem "energy audit passes over a long NVT-driven chain and still catches an injected corruption (Float64)" setup = [
    NVTOracle,
] begin
    b, st, g, ff = rubtak_setup(Float64; ncounts = [6], seed = 3)
    # 1500 cycles * 6 attempts/cycle = 9000 attempted moves before the single audit at the very
    # end (n_audit == n_production, n_warmup = 0, so the audit lands exactly on the last cycle
    # and `st` is synced to that point): comfortably past the move count where `sk_audit_tolerance`'s
    # bound (which scales with the accepted-move count, not the underlying `cis` evaluation error)
    # gets marginal. `widom_seed` differs from `seed`: `state` was built with `seed = 3`, and an
    # equal `widom_seed` would reproduce its own guest-placement draws (`run_nvt!`'s docstring).
    results = run_nvt!(
        b, st, g, ff; T = 298.15, n_warmup = 0, n_production = 1500, n_widom_per_cycle = 5, n_audit = 1500,
        step_trans = [0.3], step_rot = [0.4], seed = 3, widom_seed = 1003, nblocks = 5
    )
    @test isfinite(results[1].energy)
    @test all(>=(0), results[1].acceptance) && all(<=(1), results[1].acceptance)

    # `st` is the audit's own working copy and is synced to the run's final state (the audit
    # above landed on the last cycle); a deliberate corruption must still be caught.
    st.energy[1] += 1.0e-6
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, 1, 1)
end

@testitem "energy audit passes over a long NVT-driven chain and still catches an injected corruption (Float32)" setup = [
    NVTOracle,
] begin
    b, st, g, ff = rubtak_setup(Float32; ncounts = [6], seed = 3)
    results = run_nvt!(
        b, st, g, ff; T = 298.15f0, n_warmup = 0, n_production = 1500, n_widom_per_cycle = 5, n_audit = 1500,
        step_trans = [0.3f0], step_rot = [0.4f0], seed = 3, widom_seed = 1003, nblocks = 5
    )
    @test isfinite(results[1].energy)

    st.energy[1] += 1.0f-2
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, 1, 1)
end

@testitem "acceptance rates are sane and comparable to kUPS on the empty-box reference case" setup = [NVTOracle] begin
    ff, g, fw = empty_box_setup(Float64)
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
    st = SystemState(b, g, [50], ff; T = 298.15, seed = 3)
    results = run_nvt!(
        b, deepcopy(st), g, ff; T = 298.15, n_warmup = 20, n_production = 200, n_widom_per_cycle = 5, n_audit = 200,
        step_trans = [0.3], step_rot = [0.3], seed = 3, widom_seed = 1003, nblocks = 4
    )
    rate = results[1].acceptance
    # kUPS on the same case (empty host, 50 CO2, `nvt_co2_pressure_test.yaml`) measures 54.6%
    # translation, 70.7% rotation, 38.2% reinsertion (bench/results/README.md). Frozen, unadapted
    # step sizes (R4) mean these will not match exactly, but a translation/rotation/reinsertion
    # rate anywhere near 0% or 100% would mean something is actually wrong, not merely different.
    @test all(r -> 0.05 < r < 0.95, rate)
end

@testitem "block-averaged errors shrink as the cycle count grows" setup = [NVTOracle] begin
    ff, g, fw = empty_box_setup(Float64)
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
    st = SystemState(b, g, [20], ff; T = 298.15, seed = 11)

    # Independent `widom_seed`s (not one seed reused for both lengths, which would make `long`'s
    # first 50 cycles a verbatim prefix of `short`'s own sample): a 40x longer run at the same
    # block count.
    short = run_nvt!(
        b, deepcopy(st), g, ff; T = 298.15, n_warmup = 5, n_production = 50, n_widom_per_cycle = 4, n_audit = 1_000_000,
        step_trans = [0.3], step_rot = [0.3], seed = 11, widom_seed = 2001, nblocks = 5
    )
    long = run_nvt!(
        b, deepcopy(st), g, ff; T = 298.15, n_warmup = 5, n_production = 2000, n_widom_per_cycle = 4, n_audit = 1_000_000,
        step_trans = [0.3], step_rot = [0.3], seed = 11, widom_seed = 2002, nblocks = 5
    )
    for f in (r -> r.mu_ex_err, r -> r.q_st_err, r -> r.energy_err)
        @test f(short[1]) > 0 && isfinite(f(short[1]))
        @test f(long[1]) > 0 && isfinite(f(long[1]))
        # A 40x longer run (same block count) is not guaranteed to shrink the error by exactly
        # sqrt(40), only for it to trend down; a factor of 2 leaves ample margin over run-to-run
        # noise while still catching an estimator that does not improve with more data at all.
        @test f(long[1]) < f(short[1]) / 2
    end
end

@testitem "insertion_constant_term reduces to batch.constant_offset at Ng=0" setup = [NVTOracle] begin
    b, st, g, ff = rubtak_setup(Float64; ncounts = [0, 4])
    @test PureAdsorb.insertion_constant_term(ff, b, g, 1, 0) === b.constant_offset[1]
    # At Ng>0 the result differs from the Ng=0 baseline (a real tail/net-charge correction from
    # the extra guests), and is finite.
    ct = PureAdsorb.insertion_constant_term(ff, b, g, 2, 4)
    @test isfinite(ct)
    @test ct != b.constant_offset[b.framework_of[2]]
end

@testitem "widom_chain_kernel! adds zero guest-guest contribution when a system holds no guests" setup = [
    NVTOracle,
] begin
    using StaticArrays, KernelAbstractions
    b, st, g, ff = rubtak_setup(Float64; ncounts = [0])
    guest_c = PureAdsorb.compact_guest(b, g)
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    const_term = Float64[PureAdsorb.insertion_constant_term(ff, b, g, 1, 0)]
    sys_of = Int32[1]
    rpos = [SVector(0.1, 0.2, 0.3)]
    quat = [SVector(0.0, 0.0, 0.0, 1.0)]
    ΔU = zeros(Float64, 1)
    PureAdsorb.widom_chain_kernel!(CPU())(
        ΔU, sys_of, rpos, quat, b, guest_c, guest_types, st.refpoints, st.orientations, st.guest_offsets, st.Sk,
        st.k_offsets, const_term; ndrange = 1
    )
    pos = b.cells[1] * rpos[1]
    e = PureAdsorb.insertion_energy(
        pos, quat[1], guest_c, b.sigma, b.epsilon, b.cutoff, b.ewald_cutoff, b.positions, b.types, b.charges,
        b.atom_offsets[1], b.atom_offsets[2] - b.atom_offsets[1], b.cells[1], b.invcells[1], b.alphas[1],
        b.ks, b.kprefactor, b.Shost
    )
    @test ΔU[1] ≈ e + const_term[1]
end

@testitem "run_nvt! rejects a guest that does not match batch/state" setup = [NVTOracle] begin
    b, st, g, ff = rubtak_setup(Float64; ncounts = [2])
    other = PureAdsorb.Guest(g.sites, g.types, g.charges .+ 1, g.tc, g.pc, g.omega)
    @test_throws ArgumentError run_nvt!(
        b, st, other, ff; T = 298.15, n_warmup = 0, n_production = 1, n_widom_per_cycle = 1, n_audit = 1,
        step_trans = [0.3], step_rot = [0.3]
    )
end
