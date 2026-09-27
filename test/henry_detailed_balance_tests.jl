@testsnippet HenryDetailedBalance begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions

    function rubtak_co2_henry_setup(::Type{F}, nsys::Integer) where {F}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
        b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        return b, ff, g, sc, ewald
    end

    # Pooled ratio of two block-summed quantities with a delete-one-BLOCK jackknife error, the same
    # pooled-plus-jackknife shape `fluctuation_qst` (`src/gcmc.jl`) already uses for a nonlinear
    # statistic: `numer`/`denom` are `nblocks` PARTIAL SUMS (not per-block ratios, which would carry
    # the Jensen-inequality bias `fluctuation_qst`'s own docstring warns against), and the estimator
    # is the SINGLE pooled ratio of their totals. A block whose own leave-one-out denominator total
    # is zero (every sample of that quantity came from that one block) is dropped from the jackknife
    # rather than dividing by zero; at least two surviving blocks are required for a defined error.
    function pooled_ratio_jackknife(numer::Vector{Float64}, denom::Vector{Float64})
        nblocks = length(numer)
        length(denom) == nblocks || throw(DimensionMismatch("pooled_ratio_jackknife: numer/denom length mismatch"))
        tot_n = sum(numer)
        tot_d = sum(denom)
        tot_d > 0 || throw(ArgumentError("pooled_ratio_jackknife: zero total denominator"))
        ratio = tot_n / tot_d
        valid = [b for b in 1:nblocks if (tot_d - denom[b]) > 0]
        length(valid) >= 2 ||
            throw(ArgumentError("pooled_ratio_jackknife: fewer than 2 blocks have a nonzero leave-one-out denominator"))
        loo = [(tot_n - numer[b]) / (tot_d - denom[b]) for b in valid]
        m = sum(loo) / length(loo)
        err = sqrt((length(loo) - 1) / length(loo) * sum(abs2, loo .- m))
        return ratio, err
    end

    # Runs `nsys` replica GCMC chains (`mc_step!`/`mc_exchange!`, `src/gcmc.jl`'s own cycle
    # structure and shared-movetype/shared-coin draw sequence) at a FIXED fugacity, and on every
    # PRODUCTION cycle additionally performs `n_widom_per_cycle` Widom test-particle insertions per
    # chain (`widom_chain_kernel!`, Milestone B) into the configuration as it then stands -- exactly
    # `run_nvt!`'s own trick, generalized to whatever occupancy the chain currently holds rather than
    # only zero. Every sample (an occupancy visit, or a Widom insertion's Boltzmann weight) is
    # binned by the CURRENT occupancy `N` (`0:maxN`) and by production-cycle block, giving, per `N`:
    # `sN_visits[N+1, :]` (how many (chain, cycle) pairs held exactly `N` guests, one block-count
    # per block) and `sW[N+1, :]`/`sWcount[N+1, :]` (the Boltzmann weights of every Widom insertion
    # made while some chain held `N` guests, and how many such insertions there were). Both are
    # pooled-plus-jackknife inputs (`pooled_ratio_jackknife`): `P(N+1)/P(N)` is the ratio of two
    # VISIT-COUNT totals (the shared per-cycle normalization cancels), and `⟨exp(-ΔU_ins/kT)⟩_N` is
    # the ratio of `sW`'s total to `sWcount`'s total.
    #
    # `insertion_constant_term` (`src/nvt.jl`) is recomputed fresh every PRODUCTION cycle from the
    # chain's live occupancy (E4: it depends on `N`, and `N` drifts here, unlike NVT), never cached
    # across cycles.
    function gcmc_widom_by_occupancy(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N}, ff::PureAdsorb.ForceField{F};
            T, n_warmup::Integer, n_production::Integer, n_widom_per_cycle::Integer, step_trans, step_rot, fugacity,
            exchange_prob::Real = 0.5, min_cycle_length::Integer = 1, seed::Integer = 0, widom_seed::Integer = seed + 1_000_003,
            backend = CPU(), groupsize::Integer = PureAdsorb.DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = 1,
            maxN::Integer = 40, nblocks::Integer = 10
        ) where {F, N}
        nsys = state.nsys
        kT = F(PureAdsorb.KB * T)
        fug = F.(fugacity)
        guest_c = PureAdsorb.compact_guest(batch, guest)
        guest_types = SVector{N, Int}(batch.guest_types)
        p_host, q_host = PureAdsorb.exchange_constant_coeffs(ff, batch, guest_c)
        db = PureAdsorb.adapt(backend, batch)
        dst = PureAdsorb.adapt(backend, state)
        dstep_trans = PureAdsorb.adapt(backend, F.(step_trans))
        dstep_rot = PureAdsorb.adapt(backend, F.(step_rot))
        dp = PureAdsorb.adapt(backend, p_host)
        dq = PureAdsorb.adapt(backend, q_host)
        ws = PureAdsorb.MoveWorkspace(F, nsys, nblocks_per_chain; backend)
        rng_move = Xoshiro(seed)
        rng_widom = Xoshiro(widom_seed)
        movechoices = (PureAdsorb.MOVE_TRANSLATION, PureAdsorb.MOVE_ROTATION, PureAdsorb.MOVE_REINSERTION)

        occ_host = Vector{Int32}(undef, nsys)
        sN_visits = zeros(Int, maxN + 1, nblocks)
        sW = zeros(Float64, maxN + 1, nblocks)
        sWcount = zeros(Int, maxN + 1, nblocks)

        ninsert_per_cycle = nsys * n_widom_per_cycle
        sys_of = Vector{Int32}(undef, ninsert_per_cycle)
        for iter in 0:(n_widom_per_cycle - 1), s in 1:nsys
            sys_of[iter * nsys + s] = s
        end
        dsys = PureAdsorb.adapt(backend, sys_of)
        rpos = Vector{SVector{3, F}}(undef, ninsert_per_cycle)
        quat = Vector{SVector{4, F}}(undef, ninsert_per_cycle)
        drpos = PureAdsorb.adapt(backend, rpos)
        dquat = PureAdsorb.adapt(backend, quat)
        dΔU = PureAdsorb.adapt(backend, zeros(F, ninsert_per_cycle))
        ΔU_h = Vector{F}(undef, ninsert_per_cycle)

        function run_cycle!(block_idx::Union{Nothing, Int})
            copyto!(occ_host, dst.occupancy)
            nsteps = max(maximum(occ_host), Int(min_cycle_length))
            for _ in 1:nsteps
                do_exchange = !iszero(exchange_prob) && rand(rng_move, F) < F(exchange_prob)
                if do_exchange
                    PureAdsorb.mc_exchange!(rng_move, ws, db, dst, guest_c, guest_types, dp, dq, fug, kT; backend, groupsize, nblocks_per_chain)
                else
                    movetype = rand(rng_move, movechoices)
                    PureAdsorb.mc_step!(ws, db, dst, guest_c, guest_types, movetype, dstep_trans, dstep_rot, kT; backend, groupsize, nblocks_per_chain)
                end
            end
            isnothing(block_idx) && return nothing
            copyto!(occ_host, dst.occupancy)
            for n in 1:nsys
                Nn = Int(occ_host[n])
                Nn <= maxN || error("gcmc_widom_by_occupancy: occupancy $Nn exceeds maxN=$maxN")
                sN_visits[Nn + 1, block_idx] += 1
            end
            const_term_host = F[PureAdsorb.insertion_constant_term(ff, batch, guest, n, occ_host[n]) for n in 1:nsys]
            d_const_term = PureAdsorb.adapt(backend, const_term_host)
            for i in eachindex(sys_of)
                rpos[i] = rand(rng_widom, SVector{3, F})
                quat[i] = PureAdsorb.shoemake_quaternion(rng_widom, F)
            end
            copyto!(drpos, rpos)
            copyto!(dquat, quat)
            PureAdsorb.widom_chain_kernel!(backend)(
                dΔU, dsys, drpos, dquat, db, guest_c, guest_types, dst.refpoints, dst.orientations,
                dst.guest_offsets, dst.occupancy, dst.Sk, dst.k_offsets, d_const_term; ndrange = ninsert_per_cycle
            )
            KernelAbstractions.synchronize(backend)
            copyto!(ΔU_h, dΔU)
            for i in eachindex(sys_of)
                s = sys_of[i]
                Nn = Int(occ_host[s])
                w, _ = PureAdsorb.boltzmann_weight(ΔU_h[i], kT)
                sW[Nn + 1, block_idx] += w
                sWcount[Nn + 1, block_idx] += 1
            end
            return nothing
        end

        for _ in 1:n_warmup
            run_cycle!(nothing)
        end
        block_len = n_production ÷ nblocks
        for c in 1:n_production
            b = min(nblocks, (c - 1) ÷ block_len + 1)
            run_cycle!(b)
        end
        copyto!(state.occupancy, dst.occupancy)
        return (; sN_visits, sW, sWcount, nblocks, n_production, maxN)
    end

    # Reduces `gcmc_widom_by_occupancy`'s raw bins into one (LHS, RHS) pair per occupancy `N`
    # (R2: `P(N+1)/P(N) = (f V / ((N+1) kT)) * ⟨exp(-ΔU_ins/kT)⟩_N`), skipping any `N` whose visit
    # or Widom-sample count falls below `minvisits`/`minwidom` -- the sparse-loading fallback the
    # design anticipates, rather than reporting a statistic no sample size backs up. `f_target`
    # (eV·Å⁻³, already `PASCAL`-converted) and `V`/`kT` fix the RHS the LHS is checked against; a
    # caller demonstrating the perturbed-factor discriminator passes the chain's OWN `fugacity` for
    # the run but `f_target` from the TRUE fugacity, so any disagreement is the perturbation itself.
    function reduce_henry_r2(out, f_target::F, V::F, kT::F; minvisits::Integer = 30, minwidom::Integer = 30) where {F}
        rows = NamedTuple[]
        for Nn in 0:(out.maxN - 1)
            visits_N = Float64.(out.sN_visits[Nn + 1, :])
            visits_N1 = Float64.(out.sN_visits[Nn + 2, :])
            (sum(visits_N) < minvisits || sum(visits_N1) < minvisits) && continue
            local LHS, LHS_err
            try
                LHS, LHS_err = pooled_ratio_jackknife(visits_N1, visits_N)
            catch
                continue
            end
            sw = out.sW[Nn + 1, :]
            swc = Float64.(out.sWcount[Nn + 1, :])
            sum(swc) < minwidom && continue
            meanW, meanW_err = pooled_ratio_jackknife(sw, swc)
            pref = Float64(f_target) * Float64(V) / ((Nn + 1) * Float64(kT))
            RHS = pref * meanW
            RHS_err = pref * meanW_err
            push!(rows, (N = Nn, LHS = LHS, LHS_err = LHS_err, RHS = RHS, RHS_err = RHS_err, meanW = meanW, meanW_err = meanW_err))
        end
        return rows
    end
end

@testitem "pooled_ratio_jackknife reduces to the plain pooled ratio's own scale" setup = [HenryDetailedBalance] begin
    # Two blocks with identical per-block ratios: the pooled ratio must equal that common ratio
    # exactly, and the jackknife error (both leave-one-out estimates equal the pooled ratio too)
    # must be exactly zero.
    ratio, err = pooled_ratio_jackknife([4.0, 4.0], [2.0, 2.0])
    @test ratio == 2.0
    @test iszero(err)
    @test_throws DimensionMismatch pooled_ratio_jackknife([1.0], [1.0, 2.0])
    @test_throws ArgumentError pooled_ratio_jackknife([1.0], [0.0])
end

@testitem "500 Pa sits in the isotherm's Henry-linear regime" setup = [HenryDetailedBalance] tags = [:gpu] begin
    using KernelAbstractions
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    isnothing(backend) && error("no functional CUDA backend; this item must run on a GPU host")

    F = Float64
    b1, ff1, g1, sc, ewald = rubtak_co2_henry_setup(F, 1)
    K_H = widom(b1, g1; T = 298.15, ninsert = 500_000, seed = 11, nblocks = 10, backend)[1].K_H
    ideal_slope = K_H * PureAdsorb.PASCAL   # guests/Pa, the N -> 0 extrapolation of R2 at every N

    # This is exactly what a pressure choice for the merged test needs: `loading/pressure` matching
    # `ideal_slope` (within combined error) at both 200 Pa and 500 Pa (the merged test's own
    # pressure, below) marks the Henry-linear regime R2 does not itself require but the N=0
    # cross-check against `widom`'s OWN Henry coefficient implicitly assumes (R2 reduces to Henry's
    # law only while N=0 dominates). `bench/results/README.md`'s own 50-point isotherm (the same
    # framework and guest, run to much higher pressure) already shows the eventual saturation this
    # short two-point check is not sized to resolve on its own: `loading/pressure` there is flat at
    # 0.0037-0.0040 guests/Pa over 100-202 Pa and has fallen to 0.0010-0.0016 guests/Pa by
    # 4.9e4-1e5 Pa, several orders of magnitude in pressure above 500 Pa.
    iso = run_isotherm!(
        sc, ff1, g1, ewald; T = 298.15, pressures = [200.0, 500.0], nreplicas = 32, capacity = 30,
        n_warmup = 200, n_production = 1500, n_audit = 1_000_000, step_trans = 0.3, step_rot = 0.3,
        exchange_prob = 0.5, seed = 321, nblocks = 8, backend, nblocks_per_chain = 1
    )
    for p in eachindex(iso.pressure)
        slope = iso.loading[p] / iso.pressure[p]
        slope_err = iso.loading_err[p] / iso.pressure[p]
        @test isapprox(slope, ideal_slope; atol = 5 * slope_err)
    end
end

@testitem "R2 merged Henry/detailed-balance test agrees across loadings and with Milestone A's Henry coefficient at N=0" setup = [
    HenryDetailedBalance,
] tags = [:gpu] begin
    using KernelAbstractions
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    isnothing(backend) && error("no functional CUDA backend; this item must run on a GPU host")

    F = Float64
    nsys = 64
    b, ff, g, sc, ewald = rubtak_co2_henry_setup(F, nsys)
    T = 298.15
    kT = PureAdsorb.KB * T
    V = b.volumes[1]

    # 500 Pa: the isotherm test above pins it inside the Henry-linear regime, and it also gives
    # phi ~ 0.99997 (`peng_robinson_fugacity`), so this run does not also exercise the equation of
    # state (R3's own test does that, at 5e6 Pa). The ideal extrapolation `f*K_H` is ~2 guests,
    # chosen to spread occupancy visits usefully across N=0..~8 rather than concentrate them at
    # N=0 or N=1 alone.
    P = 500.0
    res = peng_robinson_fugacity(P, T, g)
    b1, ff1, g1, _, _ = rubtak_co2_henry_setup(F, 1)
    K_H_widom = widom(b1, g1; T, ninsert = 2_000_000, seed = 42, nblocks = 10, backend)[1]

    st = SystemState(b, g, zeros(Int, nsys), ff; T, seed = 777, capacities = fill(30, nsys))
    out = gcmc_widom_by_occupancy(
        b, st, g, ff; T, n_warmup = 200, n_production = 3000, n_widom_per_cycle = 4,
        step_trans = fill(0.3, nsys), step_rot = fill(0.3, nsys), fugacity = fill(res.f, nsys),
        exchange_prob = 0.5, seed = 2, backend, nblocks_per_chain = 1, maxN = 30, nblocks = 10
    )

    f_target = F(res.f) * PureAdsorb.PASCAL
    rows = reduce_henry_r2(out, f_target, V, F(kT))
    length(rows) >= 5 || error("fewer than 5 usable loadings ($(length(rows))); the histogram is too sparse for this pressure/run length")
    for row in rows
        combined_err = sqrt(row.LHS_err^2 + row.RHS_err^2)
        z = abs(row.LHS - row.RHS) / combined_err
        @test z < 4.0
    end

    # N=0's own Widom average, expressed as a Henry coefficient (`K_H = V*<W>/kT`), against
    # Milestone A's independent route (`widom`, a dedicated hard-core-rejection insertion sweep with
    # its own RNG stream, run on the pristine framework rather than sampled along a live GCMC
    # chain).
    row0 = only(filter(r -> iszero(r.N), rows))
    K_H_chain = V * row0.meanW / kT
    K_H_chain_err = V * row0.meanW_err / kT
    combined_K_H_err = sqrt(K_H_chain_err^2 + K_H_widom.K_H_err^2)
    @test isapprox(K_H_chain, K_H_widom.K_H; atol = 5 * combined_K_H_err)
end

@testitem "perturbing the insertion fugacity makes the R2 statistic fail loudly" setup = [HenryDetailedBalance] tags = [:gpu] begin
    using KernelAbstractions, Statistics
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    isnothing(backend) && error("no functional CUDA backend; this item must run on a GPU host")

    F = Float64
    nsys = 64
    b, ff, g, sc, ewald = rubtak_co2_henry_setup(F, nsys)
    T = 298.15
    kT = PureAdsorb.KB * T
    V = b.volumes[1]
    P = 500.0
    res = peng_robinson_fugacity(P, T, g)

    # The chain accepts insertions/deletions against a fugacity 1.5x the true one -- the same
    # position `log_insertion_prefactor`/`log_deletion_prefactor` (E2's combinatorial factors) take
    # `f` in, so this is operationally the same failure a wrong combinatorial prefactor produces: a
    # stationary distribution whose adjacent-N ratios are the TRUE ones times exactly 1.5, at every
    # N. `reduce_henry_r2`'s RHS is still built from the TRUE fugacity, so the two sides disagree by
    # that same factor.
    c_perturb = 1.5
    st = SystemState(b, g, zeros(Int, nsys), ff; T, seed = 999, capacities = fill(30, nsys))
    out = gcmc_widom_by_occupancy(
        b, st, g, ff; T, n_warmup = 200, n_production = 1500, n_widom_per_cycle = 4,
        step_trans = fill(0.3, nsys), step_rot = fill(0.3, nsys), fugacity = fill(c_perturb * res.f, nsys),
        exchange_prob = 0.5, seed = 4, backend, nblocks_per_chain = 1, maxN = 30, nblocks = 10
    )

    f_target = F(res.f) * PureAdsorb.PASCAL   # the TRUE fugacity, not the perturbed one the chain ran with
    # A higher sample floor than `reduce_henry_r2`'s own default: the highest few loadings this run
    # reaches are visited rarely enough that a single jackknife block's own noise can mask even a
    # 50% factor, exactly the sparse-loading fallback the design anticipates -- the well-sampled
    # loadings below are where this run's own discriminating power actually lives.
    rows = reduce_henry_r2(out, f_target, V, F(kT); minvisits = 4000, minwidom = 15000)
    length(rows) >= 3 || error("fewer than 3 well-sampled loadings ($(length(rows))); widen the run")
    zscores = Float64[]
    ratios = Float64[]
    for row in rows
        combined_err = sqrt(row.LHS_err^2 + row.RHS_err^2)
        push!(zscores, abs(row.LHS - row.RHS) / combined_err)
        push!(ratios, row.LHS / row.RHS)
    end
    # Every well-sampled loading fails by several sigma, and the failure is the injected factor
    # itself, not noise: the passing test's own worst case (z < 4) never approaches this.
    @test all(>(3.0), zscores)
    @test isapprox(median(ratios), c_perturb; atol = 0.15)
end
