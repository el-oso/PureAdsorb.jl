# The GCMC cycle driver: warmup cycles (discarded), then production cycles, each performing its
# `mc_step!`/`mc_exchange!` move attempts and, during production, sampling that system's loading
# and energy once per cycle for block averaging (`docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`,
# "Output"). Mirrors `run_nvt!`'s structure (`src/nvt.jl`) with two differences the μVT ensemble
# forces: a cycle attempts either an NVT move or an exchange move, chosen by `exchange_prob`, and
# the cycle length is recomputed every cycle from the LIVE occupancy rather than fixed once before
# the loop, since N drifts here and never does in NVT.

"""
    GCMCResult{F}

Per-system result of a GCMC run.

- `loading`/`loading_err` (guests): production-average occupancy and its cycle-block standard
  error (`block_mean_sem`).
- `energy`/`energy_err` (eV): production-average total configuration energy and its cycle-block
  standard error, exactly as `NVTResult`'s own fields.
- `q_st`/`q_st_err` (eV), `corr_UN`: the fluctuation isosteric heat of adsorption and its
  jackknife-over-blocks error, and the pooled `corr(U, N)` that governs how slowly `q_st`
  converges — see `fluctuation_qst`'s own docstring for the formula, the sign convention, and why
  the error is a jackknife rather than a per-block ratio. Always `Float64`, following
  `WidomResult`'s own convention for the same reason: `q_st` is a ratio of small differences of
  large sums, and a smaller float type's precision loss there is exactly what the jackknife's own
  arithmetic must not compound.
- `max_occupancy`/`capacity`: the highest occupancy this system reached over the whole run
  (warmup and production) against its reserved capacity — a system that spent the run hard
  against capacity was sampling a truncated distribution even though `mc_insert!`'s own
  `capacity_hits` check found no single move it would otherwise have accepted (that check is
  fail-fast on its own trigger condition, not a substitute for this diagnostic).
- `ncycles`: number of production cycles.
"""
struct GCMCResult{F}
    loading::F
    loading_err::F
    energy::F
    energy_err::F
    q_st::Float64
    q_st_err::Float64
    corr_UN::Float64
    max_occupancy::Int
    capacity::Int
    ncycles::Int
end

"""
    fluctuation_qst(energy, loading, kT::F, nblocks::Integer) -> (q_st, q_st_err, corr_UN)

The isosteric heat of adsorption from fluctuations over one system's `energy`/`loading` samples,
one pair per production cycle (`docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`, "Output"):

    q_st = kT - (⟨UN⟩ - ⟨U⟩⟨N⟩) / (⟨N²⟩ - ⟨N⟩²)

`kT` MINUS the fluctuation ratio is the WIDOM convention this package's own `q_st`
(`nvt.jl`'s `WidomResult`/`_reduce`) already uses. kUPS's GCMC analyzer computes the OPPOSITE sign
(`cov/var - kT`) to its own Widom analyzer (`application/mcmc/analysis.py:122-126` vs. `:326-332`),
so a comparison against kUPS's GCMC numbers must negate theirs, not ours.

Both moments are POOLED over every cycle, not averaged from a per-block ratio — the latter carries
the Jensen-inequality bias kUPS's own Widom docstring warns against, since `q_st` is a nonlinear
function of the sums. The error is a delete-one-BLOCK jackknife over `nblocks` contiguous blocks
of cycles (successive cycles are correlated, the same reason `block_mean_sem` blocks by cycle
rather than treating each as independent): `q_st^(-b)` recomputes the pooled ratio with block
`b`'s cycles excluded, for every `b`, and the jackknife variance is `(nblocks-1)/nblocks *
Σ_b (q_st^(-b) - mean(q_st^(-b)))²`.

`corr_UN` is the pooled Pearson correlation `corr(U, N)` over the same full sample: `q_st`'s
relative error scales as `sqrt((1 - corr_UN²) / (n corr_UN²))`, so a low correlation means many
more cycles are needed to pin `q_st` down than the loading alone requires — reported here rather
than left for a large error bar to explain silently.
"""
function fluctuation_qst(energy::AbstractVector{F}, loading::AbstractVector{F}, kT::F, nblocks::Integer) where {F}
    ns = length(energy)
    length(loading) == ns || throw(DimensionMismatch("energy and loading must have the same length"))
    nblocks >= 2 || throw(ArgumentError("nblocks=$nblocks: at least two blocks are needed for a jackknife error"))
    ns >= 2 * nblocks || throw(ArgumentError("$ns cycles: need at least 2*nblocks=$(2 * nblocks) production cycles"))
    block_len = ns ÷ nblocks

    sU = zeros(Float64, nblocks); sN = zeros(Float64, nblocks)
    sU2 = zeros(Float64, nblocks); sN2 = zeros(Float64, nblocks); sUN = zeros(Float64, nblocks)
    counts = zeros(Int, nblocks)
    for b in 1:nblocks
        lo = (b - 1) * block_len + 1
        hi = b == nblocks ? ns : b * block_len
        for i in lo:hi
            u = Float64(energy[i]); n = Float64(loading[i])
            sU[b] += u; sN[b] += n; sU2[b] += u^2; sN2[b] += n^2; sUN[b] += u * n
        end
        counts[b] = hi - lo + 1
    end
    totU = sum(sU); totN = sum(sN); totU2 = sum(sU2); totN2 = sum(sN2); totUN = sum(sUN)

    qst_of(sumU, sumN, sumUN, sumN2, n) = begin
        mU = sumU / n; mN = sumN / n
        cov = sumUN / n - mU * mN
        varN = sumN2 / n - mN^2
        kT - cov / varN
    end

    q_pooled = qst_of(totU, totN, totUN, totN2, ns)
    q_loo = Vector{Float64}(undef, nblocks)
    for b in 1:nblocks
        q_loo[b] = qst_of(totU - sU[b], totN - sN[b], totUN - sUN[b], totN2 - sN2[b], ns - counts[b])
    end
    q_bar = sum(q_loo) / nblocks
    q_err = sqrt((nblocks - 1) / nblocks * sum(abs2, q_loo .- q_bar))

    mU = totU / ns; mN = totN / ns
    varU = totU2 / ns - mU^2
    varN = totN2 / ns - mN^2
    corr_UN = (totUN / ns - mU * mN) / sqrt(varU * varN)

    return Float64(q_pooled), q_err, corr_UN
end

"""
    run_gcmc!(batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F,N}, ff::ForceField{F};
              T, n_warmup, n_production, n_audit, step_trans, step_rot, fugacity,
              exchange_prob = 0.5, min_cycle_length = 1, seed = 0, nblocks = 10, backend = CPU(),
              groupsize = DEFAULT_GROUPSIZE,
              nblocks_per_chain = default_nblocks_per_chain(F, state.nsys)) -> Vector{GCMCResult{F}}

Run `state`'s chains at temperature `T` (K) and per-system fugacity `fugacity[n]` (Pa, the units
`peng_robinson_fugacity` returns): `n_warmup` cycles are discarded, then `n_production` cycles are
recorded. Mirrors `run_nvt!`'s cycle structure with two differences:

Each of a cycle's move attempts is an exchange attempt with probability `exchange_prob` and an NVT
move (translation/rotation/reinsertion, drawn uniformly as in `run_nvt!`) otherwise — the SAME
`Xoshiro(seed)` stream `run_nvt!` uses for its movetype draw, so `exchange_prob = 0` draws NOTHING
before it (`&&`'s short-circuit skips `rand(rng_move, F)` entirely rather than drawing and
discarding it) and reproduces `run_nvt!`'s own per-step draw sequence exactly, bit for bit. An
exchange attempt goes through `mc_exchange!` alone, drawing its own fair insert/delete coin from
the same stream — never `mc_insert!`/`mc_delete!` directly, since only the fair mixture is
detailed-balanced (`src/moves.jl`'s own comment on the μVT exchange moves).

A cycle's length is `max(maximum(nguests(state, n) for n in 1:nsys), min_cycle_length)`,
recomputed from `state`'s LIVE occupancy at the start of EVERY cycle (kUPS's `LoopPropagator`
does the same) rather than fixed once before the loop as `run_nvt!` does: `run_nvt!`'s occupancy
never changes, so fixing it once there is exact, but N drifts under exchange, and fixing the
length to its initial value here would bias sampling density as the chain equilibrates to a
different loading. When `exchange_prob = 0`, occupancy never changes either, so this recomputation
returns the same value every cycle and the two drivers' move sequences coincide exactly.

`const_p`/`const_q` (the μVT exchange moves' pose-independent affine coefficients,
`exchange_constant_coeffs(ff, batch, guest_c)`) are computed ONCE here, from the host-resident
`batch` argument, and adapted to `backend` once — see `mc_insert!`'s own docstring for why
re-deriving them per exchange attempt would force a host round-trip of a device-resident `batch`
on every call.

`fugacity`, `step_trans`, `step_rot` are per-chain and used exactly as given for the whole run
(R4, `run_nvt!`'s own convention). The energy audit (`audit_energy!`) runs every `n_audit` cycles,
fail-fast on a mismatch, exactly as `run_nvt!`'s; `state.occupancy` is synced alongside the fields
`run_nvt!`'s own audit already syncs, since GCMC (unlike NVT) can change it. `mc_insert!` throws
immediately, independent of this driver, if any system hits capacity on a move the Metropolis test
would otherwise have accepted (`capacity_hits`) — the loud failure the design's capacity
diagnostic requires; `GCMCResult.max_occupancy` reports the softer, non-fatal signal of how close
every system came to that.

Returns one `GCMCResult` per system.
"""
function run_gcmc!(
        batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F, N}, ff::ForceField{F};
        T, n_warmup::Integer, n_production::Integer, n_audit::Integer, step_trans, step_rot, fugacity,
        exchange_prob::Real = 0.5, min_cycle_length::Integer = 1, seed::Integer = 0, nblocks::Integer = 10,
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = default_nblocks_per_chain(F, state.nsys)
    ) where {F, N}
    (
        guest.types == batch.guest_types_orig && guest.sites == batch.guest_sites_orig &&
            guest.charges == batch.guest_charges_orig
    ) || throw(
        ArgumentError(
            "guest passed to run_gcmc! does not match the guest FrameworkBatch/SystemState were built with"
        )
    )
    n_warmup >= 0 || throw(ArgumentError("n_warmup=$n_warmup must be >= 0"))
    n_production >= 1 || throw(ArgumentError("n_production=$n_production must be >= 1"))
    n_audit >= 1 || throw(ArgumentError("n_audit=$n_audit must be >= 1"))
    0 <= exchange_prob <= 1 || throw(ArgumentError("exchange_prob=$exchange_prob must be in [0, 1]"))
    nsys = state.nsys
    length(fugacity) == nsys || throw(DimensionMismatch("fugacity must have one entry per system (nsys=$nsys), got $(length(fugacity))"))
    kT = F(KB * T)
    fug = F.(fugacity)
    # See `run_nvt!`'s own comment: `select_and_propose`'s clamp needs at least one reserved guest
    # slot SOMEWHERE in the batch; with none at all there is no valid slot to clamp into, so move
    # attempts (of either kind) are skipped entirely rather than launching on an empty index space.
    any_guests = !isempty(state.refpoints)

    guest_c = compact_guest(batch, guest)
    guest_types = SVector{N, Int}(batch.guest_types)
    p_host, q_host = exchange_constant_coeffs(ff, batch, guest_c)

    db = adapt(backend, batch)
    dst = adapt(backend, state)
    dstep_trans = adapt(backend, F.(step_trans))
    dstep_rot = adapt(backend, F.(step_rot))
    dp = adapt(backend, p_host); dq = adapt(backend, q_host)
    ws = MoveWorkspace(F, nsys, nblocks_per_chain; backend)

    rng_move = Xoshiro(seed)
    movechoices = (MOVE_TRANSLATION, MOVE_ROTATION, MOVE_REINSERTION)

    occ_host = Vector{Int32}(undef, nsys)
    occ_prev = Vector{Int32}(undef, nsys)
    copyto!(occ_prev, dst.occupancy)
    energy_host = Vector{F}(undef, nsys)
    energy_samples = Matrix{F}(undef, nsys, n_production)
    loading_samples = Matrix{F}(undef, nsys, n_production)
    max_occupancy = zeros(Int, nsys)

    # `state.accepted`/`attempted` (`NMOVETYPES = 3`) count only translation/rotation/reinsertion,
    # exactly as `run_nvt!`'s do; an exchange attempt changes OCCUPANCY, not a movetype counter, so
    # its own acceptance is read off an occupancy CHANGE (any insertion or deletion accepted moves
    # exactly one guest) rather than a fourth counter slot, which would change
    # `SystemState.accepted`/`attempted`'s element type for every caller, `run_nvt!` included.
    # `nvt_accepted_offset` carries the NVT-accepted total across the production reset below (which
    # matches `run_nvt!`'s own convention of reporting production-only acceptance), since the
    # audit's own bookkeeping needs the TRUE cumulative count: `sk_abs_accum`/`energy_abs_accum`
    # keep accumulating straight through that reset (nothing zeroes them except a successful
    # `audit_energy!` call), so undoing the reset just for `nmoves` would desynchronize the two and
    # make the tolerance too tight for the first post-reset audit.
    exchange_accepted = zeros(Int, nsys)
    nvt_accepted_offset = zeros(Int, nsys)
    last_accepted = zeros(Int, nsys)
    last_exchange_accepted = zeros(Int, nsys)
    cycles_since_audit = Ref(0)

    # See `run_nvt!`'s own `run_audit!`: the sync dance `audit_energy!` needs since it recomputes
    # with plain host indexing. `occupancy` is synced here in ADDITION to `run_nvt!`'s own set of
    # fields, since GCMC (unlike NVT) can change it; `audit_energy!` only reads it (the
    # occupancy-vs-capacity check), never corrects it, so no push-back is needed for it as there is
    # for `Sk`/`energy`. `nmoves` (the audit tolerance's accumulated-move count) adds the exchange
    # acceptances tracked above to `state.accepted`'s own NVT-move total: `mc_insert_kernel!`/
    # `mc_delete_kernel!` add to `energy_abs_accum`/`sk_abs_accum` on every accepted exchange move
    # exactly as `decide_move_kernel!` does for NVT moves, so omitting them here would silently
    # undercount `nmoves` and make the audit tolerance too tight whenever exchange ran.
    function run_audit!()
        copyto!(state.refpoints, dst.refpoints)
        copyto!(state.orientations, dst.orientations)
        copyto!(state.occupancy, dst.occupancy)
        copyto!(state.Sk, dst.Sk)
        copyto!(state.energy, dst.energy)
        copyto!(state.host_energy, dst.host_energy)
        copyto!(state.accepted, dst.accepted)
        for n in 1:nsys
            total = nvt_accepted_offset[n] + Int(sum(state.accepted[n])) + exchange_accepted[n]
            prev = last_accepted[n] + last_exchange_accepted[n]
            nmoves = max(total - prev, 1)
            audit_energy!(batch, state, guest, ff, n, nmoves)
            last_accepted[n] = nvt_accepted_offset[n] + Int(sum(state.accepted[n]))
            last_exchange_accepted[n] = exchange_accepted[n]
        end
        copyto!(dst.Sk, state.Sk)
        copyto!(dst.energy, state.energy)
        cycles_since_audit[] = 0
        return nothing
    end

    function run_cycle!()
        copyto!(occ_host, dst.occupancy)
        copyto!(occ_prev, occ_host)
        for n in eachindex(occ_host)
            max_occupancy[n] = max(max_occupancy[n], Int(occ_host[n]))
        end
        nsteps = max(maximum(occ_host), Int(min_cycle_length))
        if any_guests
            for _ in 1:nsteps
                do_exchange = !iszero(exchange_prob) && rand(rng_move, F) < F(exchange_prob)
                if do_exchange
                    mc_exchange!(rng_move, ws, db, dst, guest_c, guest_types, dp, dq, fug, kT; backend, groupsize, nblocks_per_chain)
                    # `occ_host` is reused as scratch here: its cycle-start value already fed
                    # `nsteps` above and is not read again until the next cycle (or the caller's
                    # own post-cycle sample, both of which re-copy it fresh first).
                    copyto!(occ_host, dst.occupancy)
                    for n in eachindex(occ_host)
                        occ_host[n] == occ_prev[n] || (exchange_accepted[n] += 1)
                    end
                    copyto!(occ_prev, occ_host)
                else
                    movetype = rand(rng_move, movechoices)
                    mc_step!(ws, db, dst, guest_c, guest_types, movetype, dstep_trans, dstep_rot, kT; backend, groupsize, nblocks_per_chain)
                end
            end
        end
        cycles_since_audit[] += 1
        cycles_since_audit[] >= n_audit && run_audit!()
        return nothing
    end

    for _ in 1:n_warmup
        run_cycle!()
    end

    # Production accumulates its own NVT acceptance rate from a clean baseline, exactly as
    # `run_nvt!`'s does; `nvt_accepted_offset` preserves the pre-reset cumulative total for the
    # audit's own bookkeeping (this function's own comment on `run_audit!`). `exchange_accepted`
    # is NOT reset here: nothing reports a production-only exchange rate, and its own running
    # total must stay in step with `sk_abs_accum`/`energy_abs_accum`, which the reset below does
    # not touch either.
    copyto!(state.accepted, dst.accepted)
    for n in 1:nsys
        nvt_accepted_offset[n] += Int(sum(state.accepted[n]))
    end
    fill!(dst.accepted, zero(SVector{NMOVETYPES, Int32}))
    fill!(dst.attempted, zero(SVector{NMOVETYPES, Int32}))

    for c in 1:n_production
        run_cycle!()
        copyto!(energy_host, dst.energy)
        copyto!(occ_host, dst.occupancy)
        for n in 1:nsys
            energy_samples[n, c] = energy_host[n]
            loading_samples[n, c] = F(occ_host[n])
            max_occupancy[n] = max(max_occupancy[n], Int(occ_host[n]))
        end
    end

    copyto!(state.accepted, dst.accepted)
    copyto!(state.attempted, dst.attempted)
    copyto!(state.occupancy, dst.occupancy)

    results = Vector{GCMCResult{F}}(undef, nsys)
    for n in 1:nsys
        ℓ, ℓ_err = block_mean_sem(view(loading_samples, n, :), nblocks)
        ē, ē_err = block_mean_sem(view(energy_samples, n, :), nblocks)
        q, q_err, corr_UN = fluctuation_qst(view(energy_samples, n, :), view(loading_samples, n, :), kT, nblocks)
        results[n] = GCMCResult{F}(ℓ, ℓ_err, ē, ē_err, q, q_err, corr_UN, max_occupancy[n], Int(capacity(state, n)), n_production)
    end
    return results
end
