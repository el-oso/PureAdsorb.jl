# The NVT cycle driver: warmup cycles (discarded), then production cycles, each performing its
# `mc_step!` move attempts followed by Widom insertions into the configuration as it then stands.
# `WidomResult`'s `_reduce` is reused verbatim for the μ_ex/K_H/q_st reduction; only the block
# boundaries differ (cycles here, insertions there), since successive cycles are correlated and
# insertion-level blocks would understate the error
# (docs/superpowers/specs/2026-09-26-milestone-b-nvt.md, "Cycles, and where Widom fits").

"""
    NVTResult{T}

Per-system result of an NVT Monte Carlo run with Widom insertion along the chain, following
`WidomResult`'s field conventions:

- `energy`/`energy_err` (eV): production-average total configuration energy and its
  cycle-block standard error.
- `mu_ex`, `mu_ex_err`, `K_H`, `K_H_err`, `q_st`, `q_st_err`: as `WidomResult`, but from
  insertions into the chain's occupied configuration (host and every guest), block-averaged over
  cycles rather than insertions.
- `acceptance`: per-move-type (translation, rotation, reinsertion) acceptance rate over the
  production phase only.
- `ncycles`: number of production cycles.
"""
struct NVTResult{T}
    energy::T
    energy_err::T
    mu_ex::T
    mu_ex_err::T
    K_H::T
    K_H_err::T
    q_st::T
    q_st_err::T
    acceptance::SVector{NMOVETYPES, T}
    ncycles::Int
end

# The pose-independent part of a test-particle insertion's ΔU for system `n`, currently holding
# `Ng` guests: the tail-correction and net-charge-correction changes from adding one more guest,
# plus the inserted guest's own self-energy, intramolecular exclusion and orientation-averaged
# reciprocal self term (R1/R2, `guest_self_terms`), which do not depend on `Ng` at all. At `Ng=0`
# this is EXACTLY `batch.constant_offset[n]`, Milestone A's own precomputed value for inserting
# into an empty system, reused verbatim rather than re-derived — this is what makes a Widom
# insertion into an empty chain reproduce Milestone A's `widom` bit for bit.
function insertion_constant_term(ff::ForceField{T}, batch::FrameworkBatch{T}, guest::Guest{T, N}, n::Integer, Ng::Integer) where {T, N}
    fw = batch.framework_of[n]
    iszero(Ng) && return batch.constant_offset[fw]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    V = batch.volumes[fw]; alpha = batch.alphas[fw]
    ntypes_ff = length(ff.names)
    gcounts = zeros(Int, ntypes_ff)
    for t in batch.guest_types_orig
        gcounts[t] += 1
    end
    host_counts = zeros(Int, ntypes_ff)
    for j in (a0 + 1):(a0 + natoms)
        host_counts[batch.compact_to_orig[batch.types[j]]] += 1
    end
    Qh = sum(view(batch.charges, (a0 + 1):(a0 + natoms)))
    Qg = sum(guest.charges)
    net_at(m) = -T(KE) * T(π) / (2 * V * alpha^2) * ((Qh + m * Qg)^2 - Qh^2)
    # `batch.constant_offset[fw]` with its own Ng=0->1 tail/net terms removed, leaving only the
    # Ng-independent self/exclusion/orientation-average piece.
    base = batch.constant_offset[fw] - guest_tail_correction(ff, host_counts, gcounts, 1, V) - net_at(1)
    return base + guest_tail_correction(ff, host_counts, gcounts, Ng + 1, V) - guest_tail_correction(ff, host_counts, gcounts, Ng, V) +
        net_at(Ng + 1) - net_at(Ng)
end

"""
    widom_chain_kernel!(ΔU, sys_of, rpos, quat, batch, guest, guest_types, refpoints, orientations,
                        guest_offsets, Sk, sys_k_offsets, const_term)

Test-particle insertion energy at pose `(batch.cells[fw]*rpos[i], quat[i])`, `s = sys_of[i]`,
`fw = batch.framework_of[s]`, into system `s` as it currently stands: host-guest real space plus
the reciprocal cross term against the RUNNING total structure factor `Sk` (`insertion_energy`,
generalized from Milestone A's `Shost`-only background to the full host-plus-guests field — at
`Sk == Shost` this reduces to Milestone A's own formula exactly), plus guest-guest real space
against every existing guest in `guest_offsets[s]+1:guest_offsets[s+1]`
(`guest_pair_realspace_energy`, an empty sum when the system holds no guests), plus the
pose-independent `const_term[s]` (`insertion_constant_term`). `sys_k_offsets` (the caller's own
`state.k_offsets`) locates system `s`'s slice of `Sk`, which is sized one slice per SYSTEM even
when systems share a framework and so cannot reuse `batch.k_offsets` (one slice per FRAMEWORK,
`FrameworkBatch`'s docstring) directly. Reads `refpoints`/`orientations`/`Sk` only; never mutates
state.
"""
@kernel function widom_chain_kernel!(
        ΔU, @Const(sys_of), @Const(rpos), @Const(quat), batch, guest::Guest{T, N}, guest_types::SVector{N, Int},
        @Const(refpoints), @Const(orientations), @Const(guest_offsets), @Const(Sk), @Const(sys_k_offsets),
        @Const(const_term)
    ) where {T, N}
    i = @index(Global)
    s = sys_of[i]
    fw = batch.framework_of[s]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    # `batch.k_offsets` ranges over `batch.ks`/`batch.kprefactor`, one slice per FRAMEWORK
    # (`FrameworkBatch`'s docstring); `sys_k_offsets` (this call's own `state.k_offsets`) ranges
    # over `Sk`, one slice per SYSTEM — the two have the same length for system `s`'s framework
    # but different absolute starts, so each gets its own range here.
    k0 = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); k1 = batch.k_offsets[fw + 1]
    ks0 = sys_k_offsets[s] + one(eltype(sys_k_offsets)); ks1 = sys_k_offsets[s + 1]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    q = quat[i]
    pos = A * rpos[i]
    e = insertion_energy(
        pos, q, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha,
        view(batch.ks, k0:k1), view(batch.kprefactor, k0:k1), view(Sk, ks0:ks1)
    )
    test_sites = guest_sites_at(guest, pos, q)
    gr = (guest_offsets[s] + one(eltype(guest_offsets))):guest_offsets[s + 1]
    E_lj = zero(T); E_sr = zero(T)
    for j in gr
        other_sites = guest_sites_at(guest, refpoints[j], orientations[j])
        lj, sr = guest_pair_realspace_energy(
            test_sites, other_sites, guest_types, guest.charges, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
        )
        E_lj += lj; E_sr += sr
    end
    ΔU[i] = e + E_lj + T(KE) * E_sr + const_term[s]
end

# Block-averaged mean and standard error of one scalar sampled once per production cycle
# (`samples`), grouped into `nblocks` contiguous blocks of cycles: successive cycles are
# correlated, so this is a materially different (and more honest) estimate than treating every
# cycle as an independent sample. A leftover remainder after `length(samples) ÷ nblocks` fills
# whole blocks is folded into the last block, matching `_widom`'s own block-length convention.
function block_mean_sem(samples::AbstractVector{F}, nblocks::Integer) where {F}
    ns = length(samples)
    nblocks >= 2 || throw(ArgumentError("nblocks=$nblocks: at least two blocks are needed for a standard error"))
    ns >= 2 * nblocks || throw(ArgumentError("$ns cycles: need at least 2*nblocks=$(2 * nblocks) production cycles"))
    block_len = ns ÷ nblocks
    means = zeros(Float64, nblocks)
    for b in 1:nblocks
        lo = (b - 1) * block_len + 1
        hi = b == nblocks ? ns : b * block_len
        means[b] = sum(Float64.(view(samples, lo:hi))) / (hi - lo + 1)
    end
    m = sum(means) / nblocks
    v = sum(abs2, means .- m) / (nblocks - 1)
    return F(m), F(sqrt(v / nblocks))
end

"""
    run_nvt!(batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F,N}, ff::ForceField{F};
             T, n_warmup, n_production, n_widom_per_cycle, n_audit, step_trans, step_rot,
             min_cycle_length = 1, seed = 0, widom_seed = seed, nblocks = 10, backend = CPU(),
             groupsize = DEFAULT_GROUPSIZE,
             nblocks_per_chain = default_nblocks_per_chain(F, state.nsys)) -> Vector{NVTResult{F}}

Run `state`'s chains at temperature `T` (K): `n_warmup` cycles are discarded, then `n_production`
cycles are recorded. A cycle is `max(maximum(nguests(state, n) for n in 1:nsys), min_cycle_length)`
Metropolis move attempts (one shared movetype per attempt, translation/rotation/reinsertion drawn
uniformly since exchange has no counterpart in this package — task 1's normalization with
`exchange_prob=0`), followed by `n_widom_per_cycle` test-particle insertions per system into the
configuration as it then stands (`widom_chain_kernel!`), which do not mutate `state`.

`step_trans` (Å) and `step_rot` (in `[0,1]`) are per-chain vectors, used exactly as given for
BOTH warmup and production — frozen for the whole run, which trivially satisfies "frozen during
production" (R4) without adaptation logic this milestone does not require. `mc_step!` freezes them
identically; this function adds no adaptation of its own.

The energy audit (`audit_energy!`) runs every `n_audit` cycles (warmup and production share one
running cycle count), fail-fast on a mismatch. Since it needs host-resident arrays, `state` (the
argument, not `batch`'s device-adapted copy this function builds internally) is kept as the audit's
working copy and is only as stale as the last audit or the run's end; every other query reads the
device-resident copy that actually ran the chain.

Move-type selection is drawn from `Xoshiro(seed)` and Widom insertion poses from a SEPARATE
`Xoshiro(widom_seed)`, both entirely apart from `state`'s own per-chain `ChainRNG` streams
(Milestone A/B's device-side RNG, `src/rng.jl`) that `mc_step!` consumes. Widom poses are assigned
to systems in round-robin order (`sys_of[i] = mod1(i, nsys)`, `widom.jl`'s `run=1` case) so that,
when every system holds zero guests, the resulting insertion stream is exactly the one
`widom_singlephase(batch, guest; T, ninsert = state.nsys*n_widom_per_cycle*n_production, seed =
widom_seed, run=1, ...)` would draw — this is what makes `N=0` reproduce Milestone A's
μ_ex/K_H/q_st (to floating-point rounding; the two estimators' different cycle-versus-insertion
block boundaries mean their SEMs are not held to the same standard, matching how this project
already treats independently-grouped floating-point sums elsewhere, e.g. `moves_tests.jl`'s
GPU-vs-CPU comparison).

`widom_seed` defaults to `seed`, which is exact and collision-free whenever every system holds
zero guests (there is nothing for `state`'s own placement draw to have consumed). When any system
holds guests, `state` was built by `SystemState(...; seed = X)` for some `X` the caller chose
separately, and `Xoshiro(widom_seed)` reproduces the SAME draw sequence as `SystemState`'s own
placement `Xoshiro(X)` whenever `widom_seed == X`: both call `rand(rng, SVector{3,F})` then
`shoemake_quaternion(rng, F)` in the same pattern, so identical seeds give identical draws. A
widom insertion that lands exactly on a guest's placement pose (before that guest has moved) then
has a zero real-space distance, diverging to an infinite or `NaN` energy. Pass a `widom_seed`
distinct from whatever seed `state` was built with whenever `Ng > 0` anywhere in the batch.

Returns one `NVTResult` per system.
"""
function run_nvt!(
        batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F, N}, ff::ForceField{F};
        T, n_warmup::Integer, n_production::Integer, n_widom_per_cycle::Integer, n_audit::Integer,
        step_trans, step_rot, min_cycle_length::Integer = 1, seed::Integer = 0, widom_seed::Integer = seed,
        nblocks::Integer = 10, backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE,
        nblocks_per_chain::Integer = default_nblocks_per_chain(F, state.nsys)
    ) where {F, N}
    (
        guest.types == batch.guest_types_orig && guest.sites == batch.guest_sites_orig &&
            guest.charges == batch.guest_charges_orig
    ) || throw(
        ArgumentError(
            "guest passed to run_nvt! does not match the guest FrameworkBatch/SystemState were built with"
        )
    )
    n_warmup >= 0 || throw(ArgumentError("n_warmup=$n_warmup must be >= 0"))
    n_production >= 1 || throw(ArgumentError("n_production=$n_production must be >= 1"))
    n_widom_per_cycle >= 1 || throw(ArgumentError("n_widom_per_cycle=$n_widom_per_cycle must be >= 1"))
    n_audit >= 1 || throw(ArgumentError("n_audit=$n_audit must be >= 1"))
    nsys = state.nsys
    kT = F(KB * T)
    nsteps_per_cycle = max(maximum(nguests(state, n) for n in 1:nsys), Int(min_cycle_length))
    # `select_and_propose` (moves.jl) clamps a chain's guest index into `1:length(refpoints)`,
    # which assumes at least one guest exists SOMEWHERE in the batch; with none at all there is no
    # valid slot to clamp into, and no guest for a move to act on regardless, so move attempts are
    # skipped entirely rather than launching `mc_step!` on an empty index space.
    any_guests = !isempty(state.refpoints)

    guest_c = compact_guest(batch, guest)
    guest_types = SVector{N, Int}(batch.guest_types)
    const_term_host = F[insertion_constant_term(ff, batch, guest, n, nguests(state, n)) for n in 1:nsys]

    db = adapt(backend, batch)
    dst = adapt(backend, state)
    dstep_trans = adapt(backend, F.(step_trans))
    dstep_rot = adapt(backend, F.(step_rot))
    d_const_term = adapt(backend, const_term_host)
    ws = MoveWorkspace(F, nsys, nblocks_per_chain; backend)

    ninsert_per_cycle = nsys * n_widom_per_cycle
    sys_of = Vector{Int32}(undef, ninsert_per_cycle)
    rpos = Vector{SVector{3, F}}(undef, ninsert_per_cycle)
    quat = Vector{SVector{4, F}}(undef, ninsert_per_cycle)
    for iter in 0:(n_widom_per_cycle - 1), s in 1:nsys
        idx = iter * nsys + s
        sys_of[idx] = s
    end
    dsys = adapt(backend, sys_of)
    drpos = adapt(backend, rpos)
    dquat = adapt(backend, quat)
    dΔU = adapt(backend, zeros(F, ninsert_per_cycle))
    ΔU_h = Vector{F}(undef, ninsert_per_cycle)

    rng_move = Xoshiro(seed)
    rng_widom = Xoshiro(widom_seed)
    movechoices = (MOVE_TRANSLATION, MOVE_ROTATION, MOVE_REINSERTION)

    energy_host = Vector{F}(undef, nsys)
    energy_samples = Matrix{F}(undef, nsys, n_production)
    sW = zeros(Float64, nsys, n_production)
    sUW = zeros(Float64, nsys, n_production)
    ncount = zeros(Int, nsys, n_production)

    last_accepted = zeros(Int, nsys)
    cycles_since_audit = Ref(0)

    # Rebuilds `state` (the host argument) from `dst` (the device-resident chain), audits every
    # system, then pushes any correction `audit_energy!` made back to `dst` — the sync dance
    # `audit_energy!` needs since it recomputes energies/structure factors with plain host
    # indexing (`total_energy`, `structure_factor`), which is a no-op copy when `backend == CPU()`
    # and `dst` already aliases `state`'s own arrays.
    function run_audit!()
        copyto!(state.refpoints, dst.refpoints)
        copyto!(state.orientations, dst.orientations)
        copyto!(state.Sk, dst.Sk)
        copyto!(state.energy, dst.energy)
        copyto!(state.host_energy, dst.host_energy)
        copyto!(state.accepted, dst.accepted)
        for n in 1:nsys
            total_accepted = Int(sum(state.accepted[n]))
            nmoves = max(total_accepted - last_accepted[n], 1)
            audit_energy!(batch, state, guest, ff, n, nmoves)
            last_accepted[n] = total_accepted
        end
        copyto!(dst.Sk, state.Sk)
        copyto!(dst.energy, state.energy)
        cycles_since_audit[] = 0
        return nothing
    end

    function run_cycle!()
        if any_guests
            for _ in 1:nsteps_per_cycle
                movetype = rand(rng_move, movechoices)
                mc_step!(ws, db, dst, guest_c, guest_types, movetype, dstep_trans, dstep_rot, kT; backend, groupsize, nblocks_per_chain)
            end
        end
        cycles_since_audit[] += 1
        cycles_since_audit[] >= n_audit && run_audit!()
        return nothing
    end

    for _ in 1:n_warmup
        run_cycle!()
    end

    # Production accumulates its own acceptance rate and widom statistics from a clean baseline.
    fill!(dst.accepted, zero(SVector{NMOVETYPES, Int32}))
    fill!(dst.attempted, zero(SVector{NMOVETYPES, Int32}))
    fill!(last_accepted, 0)

    for c in 1:n_production
        run_cycle!()
        copyto!(energy_host, dst.energy)
        energy_samples[:, c] .= energy_host

        for i in eachindex(sys_of, rpos, quat)
            rpos[i] = rand(rng_widom, SVector{3, F})
            quat[i] = shoemake_quaternion(rng_widom, F)
        end
        copyto!(drpos, rpos)
        copyto!(dquat, quat)
        widom_chain_kernel!(backend)(
            dΔU, dsys, drpos, dquat, db, guest_c, guest_types, dst.refpoints, dst.orientations, dst.guest_offsets,
            dst.Sk, dst.k_offsets, d_const_term; ndrange = ninsert_per_cycle
        )
        KernelAbstractions.synchronize(backend)
        copyto!(ΔU_h, dΔU)
        for i in eachindex(sys_of)
            s = sys_of[i]
            w, uw = boltzmann_weight(ΔU_h[i], kT)
            sW[s, c] += w
            sUW[s, c] += uw
            ncount[s, c] += 1
        end
    end

    copyto!(state.accepted, dst.accepted)
    copyto!(state.attempted, dst.attempted)

    results = Vector{NVTResult{F}}(undef, nsys)
    for n in 1:nsys
        wr = _reduce(view(sW, n, :), view(sUW, n, :), view(ncount, n, :), kT, batch.volumes[batch.framework_of[n]], n, F)
        ē, ē_err = block_mean_sem(view(energy_samples, n, :), nblocks)
        acc = state.attempted[n]
        rate = SVector{NMOVETYPES, F}(ntuple(k -> iszero(acc[k]) ? zero(F) : F(state.accepted[n][k]) / F(acc[k]), NMOVETYPES))
        results[n] = NVTResult{F}(
            ē, ē_err, wr.mu_ex, wr.mu_ex_err, wr.K_H, wr.K_H_err, wr.q_st, wr.q_st_err, rate, n_production
        )
    end
    return results
end
