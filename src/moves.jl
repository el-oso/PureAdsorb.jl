# The three NVT Monte Carlo moves (translation, rotation, reinsertion) and the workgroup-parallel
# kernels that propose, evaluate and apply them. `bench/results/pureadsorb_chainsweep_*.json`
# (P5.3) showed no plateau from 1 to 4096 chains: the earlier per-move kernels put one work-item on
# one guest's whole host-atom and k-vector scan, so a single chain used one GPU thread. The kernels
# below put a WORKGROUP on one chain's proposal instead, splitting the host-atom loop
# (`host_guest_realspace_energy_range`), the guest-guest loop (`guest_guest_move_delta`) and the
# k-vector loop (`reciprocal_move_delta_energy`) across the workgroup's work-items and reducing
# cooperatively in `@localmem`. Adjacent work-items read adjacent host atoms of the SAME chain
# (`(a0+lane):L:(a0+natoms)`), which is what makes those reads coalesce; the earlier
# one-thread-per-proposal kernels instead had adjacent threads reading adjacent PROPOSALS, each
# against a different system's own ~74 KB block, an uncoalesced stride large enough on its own to
# matter.
#
# One workgroup is not always one whole move: `nblocks_per_chain` (`mc_step!`, `evaluate_move_kernel!`)
# additionally fans a single chain's evaluation across that many workgroups, each covering a
# `1/nblocks_per_chain` stride of the same host-atom/guest-guest/k-vector ranges, when there are too
# few chains to occupy the device with one workgroup each. This matters by precision: at one system,
# Float64 on this class of GPU is compute-bound enough (roughly 2 FP64 FMA/clock per SM) that a
# single workgroup pinned to one SM is slower than not fanning out at all, while Float32's much
# higher throughput already reaches the kernel-launch floor with one workgroup, leaving a second
# reduction stage nothing to buy back — `default_nblocks_per_chain` dispatches on the element type
# for exactly this reason, and `mc_step!` accepts an override since the right value is a property of
# the device and the move's size, not something this package can derive once and hard-code.
#
# `evaluate_move_kernel!` is register-heavy (measured 171-192 registers/thread on an RTX 4070,
# `sm_89`, depending on `groupsize`; `nvdisasm` shows several hundred spill loads/stores per
# compiled variant), which caps occupancy well below what the arithmetic alone would allow and is
# the dominant reason measured Float64 cost at `nsys = 1` (roughly 1.6-2 ms/move at the best
# `(groupsize, nblocks_per_chain)` point found by hand, `bench/mc_step_bench.jl`) has not reached
# the few-microsecond floor the card's FP64 throughput implies, nor kUPS's own measured 388 us/move
# at one system. The likely source is passing `batch::FrameworkBatch` and `guest::Guest` whole into
# every helper call rather than only the handful of fields each one reads, and the reduction's
# `nsteps`-fold unrolling multiplying live ranges across three separately-computed energy terms;
# neither is fixed here. `groupsize = 64` with `nblocks_per_chain` in the low tens measured
# consistently better than the defaults this file ships (`bench/mc_step_bench.jl`'s own sweep), but
# no configuration closed the gap.
#
# The evaluation, the accept/reject decision and the resulting `Sk` update are three separate kernel
# launches (`evaluate_move_kernel!`, `decide_move_kernel!`, `apply_sk_kernel!`) rather than one,
# because `nblocks_per_chain` workgroups computing one chain's move cannot share `@localmem` with
# each other — only work-items within the SAME workgroup can — so their partial sums must go through
# device memory (`MoveWorkspace.partial1`/`partial2`) and a synchronization point between kernel
# launches, which `KernelAbstractions.synchronize` already gives for free between the launches in
# `mc_step!`. `decide_move_kernel!` is one work-item per chain: cheap enough (a handful of RNG draws
# and a short serial sum over `nblocks_per_chain` partials) that it needs no workgroup of its own.
#
# Every chain in a batch performs the SAME move type on the SAME call (`movetype`, shared across
# the whole launch): each of translation, rotation and reinsertion individually satisfies detailed
# balance for the target distribution on its own, and a Markov chain built by applying any fixed
# sequence of individually-detailed-balanced kernels still has that distribution as its stationary
# distribution, since each step alone preserves it. What a FIXED (non-random) per-step order gives
# up is reversibility of the *composite* one-cycle transition (running translation then rotation is
# not, in general, its own time-reverse) — a palindromic ordering (e.g. translate, rotate,
# translate) would restore that composite reversibility, but reversibility of the composite is not
# needed for the stationary distribution to be preserved, only each step's own detailed balance is.

const MOVE_TRANSLATION = Int32(1)
const MOVE_ROTATION = Int32(2)
const MOVE_REINSERTION = Int32(3)

# Hamilton product in the (x, y, z, w) convention `rotate` uses (scalar last): `qmul(q2, q1)`
# composes "apply q1, then q2", i.e. `rotate(qmul(q2, q1), v) == rotate(q2, rotate(q1, v))`.
function qmul(q2::SVector{4, T}, q1::SVector{4, T}) where {T}
    x1, y1, z1, w1 = q1[1], q1[2], q1[3], q1[4]
    x2, y2, z2, w2 = q2[1], q2[2], q2[3], q2[4]
    w = w2 * w1 - (x2 * x1 + y2 * y1 + z2 * z1)
    x = w2 * x1 + w1 * x2 + (y2 * z1 - z2 * y1)
    y = w2 * y1 + w1 * y2 + (z2 * x1 - x2 * z1)
    z = w2 * z1 + w1 * z2 + (x2 * y1 - y2 * x1)
    return SVector(x, y, z, w)
end

"""
    quaternion_pow(q, t) -> SVector{4}

Unit quaternion `q` (angle `θ`, axis `u`, i.e. `q = (sin(θ/2)u, cos(θ/2))`) raised to the
fractional power `t ∈ [0, 1]`: `(sin(tθ/2)u, cos(tθ/2))`, the same axis with the rotation angle
scaled by `t` (kUPS's `Quaternion.__pow__`, `src/kups/core/utils/quaternion.py:184-210,242-264`).
`t = 0` gives the identity quaternion, `t = 1` returns `q` itself (up to roundoff). When `q`'s
vector part is (numerically) zero, `θ` is a multiple of `2π` and the rotation is the identity
regardless of axis, so the axis is taken as an arbitrary fixed direction rather than dividing by a
near-zero norm.
"""
function quaternion_pow(q::SVector{4, T}, t::T) where {T}
    w = clamp(q[4], -one(T), one(T))
    θ = 2 * acos(w)
    s = sqrt(max(one(T) - w * w, zero(T)))
    axis = s > eps(T) ? SVector(q[1], q[2], q[3]) / s : SVector(one(T), zero(T), zero(T))
    half = t * θ / 2
    sh = sin(half)
    return SVector(axis[1] * sh, axis[2] * sh, axis[3] * sh, cos(half))
end

"""
    propose_translation(rng, oldpos, step) -> (newpos, rng)

Gaussian displacement of the reference point, standard deviation `step` (Å) in each Cartesian
direction (R3: kUPS's own translation proposal is `distribution(key, (3,)) * step_width` with
`distribution` defaulting to `jax.random.normal`, not a uniform vector in a cube). Symmetric
about the current position since a normal distribution is symmetric about its mean, so
`metropolis_accept` needs no proposal-density correction for this move.
"""
function propose_translation(rng::ChainRNG, oldpos::SVector{3, T}, step::T) where {T}
    dx, rng = rand_normal(rng, T)
    dy, rng = rand_normal(rng, T)
    dz, rng = rand_normal(rng, T)
    return oldpos + step * SVector(dx, dy, dz), rng
end

"""
    propose_rotation(rng, oldq, step) -> (newq, rng)

Composes the current orientation with a fresh Shoemake-uniform random rotation raised to the
fractional power `step ∈ [0, 1]` (R3: kUPS's `Quaternion.random(key) ** step_width`,
`quaternion_pow`). Symmetric: the powered rotation's axis is uniform on the sphere and
independent of its (scaled) angle, so the rotation and its inverse (same axis negated, or
equivalently the same axis with the angle negated) are equally likely — see the "rotation
symmetry" test, which checks this directly against `shoemake_quaternion`/`quaternion_pow` rather
than trusting the argument.
"""
function propose_rotation(rng::ChainRNG, oldq::SVector{4, T}, step::T) where {T}
    qrand, rng = rand_quaternion(rng, T)
    qpow = quaternion_pow(qrand, step)
    return normalize(qmul(qpow, oldq)), rng
end

"""
    propose_reinsertion(rng, cell) -> (newpos, newq, rng)

Fresh uniform position in `cell` and fresh uniform orientation, independent of the guest's
current pose (R3: kUPS's reinsertion always uses a full random rotation and a translation
uniform over the whole cell, `src/kups/mcmc/moves.py:268-296`, which composes to a
position/orientation independent of the starting pose). Trivially symmetric since the proposal
density does not depend on the current state at all.
"""
function propose_reinsertion(rng::ChainRNG, cell::SMatrix{3, 3, T}) where {T}
    f1, rng = rand_uniform(rng, T)
    f2, rng = rand_uniform(rng, T)
    f3, rng = rand_uniform(rng, T)
    newpos = cell * SVector(f1, f2, f3)
    newq, rng = rand_quaternion(rng, T)
    return newpos, newq, rng
end

"""
    propose_move(rng, movetype, oldpos, oldq, step_trans, step_rot, cell) -> (newpos, newq, rng)

Dispatches to `propose_translation`, `propose_rotation` or `propose_reinsertion` by `movetype`
(`MOVE_TRANSLATION`, `MOVE_ROTATION`, `MOVE_REINSERTION`); anything else falls through to
reinsertion rather than throwing, since a kernel may not throw — `mc_step!` validates `movetype`
once, at the host, before any kernel launch. All three proposals are symmetric (see their own
docstrings), so `metropolis_accept`'s log-ratio needs no proposal-density term regardless of
which one ran.
"""
function propose_move(
        rng::ChainRNG, movetype::Int32, oldpos::SVector{3, T}, oldq::SVector{4, T}, step_trans::T, step_rot::T,
        cell::SMatrix{3, 3, T}
    ) where {T}
    if movetype == MOVE_TRANSLATION
        newpos, rng = propose_translation(rng, oldpos, step_trans)
        return newpos, oldq, rng
    elseif movetype == MOVE_ROTATION
        newq, rng = propose_rotation(rng, oldq, step_rot)
        return oldpos, newq, rng
    else
        return propose_reinsertion(rng, cell)
    end
end

"""
    metropolis_accept(ΔU, kT, u) -> Bool

Metropolis acceptance test for a symmetric proposal: `accept = log_p_ratio > log(u)` with
`log_p_ratio = -ΔU/kT` (kUPS's own `src/kups/core/propagator.py:325`), `u` a single `[0, 1)`
uniform draw. A non-finite `ΔU` (an overlapping pose driving the real-space term to `Inf`, or a
`NaN` from a corrupted intermediate) is rejected by the explicit `isfinite` check below rather
than by `-Inf > log(u)`/`NaN > log(u)` merely happening to be `false` under IEEE-754 comparison —
R7: kUPS gets the same outcome by that accident; this makes it a designed code path instead.
"""
function metropolis_accept(ΔU::T, kT::T, u::T) where {T}
    isfinite(ΔU) || return false
    return -ΔU / kT > log(u)
end

# One-hot `SVector{NMOVETYPES, Int32}` with a `1` in slot `movetype` (1-based `MOVE_TRANSLATION`/
# `MOVE_ROTATION`/`MOVE_REINSERTION`), branch-free, for bumping `SystemState`'s per-move-type
# `accepted`/`attempted` counters.
onehot_movetype(movetype::Int32) = SVector{NMOVETYPES, Int32}(ntuple(k -> Int32(Int32(k) == movetype), NMOVETYPES))

# The guest's own site types, reindexed to `batch`'s compact Lennard-Jones type index: every
# host-guest energy function in this file and in `guest.jl` expects `guest.types` in that index,
# matching `batch.sigma`/`batch.epsilon` (the same construction `total_energy`/`guest_move_delta`
# each build inline).
compact_guest(batch::FrameworkBatch{T}, guest::Guest{T, N}) where {T, N} =
    Guest{T, N}(guest.sites, SVector{N, Int}(batch.guest_types), guest.charges, guest.tc, guest.pc, guest.omega)

# Deterministically reproduces chain `n`'s guest selection and proposed new pose from
# `(rng_seed[n], rng_counter[n])` alone: every work-item of every `evaluate_move_kernel!` workgroup
# assigned to chain `n` calls this identically to know what to evaluate, and `decide_move_kernel!`
# calls it again (a separate kernel launch, so this is an ordinary, unproblematic re-read of
# `refpoints`/`rng_counter` rather than a cross-synchronize-point rerun) to know what to commit —
# recomputing a few RNG draws and a handful of flops a second time is cheaper than passing the
# proposal through device memory. `nrefpoints` is `length(refpoints)`, passed in rather than
# recomputed so every call site uses the identical value. A guest-less chain (`Ng == 0`) has no
# valid slot at all; `gr0 + gidx_local` (`== gr0 + 1`, since `rand_range` never sees `Ng` below `1`)
# can then exceed `nrefpoints` when chain `n` is the last one and empty, so the index is clamped
# into range. The "guest" this reads (and, if committed, would write) is fictitious for such a
# chain, but `decide_move_kernel!` forces `accept = false` whenever `Ng` is zero, so nothing from it
# is ever actually committed.
@inline function select_and_propose(
        n::Integer, guest_offsets, rng_seed, rng_counter, movetype::Int32, step_trans, step_rot, cell::SMatrix{3, 3, T},
        refpoints, orientations, nrefpoints::Integer
    ) where {T}
    gr0 = guest_offsets[n]
    Ng = guest_offsets[n + 1] - gr0
    Ng_eff = max(Ng, one(Ng))
    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    gidx_local, rng = rand_range(rng, Ng_eff)
    i = clamp(gr0 + gidx_local, one(gr0), Int32(nrefpoints))
    oldpos = refpoints[i]; oldq = orientations[i]
    newpos, newq, rng = propose_move(rng, movetype, oldpos, oldq, step_trans[n], step_rot[n], cell)
    return i, oldpos, oldq, newpos, newq, Ng, rng
end

"""
    default_nblocks_per_chain(::Type{T}, nsys, target_blocks = 256) -> Int

Workgroups to fan a single chain's move evaluation across, per chain. For `T = Float32`, always
`1`: Float32's throughput already drives one workgroup's evaluation down near the kernel-launch
floor, so a second reduction stage (another pair of kernel launches, through `decide_move_kernel!`
and `apply_sk_kernel!`) has nothing left to buy back. For every other type (`Float64` in
particular), `nsys` chains already give `nsys` independent workgroups (one per chain) once `nsys`
reaches `target_blocks`, so this is `1` there too; below that, each chain's own host-atom,
guest-guest and k-vector loops are additionally split across `cld(target_blocks, nsys)` workgroups.
This matters specifically for Float64: a 4070 does roughly 2 FP64 FMA/clock per SM, so a single
workgroup pinned to one SM evaluates a move an order of magnitude slower than spreading the same
work across the device's ~46 SMs. `target_blocks` is a fixed, portable default rather than a
per-device occupancy query: this project's other KernelAbstractions kernels run unchanged on CPU,
CUDA, ROCm and Metal without reading back a device's SM/CU count, and this keeps that property —
`mc_step!` accepts an explicit override for callers that have measured a better value for their
device.
"""
default_nblocks_per_chain(::Type{Float32}, nsys::Integer, target_blocks::Integer = 256) = 1
default_nblocks_per_chain(::Type{T}, nsys::Integer, target_blocks::Integer = 256) where {T} =
    max(1, cld(Int(target_blocks), max(Int(nsys), 1)))

"""
    MoveWorkspace{T}

Device-memory scratch `mc_step!` reuses across calls so a move attempt allocates nothing:
`partial1`/`partial2` hold each chain's up-to-`nblocks_max` per-workgroup partial sums from
`evaluate_move_kernel!` (`decide_move_kernel!`'s input), shaped `(nblocks_max, nsys)` so any call
may use `nblocks_per_chain <= nblocks_max`; `accept_flag`, `move_oldpos`, `move_oldq`,
`move_newpos`, `move_newq` are `decide_move_kernel!`'s per-chain output and `apply_sk_kernel!`'s
input — this handoff is what lets `evaluate_move_kernel!`'s several workgroups per chain
communicate with each other at all, since only work-items within the SAME workgroup share
`@localmem`. Build with `MoveWorkspace(T, nsys, nblocks_max; backend)` and adapt it to `backend`
once, like `batch`/`state`, rather than per call.
"""
struct MoveWorkspace{VF, VU8, VP, VQ}
    partial1::VF
    partial2::VF
    accept_flag::VU8
    move_oldpos::VP
    move_oldq::VQ
    move_newpos::VP
    move_newq::VQ
    nblocks_max::Int
end
Adapt.@adapt_structure MoveWorkspace

function MoveWorkspace(::Type{T}, nsys::Integer, nblocks_max::Integer; backend = CPU()) where {T}
    return MoveWorkspace(
        adapt(backend, zeros(T, Int(nblocks_max), Int(nsys))),
        adapt(backend, zeros(T, Int(nblocks_max), Int(nsys))),
        adapt(backend, zeros(UInt8, Int(nsys))),
        adapt(backend, zeros(SVector{3, T}, Int(nsys))),
        adapt(backend, zeros(SVector{4, T}, Int(nsys))),
        adapt(backend, zeros(SVector{3, T}, Int(nsys))),
        adapt(backend, zeros(SVector{4, T}, Int(nsys))),
        Int(nblocks_max)
    )
end

"""
    evaluate_move_kernel!(partial1, partial2, refpoints, orientations, Sk, batch, guest,
                          guest_types, guest_offsets, k_offsets, rng_seed, rng_counter, movetype,
                          step_trans, step_rot, nblocks_per_chain, ::Val{G})

`nblocks_per_chain` workgroups of `G` work-items evaluate one chain's proposal
(`ndrange = (G, nblocks_per_chain, nsys)`, `@index(Group, NTuple) == (1, block, chain)` since the
first `ndrange` axis is exactly one workgroup wide). Every work-item in every workgroup assigned
to chain `n` independently replays the SAME deterministic draw
(`select_and_propose(n, ...)`, keyed only on `(rng_seed[n], rng_counter[n])`), so no
cross-workgroup communication is needed to agree on which guest moved or where to. Each work-item
then takes its own disjoint, stride-`L` slice (`L = nblocks_per_chain*G`, the total work-items
assigned to chain `n` across every one of its workgroups) of the host-atom loop
(`host_guest_realspace_energy_range`), the guest-guest loop (`guest_guest_move_delta`) and the
k-vector loop (`reciprocal_move_delta_energy`); the workgroup reduces its own work-items'
contributions (`buf1`/`buf2`, a shared-memory tree reduction) into one partial sum per channel,
written to `partial1[block, n]`/`partial2[block, n]`. `decide_move_kernel!` sums across blocks.
"""
@kernel function evaluate_move_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(k_offsets), @Const(rng_seed), @Const(rng_counter),
        movetype::Int32, @Const(step_trans), @Const(step_rot), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    i, oldpos, oldq, newpos, newq, _, _ = select_and_propose(
        n, guest_offsets, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[n], refpoints,
        orientations, length(refpoints)
    )

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    A = batch.cells[n]
    gr0 = guest_offsets[n]
    a0 = batch.atom_offsets[n]
    natoms = batch.atom_offsets[n + 1] - a0
    invA = batch.invcells[n]; alpha = batch.alphas[n]
    e_new_partial = host_guest_realspace_energy_range(
        newpos, newq, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, (a0 + lane):L:(a0 + natoms), A, invA, alpha
    )

    gr_lo = gr0 + one(gr0); gr_hi = guest_offsets[n + 1]
    ΔU_gg_partial = guest_guest_move_delta(
        guest, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
    ΔU_recip_partial = reciprocal_move_delta_energy(
        guest, oldpos, oldq, newpos, newq, view(batch.ks, (kr_lo + lane - 1):L:kr_hi),
        view(batch.kprefactor, (kr_lo + lane - 1):L:kr_hi), view(Sk, (kr_lo + lane - 1):L:kr_hi)
    )

    buf1 = @localmem T (G,)
    buf2 = @localmem T (G,)
    buf1[tid] = e_new_partial
    buf2[tid] = ΔU_gg_partial + ΔU_recip_partial
    @synchronize()
    for step in 1:nsteps
        stride = G >> step
        if tid <= stride
            buf1[tid] += buf1[tid + stride]
            buf2[tid] += buf2[tid + stride]
        end
        @synchronize()
    end
    # Re-fetched rather than reusing `b`/`n` from above: KernelAbstractions' CPU backend compiles
    # the code on either side of a `@synchronize()` as separate closures, so a plain local computed
    # before one is not guaranteed to survive to be read after it (see `select_and_propose`'s
    # comment for the same point about chain-level state).
    grp2 = @index(Group, NTuple)
    b2, n2 = grp2[2], grp2[3]
    tid == 1 && (partial1[b2, n2] = buf1[1]; partial2[b2, n2] = buf2[1])
end

"""
    decide_move_kernel!(refpoints, orientations, host_energy, energy, rng_counter, accepted,
                        attempted, accept_flag, move_oldpos, move_oldq, move_newpos, move_newq,
                        partial1, partial2, batch, guest, guest_offsets, rng_seed, movetype,
                        step_trans, step_rot, kT, nblocks_per_chain)

One work-item per chain (`ndrange = nsys`). Redraws chain `n`'s proposal (cheap: a handful of RNG
draws and flops, unlike `evaluate_move_kernel!`'s fanned-out scan), sums `evaluate_move_kernel!`'s
`nblocks_per_chain` partial sums into `ΔU` together with the cached OLD host-guest energy
(`host_energy[i]`, P2's per-guest cache — never re-scanned here), and applies `metropolis_accept`
with the next draw from the same stream. `Ng` (the chain's guest count) being zero forces
`accept = false` directly rather than branching around any of this, since a guest-less chain still
needs a (fictitious) pose to read/propose without indexing out of range — see
`select_and_propose`'s comment. On acceptance, commits the pose/`host_energy`/`energy` in place;
always advances `rng_counter`/`accepted`/`attempted` (attempt counters, so they advance regardless
of acceptance). Always records the outcome (`accept_flag`) and the old/new pose in the workspace,
regardless of acceptance, for `apply_sk_kernel!` to read; it only acts on chains whose
`accept_flag` is set.
"""
@kernel function decide_move_kernel!(
        refpoints, orientations, host_energy, energy, rng_counter, accepted, attempted, accept_flag, move_oldpos,
        move_oldq, move_newpos, move_newq, @Const(partial1), @Const(partial2), batch, guest::Guest{T, N},
        @Const(guest_offsets), @Const(rng_seed), movetype::Int32, @Const(step_trans), @Const(step_rot), kT::T,
        nblocks_per_chain::Int32
    ) where {T, N}
    n = @index(Global, Linear)
    i, oldpos, oldq, newpos, newq, Ng, rng = select_and_propose(
        n, guest_offsets, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[n], refpoints,
        orientations, length(refpoints)
    )
    e_new = zero(T); rest = zero(T)
    for blk in Int32(1):nblocks_per_chain
        e_new += partial1[blk, n]
        rest += partial2[blk, n]
    end
    ΔU = (e_new - host_energy[i]) + rest

    u, rng = rand_uniform(rng, T)
    accept = metropolis_accept(ΔU, kT, u) & !iszero(Ng)
    if accept
        refpoints[i] = newpos
        orientations[i] = newq
        host_energy[i] = e_new
        energy[n] += ΔU
    end
    delta = onehot_movetype(movetype)
    attempted[n] = attempted[n] + delta
    accept && (accepted[n] = accepted[n] + delta)
    rng_counter[n] = rng_counter[n] + one(eltype(rng_counter))

    accept_flag[n] = UInt8(accept)
    move_oldpos[n] = oldpos; move_oldq[n] = oldq
    move_newpos[n] = newpos; move_newq[n] = newq
end

"""
    apply_sk_kernel!(Sk, ks, accept_flag, move_oldpos, move_oldq, move_newpos, move_newq, guest,
                     k_offsets)

Applies the accepted chains' structure-factor change, fanned across `ndrange = (L, nsys)`
work-items (`L` any convenient lane count — `mc_step!` reuses `nblocks_per_chain*groupsize`, but
this kernel does not need it to match `evaluate_move_kernel!`'s own `L`) with no workgroup
reduction needed: `Sk[kk] += ds` for a disjoint `kk` per work-item, since the `L` lanes assigned to
one chain partition its k-range exactly once. Rotations are hoisted out of the k-loop
(`guest_sites_at` once per chain, not per k — P3), matching every other reciprocal-space loop in
this package. A rejected chain's `accept_flag` is `0` and this kernel does nothing for it, so a
rejected move leaves `Sk` untouched, exactly as `guest_move_delta`'s own contract requires.
"""
@kernel function apply_sk_kernel!(
        Sk, @Const(ks), @Const(accept_flag), @Const(move_oldpos), @Const(move_oldq), @Const(move_newpos),
        @Const(move_newq), guest::Guest{T, N}, @Const(k_offsets)
    ) where {T, N}
    idx = @index(Global, NTuple)
    lane, n = idx[1], idx[2]
    L = @ndrange()[1]
    if !iszero(accept_flag[n])
        old_sites = guest_sites_at(guest, move_oldpos[n], move_oldq[n])
        new_sites = guest_sites_at(guest, move_newpos[n], move_newq[n])
        kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
        for kk in (kr_lo + lane - 1):L:kr_hi
            ds, _ = _reciprocal_move_delta_k(ks[kk], guest.charges, old_sites, new_sites, Sk[kk])
            Sk[kk] += ds
        end
    end
end

"""
    mc_step!(ws, batch, state, guest, guest_types, movetype, step_trans, step_rot, kT;
             backend = CPU(), groupsize = 256,
             nblocks_per_chain = default_nblocks_per_chain(T, state.nsys)) -> nothing

One Metropolis Monte Carlo move attempt for every chain in `state`, all performing the same
`movetype` (`MOVE_TRANSLATION`, `MOVE_ROTATION` or `MOVE_REINSERTION`) — see this file's opening
comment for why a schedule fixed across the batch still preserves the target distribution, and for
what `nblocks_per_chain` fans out and why the fan-out is precision-dependent. `ws` (`MoveWorkspace`),
`batch` and `state` must already be resident on `backend`; adapt them once outside a chain's move
loop, not on every call. `guest` must already be reindexed to `batch`'s compact type
(`compact_guest`). `step_trans` (Å) and `step_rot` (in `[0, 1]`) are per-chain vectors, used exactly
as given (R4: this function has no adaptation logic of its own, so a step size is frozen for as
long as the caller keeps passing the same vector). `kT` is Boltzmann's constant times the
temperature, in the same energy units as `batch`/`guest`.

Three kernel launches: `evaluate_move_kernel!` fans each chain's proposal across
`nblocks_per_chain` workgroups of `groupsize` work-items each; `decide_move_kernel!` (one work-item
per chain) sums their partial sums and applies `metropolis_accept`; `apply_sk_kernel!` fans back
out to update `state.Sk` on the chains that accepted. `groupsize` must be a power of two (the
reduction halves it every step) and `nblocks_per_chain` must not exceed `ws.nblocks_max`.
"""
function mc_step!(
        ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        movetype::Integer, step_trans, step_rot, kT::T;
        backend = CPU(), groupsize::Integer = 256, nblocks_per_chain::Integer = default_nblocks_per_chain(T, state.nsys)
    ) where {T, N}
    movetype in (MOVE_TRANSLATION, MOVE_ROTATION, MOVE_REINSERTION) || throw(
        ArgumentError("mc_step!: movetype=$movetype must be MOVE_TRANSLATION, MOVE_ROTATION or MOVE_REINSERTION")
    )
    ispow2(groupsize) || throw(ArgumentError("mc_step!: groupsize=$groupsize must be a power of two"))
    nblocks_per_chain >= 1 || throw(ArgumentError("mc_step!: nblocks_per_chain=$nblocks_per_chain must be >= 1"))
    nblocks_per_chain <= ws.nblocks_max ||
        throw(ArgumentError("mc_step!: nblocks_per_chain=$nblocks_per_chain exceeds the workspace's nblocks_max=$(ws.nblocks_max)"))
    G = Int(groupsize)
    nbpc = Int32(nblocks_per_chain)
    mt = Int32(movetype)
    nsys = state.nsys

    evaluate_move_kernel!(backend)(
        ws.partial1, ws.partial2, state.refpoints, state.orientations, state.Sk, batch, guest, guest_types,
        state.guest_offsets, state.k_offsets, state.rng_seed, state.rng_counter, mt, step_trans, step_rot, nbpc, Val(G);
        ndrange = (G, nblocks_per_chain, nsys), workgroupsize = (G, 1, 1)
    )
    KernelAbstractions.synchronize(backend)

    decide_move_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.energy, state.rng_counter, state.accepted,
        state.attempted, ws.accept_flag, ws.move_oldpos, ws.move_oldq, ws.move_newpos, ws.move_newq, ws.partial1,
        ws.partial2, batch, guest, state.guest_offsets, state.rng_seed, mt, step_trans, step_rot, kT, nbpc;
        ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)

    apply_sk_kernel!(backend)(
        state.Sk, batch.ks, ws.accept_flag, ws.move_oldpos, ws.move_oldq, ws.move_newpos, ws.move_newq, guest,
        state.k_offsets;
        ndrange = (nblocks_per_chain * G, nsys)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
