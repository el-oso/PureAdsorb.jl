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
# `evaluate_move_kernel!` is register-heavy (measured via `CUDA.registers`/`CUDA.occupancy` on the
# actual compiled kernel object, not estimated: 168-194 registers/thread on an RTX 4070, `sm_89`,
# depending on `groupsize`, with several KB/thread of local-memory spill), which caps theoretical
# occupancy at 17-25% of the device. Isolating `host_guest_realspace_energy_range` (106
# registers alone), `guest_guest_move_delta` (162) and `reciprocal_move_delta_energy` (80) shows
# `guest_guest_move_delta` accounts for nearly all of it: it evaluates the full `N x N` guest-guest
# pair energy twice per other guest (once at the old pose, once at the new one), each pair costing
# a Chebyshev-recurrence `erfc` evaluation, and this is fully unrolled since `N` is a compile-time
# constant. Passing only the fields each helper reads instead of `batch`/`guest` whole, and fusing
# the old/new pair-energy loops into one, were both tried and measured (not merely reasoned about):
# neither moved the register count by more than a few percent, so the register pressure is intrinsic
# to that unrolled computation, not to how the arguments are passed.
#
# That register pressure is real, but it turned out not to be what a first benchmarking pass was
# actually measuring. A single warm-up call before timing leaves the GPU at its idle clock state;
# reaching the boosted clock the timed samples need takes on the order of 100-150 calls (tens of
# milliseconds), measured directly by timing `mc_step!` after an increasing number of untimed
# calls. Benchmarked cold this way, `mc_step!` appeared to cost roughly 1.6-2.8 ms/move at
# `nsys = 1` (Float64); with the device properly warmed up first (`bench/mc_step_bench.jl`'s fix),
# the SAME unmodified kernel costs 180-230 us/move there with `groupsize = 64`, under kUPS's own
# measured 388 us/move at one system, and less per move again at larger chain counts
# (`bench/results/pureadsorb_mcstep_*.json`). No source change was needed to close that gap; the
# occupancy ceiling above is a real, measured property of this kernel, just not one this milestone's
# target required lifting.
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

# `mc_step!`'s default `groupsize`, chosen by the minimax ratio to the best groupsize measured
# across {32, 64, 128, 256} at nsys ∈ {1, 64, 256}, both precisions, on an RTX 4070
# (`bench/results/pureadsorb_groupsizesweep_*.json`, `bench/results/README.md`): 64 is never more
# than 45% above the best groupsize in any measured cell, against 59-177% for the alternatives.
const DEFAULT_GROUPSIZE = 64

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

"""
    log_insertion_prefactor(f, V, kT, N) -> log(f V / ((N+1) kT))

Log of the μVT insertion acceptance ratio's combinatorial prefactor
(`docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`, "Acceptance ratios"), `N` the system's
occupancy BEFORE the insertion (so `N+1` is the occupancy the move would produce):
`A_ins = min(1, exp(log_insertion_prefactor(f, V, kT, N) - ΔU/kT))`.
"""
function log_insertion_prefactor(f::T, V::T, kT::T, N::Integer) where {T}
    return log(f * V) - log(T(N) + one(T)) - log(kT)
end

"""
    log_deletion_prefactor(f, V, kT, N) -> log(N kT / (f V))

Log of the μVT deletion acceptance ratio's combinatorial prefactor, `N` the system's occupancy
BEFORE the deletion (`N >= 1`). Deleting the guest that an insertion into the `(N-1)`-guest system
would add back is that insertion's detailed-balance partner, so this is defined as the exact
negative of `log_insertion_prefactor` evaluated at `N-1` guests, not as a second, independently
typed formula that could disagree with the first: `log_insertion_prefactor(f, V, kT, N-1) ==
log(f V) - log(N) - log(kT)`, whose negative is `log(N kT / (f V))` — `A_del`'s own log-prefactor,
reached here by construction rather than by re-deriving it.
"""
function log_deletion_prefactor(f::T, V::T, kT::T, N::Integer) where {T}
    return -log_insertion_prefactor(f, V, kT, N - 1)
end

"""
    metropolis_accept_muvt(ΔU, kT, log_prefactor, u) -> Bool

μVT Metropolis test: `accept = log_prefactor - ΔU/kT > log(u)`, `log_prefactor` one of
`log_insertion_prefactor`/`log_deletion_prefactor`. Same deliberate non-finite guard as
`metropolis_accept` (R7): a non-finite `ΔU` is rejected by this explicit check, not by an
IEEE-754 comparison merely happening to return `false`.
"""
function metropolis_accept_muvt(ΔU::T, kT::T, log_prefactor::T, u::T) where {T}
    isfinite(ΔU) || return false
    return log_prefactor - ΔU / kT > log(u)
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
# recomputed so every call site uses the identical value. `Ng` is the chain's LIVE occupancy
# (`occupancy[n]`, `guest_range(state, n)`'s length), not its reserved capacity: a slot beyond
# occupancy holds a stale or sentinel pose no move may touch. A guest-less chain (`Ng` zero) has no
# valid slot at all; `gr0 + gidx_local` (`== gr0 + 1`, since `rand_range` never sees `Ng` below `1`)
# can then exceed `nrefpoints` when chain `n` is the last one and reserves no capacity at all, so
# the index is clamped into range. The "guest" this reads (and, if committed, would write) is
# fictitious for such a chain, but `decide_move_kernel!` forces `accept = false` whenever `Ng` is
# zero, so nothing from it is ever actually committed.
@inline function select_and_propose(
        n::Integer, guest_offsets, occupancy, rng_seed, rng_counter, movetype::Int32, step_trans, step_rot,
        cell::SMatrix{3, 3, T}, refpoints, orientations, nrefpoints::Integer
    ) where {T}
    gr0 = guest_offsets[n]
    Ng = occupancy[n]
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
                          guest_types, guest_offsets, occupancy, k_offsets, rng_seed, rng_counter,
                          movetype, step_trans, step_rot, nblocks_per_chain, ::Val{G})

`nblocks_per_chain` workgroups of `G` work-items evaluate one chain's proposal
(`ndrange = (G, nblocks_per_chain, nsys)`, `@index(Group, NTuple) == (1, block, chain)` since the
first `ndrange` axis is exactly one workgroup wide). Every work-item in every workgroup assigned
to chain `n` independently replays the SAME deterministic draw
(`select_and_propose(n, ...)`, keyed only on `(rng_seed[n], rng_counter[n])`), so no
cross-workgroup communication is needed to agree on which guest moved or where to. Each work-item
then takes its own disjoint, stride-`L` slice (`L = nblocks_per_chain*G`, the total work-items
assigned to chain `n` across every one of its workgroups) of the host-atom loop
(`host_guest_realspace_energy_range`), the guest-guest loop (`guest_guest_move_delta`, bounded by
`occupancy[n]` rather than the chain's reserved capacity — a slot beyond occupancy holds no live
guest) and the k-vector loop (`reciprocal_move_delta_energy`); the workgroup reduces its own
work-items' contributions (`buf1`/`buf2`, a shared-memory tree reduction) into one partial sum per
channel, written to `partial1[block, n]`/`partial2[block, n]`. `decide_move_kernel!` sums across
blocks.
"""
@kernel function evaluate_move_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(occupancy), @Const(k_offsets), @Const(rng_seed),
        @Const(rng_counter), movetype::Int32, @Const(step_trans), @Const(step_rot), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    fw = batch.framework_of[n]
    i, oldpos, oldq, newpos, newq, _, _ = select_and_propose(
        n, guest_offsets, occupancy, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[fw], refpoints,
        orientations, length(refpoints)
    )

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    A = batch.cells[fw]
    gr0 = guest_offsets[n]
    a0 = batch.atom_offsets[fw]
    natoms = batch.atom_offsets[fw + 1] - a0
    invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    e_new_partial = host_guest_realspace_energy_range(
        newpos, newq, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, (a0 + lane):L:(a0 + natoms), A, invA, alpha
    )

    gr_lo = gr0 + one(gr0); gr_hi = gr0 + occupancy[n]
    ΔU_gg_partial = guest_guest_move_delta(
        guest, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    # `k_offsets` (this kernel's own argument, `state.k_offsets`) ranges over `Sk`, which stays
    # sized one slice per SYSTEM even when systems share a framework; `batch.k_offsets` ranges
    # over `batch.ks`/`batch.kprefactor`, deduplicated to one slice per FRAMEWORK
    # (`FrameworkBatch`'s docstring). The two ranges have the same length (framework `fw`'s own
    # k-vector count) but different absolute starts, so each is built from its own offsets and
    # paired by the shared relative stride `lane:L`, never by one absolute range applied to both.
    kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
    kr_lo_b = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kr_hi_b = batch.k_offsets[fw + 1]
    ΔU_recip_partial = reciprocal_move_delta_energy(
        guest, oldpos, oldq, newpos, newq, view(batch.ks, (kr_lo_b + lane - 1):L:kr_hi_b),
        view(batch.kprefactor, (kr_lo_b + lane - 1):L:kr_hi_b), view(Sk, (kr_lo + lane - 1):L:kr_hi)
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
    decide_move_kernel!(refpoints, orientations, host_energy, energy, energy_abs_accum, rng_counter,
                        accepted, attempted, accept_flag, move_oldpos, move_oldq, move_newpos, move_newq,
                        partial1, partial2, batch, guest, guest_offsets, occupancy, rng_seed, movetype,
                        step_trans, step_rot, kT, nblocks_per_chain)

One work-item per chain (`ndrange = nsys`). Redraws chain `n`'s proposal (cheap: a handful of RNG
draws and flops, unlike `evaluate_move_kernel!`'s fanned-out scan), sums `evaluate_move_kernel!`'s
`nblocks_per_chain` partial sums into `ΔU` together with the cached OLD host-guest energy
(`host_energy[i]`, P2's per-guest cache — never re-scanned here), and applies `metropolis_accept`
with the next draw from the same stream. `Ng` (`occupancy[n]`, the chain's LIVE guest count) being
zero forces `accept = false` directly rather than branching around any of this, since a guest-less
chain still needs a (fictitious) pose to read/propose without indexing out of range — see
`select_and_propose`'s comment. On acceptance, commits the pose/`host_energy`/`energy` in place and
adds `abs(ΔU)` to `energy_abs_accum[n]` (`audit_energy!`'s tolerance scale, `SystemState`'s
docstring); always advances `rng_counter`/`accepted`/`attempted` (attempt counters, so they advance
regardless of acceptance). Always records the outcome (`accept_flag`) and the old/new pose in the
workspace, regardless of acceptance, for `apply_sk_kernel!` to read; it only acts on chains whose
`accept_flag` is set.
"""
@kernel function decide_move_kernel!(
        refpoints, orientations, host_energy, energy, energy_abs_accum, rng_counter, accepted, attempted, accept_flag,
        move_oldpos, move_oldq, move_newpos, move_newq, @Const(partial1), @Const(partial2), batch, guest::Guest{T, N},
        @Const(guest_offsets), @Const(occupancy), @Const(rng_seed), movetype::Int32, @Const(step_trans), @Const(step_rot),
        kT::T, nblocks_per_chain::Int32
    ) where {T, N}
    n = @index(Global, Linear)
    i, oldpos, oldq, newpos, newq, Ng, rng = select_and_propose(
        n, guest_offsets, occupancy, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[batch.framework_of[n]],
        refpoints, orientations, length(refpoints)
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
        energy_abs_accum[n] += abs(ΔU)
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
    apply_sk_kernel!(Sk, sk_abs_accum, ks, accept_flag, move_oldpos, move_oldq, move_newpos,
                     move_newq, guest, k_offsets)

Applies the accepted chains' structure-factor change, fanned across `ndrange = (L, nsys)`
work-items (`L` any convenient lane count — `mc_step!` reuses `nblocks_per_chain*groupsize`, but
this kernel does not need it to match `evaluate_move_kernel!`'s own `L`) with no workgroup
reduction needed: `Sk[kk] += ds` for a disjoint `kk` per work-item, since the `L` lanes assigned to
one chain partition its k-range exactly once. `sk_abs_accum[kk] += 2*Σ|guest.charges|` alongside
it, feeding `audit_energy!`'s tolerance (`SystemState`'s docstring): `ds = Snew - Sold` with
`|Snew|, |Sold| <= Σ|guest.charges|` (each is a sum of unit-modulus phases times a charge), so
`2*Σ|guest.charges|` is a guaranteed upper bound on `abs(ds)` that also, unlike `abs(ds)` itself,
stays representative when a small move makes `Snew` and `Sold` nearly cancel — exactly the
regime where `abs(ds)` alone understates the rounding a direct `Snew - Sold` subtraction carries.
It is the same value at every k-vector and every accepted move (the guest's charges never
change), so it costs nothing to recompute per work-item. Rotations are hoisted out of the k-loop
(`guest_sites_at` once per chain, not per k — P3), matching every other reciprocal-space loop in
this package. A rejected chain's `accept_flag` is `0` and this kernel does nothing for it, so a
rejected move leaves `Sk`/`sk_abs_accum` untouched, exactly as `guest_move_delta`'s own contract
requires.
"""
@kernel function apply_sk_kernel!(
        Sk, sk_abs_accum, batch, @Const(accept_flag), @Const(move_oldpos), @Const(move_oldq), @Const(move_newpos),
        @Const(move_newq), guest::Guest{T, N}, @Const(k_offsets)
    ) where {T, N}
    idx = @index(Global, NTuple)
    lane, n = idx[1], idx[2]
    L = @ndrange()[1]
    if !iszero(accept_flag[n])
        old_sites = guest_sites_at(guest, move_oldpos[n], move_oldq[n])
        new_sites = guest_sites_at(guest, move_newpos[n], move_newq[n])
        ds_bound = 2 * sum(abs, guest.charges)
        # `k_offsets` (this kernel's own argument, `state.k_offsets`) ranges over `Sk`, one slice
        # per SYSTEM; `batch.k_offsets` ranges over `batch.ks`, one slice per FRAMEWORK
        # (`FrameworkBatch`'s docstring) — same length for system `n`'s framework, different
        # absolute start, so the loop pairs them by the shared relative offset `rel`.
        fw = batch.framework_of[n]
        kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
        kr_lo_b = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kr_hi_b = batch.k_offsets[fw + 1]
        for rel in lane:L:(kr_hi - kr_lo + one(kr_hi))
            kk = kr_lo + rel - one(rel)
            kk_b = kr_lo_b + rel - one(rel)
            ds, _ = _reciprocal_move_delta_k(batch.ks[kk_b], guest.charges, old_sites, new_sites, Sk[kk])
            Sk[kk] += ds
            sk_abs_accum[kk] += ds_bound
        end
    end
end

"""
    mc_step!(ws, batch, state, guest, guest_types, movetype, step_trans, step_rot, kT;
             backend = CPU(), groupsize = DEFAULT_GROUPSIZE,
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
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = default_nblocks_per_chain(T, state.nsys)
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
        state.guest_offsets, state.occupancy, state.k_offsets, state.rng_seed, state.rng_counter, mt, step_trans,
        step_rot, nbpc, Val(G);
        ndrange = (G, nblocks_per_chain, nsys), workgroupsize = (G, 1, 1)
    )
    KernelAbstractions.synchronize(backend)

    decide_move_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.energy, state.energy_abs_accum,
        state.rng_counter, state.accepted, state.attempted, ws.accept_flag, ws.move_oldpos, ws.move_oldq,
        ws.move_newpos, ws.move_newq, ws.partial1, ws.partial2, batch, guest, state.guest_offsets, state.occupancy,
        state.rng_seed, mt, step_trans, step_rot, kT, nbpc;
        ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)

    apply_sk_kernel!(backend)(
        state.Sk, state.sk_abs_accum, batch, ws.accept_flag, ws.move_oldpos, ws.move_oldq, ws.move_newpos,
        ws.move_newq, guest, state.k_offsets;
        ndrange = (nblocks_per_chain * G, nsys)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# The μVT exchange moves (insertion, deletion): unlike the NVT moves above, these change a
# system's OCCUPANCY, not just a guest's pose, so each is one work-item per chain (`ndrange =
# nsys`, no workgroup fan-out across a chain's own host-atom/guest-guest/k-vector loops) rather
# than `mc_step!`'s three-kernel evaluate/decide/apply split: a chain's own capacity block is only
# ever written by that chain's own work-item, so there is no cross-work-item reduction to
# coordinate. Each chain loops its own guest count directly rather than a padded, capacity-wide
# range (`docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`, "What is genuinely hard: a
# variable particle count"): correctness first, GPU divergence measured before it is optimized away.

# `N`-site guest with every site at the coordinate origin: the "old" or "new" side of a reciprocal
# structure-factor exchange when a guest is being added (no "old" pose exists) or removed (no
# "new" pose exists) — `_reciprocal_move_delta_k` still needs SOME site positions on that side, but
# every charge multiplying them is `guest.charges[s]`, so any actual coordinate would do; the
# origin needs no pose to have been drawn at all.
no_guest_sites(::Type{T}, ::Val{N}) where {T, N} = SVector{N}(ntuple(_ -> zero(SVector{3, T}), N))

"""
    reciprocal_exchange_energy(guest, old_sites, new_sites, ks, kprefactor, Sk) -> ΔU

Reciprocal-space energy change from replacing one guest's already-rotated site positions
`old_sites` with `new_sites` in the running field `Sk` (`_reciprocal_move_delta_k`, `guest.jl`),
without writing `Sk` anywhere: passing `no_guest_sites(T, Val(N))` as `old_sites` gives an
INSERTION's `ΔU` (no guest before, `new_sites` after); passing it as `new_sites` gives a
DELETION's `ΔU` (`old_sites` before, no guest after) — see `guest_guest_move_delta`'s own comment
for why summing this over any partition of a system's k-vector range gives the same total. This
feeds the accept/reject decision; `apply_exchange_sk!` runs the (identical) k-loop again to
actually write `Sk` on acceptance, the same two-pass trade `apply_sk_kernel!` already makes for the
NVT moves rather than carrying every k-vector's `ΔS` through a scratch array while a decision is
still pending.
"""
function reciprocal_exchange_energy(guest::Guest{T, N}, old_sites, new_sites, ks, kprefactor, Sk) where {T, N}
    ΔU = zero(T)
    for i in eachindex(ks)
        _, contribution = _reciprocal_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
        ΔU += kprefactor[i] * contribution
    end
    return T(KE) * ΔU
end

"""
    apply_exchange_sk!(guest, old_sites, new_sites, ks, Sk, sk_abs_accum) -> nothing

Applies an ACCEPTED exchange move's structure-factor change: `Sk[i] += ds` and
`sk_abs_accum[i] += 2*Σ|guest.charges|` (`apply_sk_kernel!`'s own tolerance-scale bound,
`SystemState`'s docstring) for every k-vector, `ds` from the SAME `old_sites`/`new_sites`
convention `reciprocal_exchange_energy` uses (recomputed here rather than carried over from that
call, for the same reason `apply_sk_kernel!` recomputes rather than stores `ds`).
"""
function apply_exchange_sk!(guest::Guest{T, N}, old_sites, new_sites, ks, Sk, sk_abs_accum) where {T, N}
    ds_bound = 2 * sum(abs, guest.charges)
    for i in eachindex(ks)
        ds, _ = _reciprocal_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
        Sk[i] += ds
        sk_abs_accum[i] += ds_bound
    end
    return nothing
end

"""
    insertion_constant_coeffs(ff, batch, guest, n) -> (p, q)

`insertion_constant_term(ff, batch, guest, n, Ng)` (`nvt.jl`), as a function of the system's
occupancy `Ng` alone — everything else it depends on (host/guest type counts, `V`, `α`, net
charge) is fixed for system `n` — is exactly AFFINE in `Ng`: `guest_tail_correction`'s underlying
`tail_delta` is quadratic in the guest count with no constant term
(`8π/(3V) · Σ_ab (2·counts[i]·N·gcounts[j] + N²·gcounts[i]·gcounts[j])·c(a,b)`), so its forward
difference at `Ng+1` vs `Ng` — exactly what `insertion_constant_term` takes — is affine in `Ng`;
the net-charge correction `net_at(m)` is the same quadratic-in-`m` shape, so its own forward
difference is affine too. `p = f(0)`, `q = f(1) - f(0)` therefore determine `f(Ng) = p + q·Ng` for
every `Ng`, letting a μVT move's kernel evaluate the pose-independent term with two scalar
multiplies per attempt instead of recomputing `insertion_constant_term`'s allocating
type-histogram sums (not device-kernel-safe at all) on every one of a chain's insertion/deletion
attempts.
"""
function insertion_constant_coeffs(ff::ForceField{T}, batch::FrameworkBatch{T}, guest::Guest{T, N}, n::Integer) where {T, N}
    p = insertion_constant_term(ff, batch, guest, n, 0)
    q = insertion_constant_term(ff, batch, guest, n, 1) - p
    return p, q
end

"""
    mc_insert_kernel!(refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum,
                      Sk, sk_abs_accum, rng_counter, batch, guest, guest_types, guest_offsets,
                      k_offsets, rng_seed, fugacity, kT, const_p, const_q)

One work-item per chain (`ndrange = nsys`). Draws a fresh uniform pose in the chain's own cell
(`propose_reinsertion`, the same symmetric proposal translation/rotation/reinsertion already
share) from `ChainRNG(rng_seed[n], rng_counter[n], 0)` — the same per-chain, per-attempt keying
`select_and_propose` uses for the NVT moves, so a μVT insertion attempt is just another draw from
the SAME per-chain move-attempt stream, sharing its counter. Computes `ΔU` from the new guest's
host-guest real-space energy (`host_guest_realspace_energy`), its real-space energy against every
LIVE existing guest (`guest_pair_realspace_energy`, `occupancy[n]`-bounded), the reciprocal-space
change against the running field `Sk` (`reciprocal_exchange_energy`) and the pose-independent term
`const_p[n] + const_q[n]*occupancy[n]` (`insertion_constant_coeffs`), then accepts with probability
`min(1, exp(log_insertion_prefactor(fugacity[n], V, kT, occupancy[n]) - ΔU/kT))`
(`metropolis_accept_muvt`). At capacity (`occupancy[n] == capacity(state, n)`) the draw and the
energy evaluation still happen (keeping the RNG stream advance identical to every other chain
regardless of outcome, matching `decide_move_kernel!`'s handling of a guest-less chain for the NVT
moves) but `accept` is forced `false`, so a slot beyond the reserved capacity block is computed but
never written.

On acceptance: writes the new pose and host-guest energy into slot
`guest_offsets[n] + occupancy[n] + 1` (`insert_guest!`'s own placement rule, inlined here since a
kernel may not call a throwing host function), increments `occupancy[n]`, applies the structure-
factor change (`apply_exchange_sk!`) and updates `energy[n]`/`energy_abs_accum[n]`. On rejection,
`refpoints`/`orientations`/`host_energy`/`occupancy`/`Sk`/`energy` are all left untouched.
`rng_counter[n]` advances by one regardless of the outcome.
"""
@kernel function mc_insert_kernel!(
        refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum, Sk, sk_abs_accum, rng_counter,
        batch, guest::Guest{T, N}, guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(k_offsets),
        @Const(rng_seed), @Const(fugacity), kT::T, @Const(const_p), @Const(const_q)
    ) where {T, N}
    n = @index(Global, Linear)
    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    V = batch.volumes[fw]

    gr0 = guest_offsets[n]
    Ng = occupancy[n]
    cap = guest_offsets[n + 1] - gr0
    gr = (gr0 + one(gr0)):(gr0 + Ng)

    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    pos, q, rng = propose_reinsertion(rng, A)
    u, rng = rand_uniform(rng, T)

    ks0 = k_offsets[n] + one(eltype(k_offsets)); ks1 = k_offsets[n + 1]
    kb0 = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kb1 = batch.k_offsets[fw + 1]
    Sk_n = view(Sk, ks0:ks1)
    ks_n = view(batch.ks, kb0:kb1)
    kpref_n = view(batch.kprefactor, kb0:kb1)

    host_new = host_guest_realspace_energy(
        pos, q, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff, batch.positions, batch.types,
        batch.charges, a0, natoms, A, invA, alpha
    )
    test_sites = guest_sites_at(guest, pos, q)
    E_lj = zero(T); E_sr = zero(T)
    for j in gr
        other_sites = guest_sites_at(guest, refpoints[j], orientations[j])
        lj, sr = guest_pair_realspace_energy(
            test_sites, other_sites, guest_types, guest.charges, batch.sigma, batch.epsilon, batch.cutoff,
            batch.ewald_cutoff, A, invA, alpha
        )
        E_lj += lj; E_sr += sr
    end
    zero_sites = no_guest_sites(T, Val(N))
    ΔU_recip = reciprocal_exchange_energy(guest, zero_sites, test_sites, ks_n, kpref_n, Sk_n)
    const_term_n = const_p[n] + const_q[n] * T(Ng)
    ΔU = host_new + E_lj + T(KE) * E_sr + ΔU_recip + const_term_n

    log_pref = log_insertion_prefactor(fugacity[n], V, kT, Ng)
    accept = metropolis_accept_muvt(ΔU, kT, log_pref, u) & (Ng < cap)

    if accept
        apply_exchange_sk!(guest, zero_sites, test_sites, ks_n, Sk_n, view(sk_abs_accum, ks0:ks1))
        slot = gr0 + Ng + one(gr0)
        refpoints[slot] = pos
        orientations[slot] = q
        host_energy[slot] = host_new
        occupancy[n] = Ng + one(Ng)
        energy[n] += ΔU
        energy_abs_accum[n] += abs(ΔU)
    end
    rng_counter[n] = rng_counter[n] + one(eltype(rng_counter))
end

"""
    mc_delete_kernel!(refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum,
                      Sk, sk_abs_accum, rng_counter, batch, guest, guest_types, guest_offsets,
                      k_offsets, rng_seed, fugacity, kT, const_p, const_q)

One work-item per chain (`ndrange = nsys`), the μVT deletion counterpart to `mc_insert_kernel!`.
Selects a guest uniformly among the chain's `occupancy[n]` LIVE guests via the same
`rand_range`-on-`ChainRNG` mechanism `select_and_propose` uses (a zero occupancy clamps to a
fictitious slot, matching `select_and_propose`'s own comment, since `accept` is forced `false`
below in that case regardless). `ΔU` is the exact negative of `mc_insert_kernel!`'s formula for
inserting THIS SAME guest, at its own current pose, back into the `occupancy[n]-1`-guest system its
removal leaves: `host_energy[i]` (P2's cache, no host-atom rescan), the guest-guest real-space
energy against every OTHER live guest, and `reciprocal_exchange_energy` with the sites swapped
(`old_sites = guest i`, `new_sites = none`) compose the same three pieces `mc_insert_kernel!` does,
with the opposite overall sign, so the two can never independently disagree about what "the
guest's interaction with everything else" means. Accepts with probability
`min(1, exp(log_deletion_prefactor(fugacity[n], V, kT, occupancy[n]) - ΔU/kT))`, forced `false`
whenever `occupancy[n]` is zero.

On acceptance: applies the structure-factor change (`apply_exchange_sk!`), copies the system's LAST
occupied slot's pose and cached host energy into the freed slot (`delete_guest!`'s own swap rule, a
no-op when the removed slot already is the last one), decrements `occupancy[n]`, and updates
`energy[n]`/`energy_abs_accum[n]`. On rejection every one of those is left untouched.
`rng_counter[n]` advances by one regardless of the outcome.
"""
@kernel function mc_delete_kernel!(
        refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum, Sk, sk_abs_accum, rng_counter,
        batch, guest::Guest{T, N}, guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(k_offsets),
        @Const(rng_seed), @Const(fugacity), kT::T, @Const(const_p), @Const(const_q)
    ) where {T, N}
    n = @index(Global, Linear)
    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    V = batch.volumes[fw]

    gr0 = guest_offsets[n]
    Ng = occupancy[n]
    Ng_eff = max(Ng, one(Ng))

    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    gidx_local, rng = rand_range(rng, Ng_eff)
    i = clamp(gr0 + gidx_local, one(gr0), Int32(length(refpoints)))
    u, rng = rand_uniform(rng, T)

    gr = (gr0 + one(gr0)):(gr0 + Ng)
    ks0 = k_offsets[n] + one(eltype(k_offsets)); ks1 = k_offsets[n + 1]
    kb0 = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kb1 = batch.k_offsets[fw + 1]
    Sk_n = view(Sk, ks0:ks1)
    ks_n = view(batch.ks, kb0:kb1)
    kpref_n = view(batch.kprefactor, kb0:kb1)

    pos = refpoints[i]; q = orientations[i]
    sites_i = guest_sites_at(guest, pos, q)
    zero_sites = no_guest_sites(T, Val(N))
    E_lj = zero(T); E_sr = zero(T)
    for j in gr
        j == i && continue
        other_sites = guest_sites_at(guest, refpoints[j], orientations[j])
        lj, sr = guest_pair_realspace_energy(
            sites_i, other_sites, guest_types, guest.charges, batch.sigma, batch.epsilon, batch.cutoff,
            batch.ewald_cutoff, A, invA, alpha
        )
        E_lj += lj; E_sr += sr
    end
    ΔU_recip = reciprocal_exchange_energy(guest, sites_i, zero_sites, ks_n, kpref_n, Sk_n)
    const_term_n1 = const_p[n] + const_q[n] * T(Ng - one(Ng))
    ΔU = -(host_energy[i] + E_lj + T(KE) * E_sr + ΔU_recip + const_term_n1)

    log_pref = log_deletion_prefactor(fugacity[n], V, kT, Ng)
    accept = metropolis_accept_muvt(ΔU, kT, log_pref, u) & !iszero(Ng)

    if accept
        apply_exchange_sk!(guest, sites_i, zero_sites, ks_n, Sk_n, view(sk_abs_accum, ks0:ks1))
        hi = gr0 + Ng
        if i != hi
            refpoints[i] = refpoints[hi]
            orientations[i] = orientations[hi]
            host_energy[i] = host_energy[hi]
        end
        occupancy[n] = Ng - one(Ng)
        energy[n] += ΔU
        energy_abs_accum[n] += abs(ΔU)
    end
    rng_counter[n] = rng_counter[n] + one(eltype(rng_counter))
end

"""
    mc_insert!(batch, state, guest, guest_types, ff, fugacity, kT; backend = CPU()) -> nothing
    mc_delete!(batch, state, guest, guest_types, ff, fugacity, kT; backend = CPU()) -> nothing

One μVT insertion (`mc_insert!`) or deletion (`mc_delete!`) attempt for every chain in `state`,
at fixed fugacity `fugacity[n]` per system (`src/fugacity.jl`, computed by the caller once per
system per run; same units as `kT / V`, i.e. energy per volume, so `fugacity[n]*V/kT` is
dimensionless — this function does no unit conversion of its own). `ws`-free, unlike `mc_step!`:
each chain's attempt is one work-item with no workgroup fan-out (`mc_insert_kernel!`/
`mc_delete_kernel!`'s own docstrings). `batch` and `state` must already be resident on `backend`.
`guest` must already be reindexed to `batch`'s compact type (`compact_guest`). `ff` is the force
field `batch` was built from, used only to derive the pose-independent term's affine coefficients
(`insertion_constant_coeffs`) — a cheap, allocating, host-side computation done once per call
rather than per chain, since it does not depend on which chain's occupancy is being evaluated.
`kT` is Boltzmann's constant times the temperature, in the same energy units as `batch`/`guest`.
Exceeding a chain's capacity, or attempting a deletion on an empty chain, is handled by forcing
that chain's acceptance to `false` (`mc_insert_kernel!`/`mc_delete_kernel!`'s own docstrings), not
by throwing: only `insert_guest!`/`delete_guest!` (`state.jl`, not called by this device path) do
that, since a kernel may not throw.
"""
function mc_insert!(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        ff::ForceField{T}, fugacity, kT::T; backend = CPU()
    ) where {T, N}
    nsys = state.nsys
    p = T[insertion_constant_coeffs(ff, batch, guest, n)[1] for n in 1:nsys]
    q = T[insertion_constant_coeffs(ff, batch, guest, n)[2] for n in 1:nsys]
    dp = adapt(backend, p); dq = adapt(backend, q)
    dfug = adapt(backend, T.(fugacity))
    mc_insert_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.occupancy, state.energy, state.energy_abs_accum,
        state.Sk, state.sk_abs_accum, state.rng_counter, batch, guest, guest_types, state.guest_offsets,
        state.k_offsets, state.rng_seed, dfug, kT, dp, dq; ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function mc_delete!(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        ff::ForceField{T}, fugacity, kT::T; backend = CPU()
    ) where {T, N}
    nsys = state.nsys
    p = T[insertion_constant_coeffs(ff, batch, guest, n)[1] for n in 1:nsys]
    q = T[insertion_constant_coeffs(ff, batch, guest, n)[2] for n in 1:nsys]
    dp = adapt(backend, p); dq = adapt(backend, q)
    dfug = adapt(backend, T.(fugacity))
    mc_delete_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.occupancy, state.energy, state.energy_abs_accum,
        state.Sk, state.sk_abs_accum, state.rng_counter, batch, guest, guest_types, state.guest_offsets,
        state.k_offsets, state.rng_seed, dfug, kT, dp, dq; ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
