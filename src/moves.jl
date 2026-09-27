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
`A_ins = min(1, exp(log_insertion_prefactor(f, V, kT, N) - ΔU/kT))`. `f` is fugacity in eV·Å⁻³ (the
same units as `kT/V`), NOT the pascals `peng_robinson_fugacity` returns — `mc_insert!`/`mc_delete!`
are the only places that cross a caller's pascal-valued fugacity into this eV·Å⁻³ convention
(`PASCAL`, `src/constants.jl`), so this function and `log_deletion_prefactor` do no unit
conversion of their own.
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
reached here by construction rather than by re-deriving it. `f` is in eV·Å⁻³, exactly as
`log_insertion_prefactor` documents.
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

Device-memory scratch `mc_step!` (and the μVT exchange moves, `mc_insert!`/`mc_delete!`) reuse
across calls so a move attempt allocates nothing: `partial1`/`partial2` hold each chain's
up-to-`nblocks_max` per-workgroup partial sums from `evaluate_move_kernel!`/
`evaluate_insert_kernel!`/`evaluate_delete_kernel!` (the matching `decide_*_kernel!`'s input),
shaped `(nblocks_max, nsys)` so any call may use `nblocks_per_chain <= nblocks_max`; `accept_flag`,
`move_oldpos`, `move_oldq`, `move_newpos`, `move_newq` are a `decide_*_kernel!`'s per-chain output
and the matching apply kernel's input — this handoff is what lets an evaluate kernel's several
workgroups per chain communicate with each other at all, since only work-items within the SAME
workgroup share `@localmem`. An NVT move (`decide_move_kernel!`) uses all four pose fields, one
old/new pair for the single guest that moved; a μVT exchange move (`decide_insert_kernel!`/
`decide_delete_kernel!`) has only one real pose — the newly inserted guest, or the guest about to
be removed — and writes it into `move_newpos`/`move_newq` alone, leaving `move_oldpos`/`move_oldq`
unused (`apply_exchange_sk_kernel!` never reads them). `capacity_hits` is `mc_insert_kernel!`'s
own per-chain diagnostic, read back and checked host-side by `mc_insert!`. Build with
`MoveWorkspace(T, nsys, nblocks_max; backend)` and adapt it to `backend` once, like `batch`/
`state`, rather than per call.
"""
struct MoveWorkspace{VF, VU8, VP, VQ}
    partial1::VF
    partial2::VF
    accept_flag::VU8
    capacity_hits::VU8
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
# system's OCCUPANCY, not just a guest's pose. `mc_insert!`/`mc_delete!` use the SAME
# evaluate/decide/apply three-kernel structure `mc_step!` does, and share its machinery directly
# rather than duplicating it: `evaluate_insert_kernel!`/`evaluate_delete_kernel!` fan a chain's
# host-atom loop (insertion only — deletion reads `host_energy`'s cache instead), guest-guest loop
# and k-vector loop across `nblocks_per_chain` workgroups of `groupsize` work-items exactly as
# `evaluate_move_kernel!` does, calling the IDENTICAL `host_guest_realspace_energy_range` and
# `reciprocal_exchange_energy` (over a strided view, the same trick `evaluate_move_kernel!` plays
# on `reciprocal_move_delta_energy`) that the NVT path calls. The one piece that could not be
# reused as-is is the guest-guest loop: `guest_guest_move_delta` sums an old/new POSE-PAIR delta for
# a guest that already exists and is moving, whereas an exchange move's guest has only ONE real pose
# (a candidate insertion, or a guest about to be removed) — `guest_pair_realspace_energy_range`
# (`guest.jl`) is that single-pose analogue, over the same kind of strided range.
#
# `decide_insert_kernel!`/`decide_delete_kernel!` stay one-work-item-per-chain, exactly as
# `decide_move_kernel!` is: summing `nblocks_per_chain` partial sums and committing the outcome is
# O(1) work, and a chain's own occupancy (writing a new slot, or swap-deleting one) is only ever
# touched by that chain's own work-item. `apply_exchange_sk_kernel!` fans the k-vector write for an
# accepted move across `nblocks_per_chain*groupsize` work-items with no reduction needed — the same
# disjoint-k-range parallelism `apply_sk_kernel!` uses — taking `Val(INSERT)` to pick which side of
# `reciprocal_exchange_energy`'s own `old_sites`/`new_sites` convention is the real pose and which
# is `no_guest_sites`'s sentinel, since an exchange move supplies only one real pose where
# `apply_sk_kernel!` has an old-and-new pair for a single guest that moved.
#
# Each chain still loops its own guest count directly rather than a padded, capacity-wide range
# (`docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`, "What is genuinely hard: a variable
# particle count"), exactly as the NVT moves do.
#
# Measured on an RTX 4070 (RUBTAK 3×3×3 + CO2, `bench/gpu/exchange_workgroup_bench.jl`,
# `bench/results/pureadsorb_exchange_workgroup_*.json`): `mc_insert!`/`mc_delete!` drop from
# 15.5/10.4 ms/call to 189/134 us/call at nsys=1 (Float64) and 3.29/1.87 ms to 159/105 us/call
# (Float32) — an 82×/78× (Float64) and 21×/18× (Float32) reduction, both now at or under
# `mc_step!`'s own ~187-230 us/move. At nsys=64 and 256 the per-call cost (every chain's own
# attempt, one launch) rises with nsys exactly as `mc_step!`'s does, since more chains' work is
# being done per call, not because the fan-out degrades: 822/485 us (nsys=64) and 2.52/1.62 ms
# (nsys=256) in Float64, 205/124 us and 313/172 us in Float32.
#
# This file's own opening comment argues that translation, rotation and reinsertion each satisfy
# detailed balance ON THEIR OWN, so applying any fixed (even non-random) sequence of them still
# preserves the target distribution. That argument does NOT extend to insertion and deletion:
# neither move has an inverse of the SAME type (an insertion cannot be undone by another
# insertion), so `K_ins` alone and `K_del` alone each fail detailed balance, and only the
# equal-probability mixture `½K_ins + ½K_del` is π-preserving. A fixed alternation (or any other
# deterministic schedule) between `mc_insert!`/`mc_delete!` samples the WRONG loading while every
# existing energy audit still passes, since the audit checks energies, not the acceptance ratio's
# statistics. `mc_exchange!`, below, is the only sanctioned way to call these two moves: it draws a
# fresh, fair coin once per launch (shared across every chain in the batch, exactly as `movetype`
# is shared across a batch in `mc_step!`/`run_nvt!`) and both `log_insertion_prefactor` and
# `log_deletion_prefactor` are derived assuming `p_ins = p_del = 1/2` holds for that draw.

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
    exchange_constant_term(ff, batch, guest, n, Ng) -> value

The pose-independent part of a μVT exchange move's `ΔU` for system `n`, currently holding `Ng`
guests: the intramolecular self-energy and exclusion correction `KE*(gself+gexcl)`
(`guest_self_terms`, exact and Ng-independent — this is the cost of the guest's OWN geometry
existing at all, not the reciprocal cross/self term below), plus the tail-correction and
net-charge-correction CHANGES from adding one more guest (`guest_tail_correction(Ng+1) -
guest_tail_correction(Ng)`, `net_at(Ng+1) - net_at(Ng)`).

This is NOT `insertion_constant_term` (`nvt.jl`), and must not reuse it: `insertion_constant_term`
is built on `FrameworkBatch.constant_offset`, which folds in `self_mean` — the ORIENTATION-AVERAGED
reciprocal self term `KE Σ_k pref_k |S_guest(k)|²` (`FrameworkBatch`'s docstring) that Milestone A's
`insertion_energy`/`widom_chain_kernel!` need because THEIR reciprocal term is cross-only
(`2 Re(Shost*·Sg)`, `insertion_energy`'s own comment) and never computes `|S_guest|²` at all.
`mc_insert_kernel!`/`mc_delete_kernel!` instead call `reciprocal_exchange_energy`, which computes
the EXACT `|Sk+ΔS|² − |Sk|²` (`_reciprocal_move_delta_k`) — this already contains the guest's real,
un-averaged `|ΔS|²` self term at its actual accepted orientation. Adding `self_mean` on top of that
would double-count the reciprocal self term by exactly `self_mean` per accepted insertion — a
discrepancy `test/exchange_tests.jl`'s reversibility and long-chain-audit tests are sized to catch
against a from-scratch `total_energy` recomputation.

`exchange_constant_coeffs`, below, exploits that this is exactly AFFINE in `Ng`: `tail_delta`
(underlying `guest_tail_correction`) is quadratic in the guest count with no constant term
(`8π/(3V) · Σ_ab (2·counts[i]·N·gcounts[j] + N²·gcounts[i]·gcounts[j])·c(a,b)`), so its forward
difference is affine in `Ng`; `net_at(m)` is the same quadratic-in-`m` shape, so its forward
difference is affine too; and `KE*(gself+gexcl)` does not depend on `Ng` at all.
"""
function exchange_constant_term(ff::ForceField{T}, batch::FrameworkBatch{T}, guest::Guest{T, N}, n::Integer, Ng::Integer) where {T, N}
    fw = batch.framework_of[n]
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
    gself, gexcl = guest_self_terms(guest, alpha, batch.ewald_cutoff)
    return T(KE) * (gself + gexcl) +
        guest_tail_correction(ff, host_counts, gcounts, Ng + 1, V) - guest_tail_correction(ff, host_counts, gcounts, Ng, V) +
        net_at(Ng + 1) - net_at(Ng)
end

"""
    exchange_constant_coeffs(ff, batch, guest, n) -> (p, q)

`exchange_constant_term(ff, batch, guest, n, Ng)`, as a function of `Ng` alone, is exactly affine
(see its own docstring): `p = f(0)`, `q = f(1) - f(0)` determine `f(Ng) = p + q·Ng` for every `Ng`,
letting a μVT move's kernel evaluate the pose-independent term with two scalar multiplies per
attempt instead of recomputing `exchange_constant_term`'s allocating type-histogram sums (not
device-kernel-safe at all) on every one of a chain's insertion/deletion attempts.
"""
function exchange_constant_coeffs(ff::ForceField{T}, batch::FrameworkBatch{T}, guest::Guest{T, N}, n::Integer) where {T, N}
    p = exchange_constant_term(ff, batch, guest, n, 0)
    q = exchange_constant_term(ff, batch, guest, n, 1) - p
    return p, q
end

"""
    exchange_constant_coeffs(ff, batch, guest) -> (p::Vector{T}, q::Vector{T})

`exchange_constant_coeffs(ff, batch, guest, n)` for every one of `batch.nsys` systems, from a
HOST-resident `batch` (this indexes `batch.charges`/`batch.types` etc. by scalar, which a
device-resident array does not support). `batch` and `guest` are fixed for an entire run, so
`mc_insert!`/`mc_delete!`/`mc_exchange!` take the result as a precomputed pair rather than
deriving it internally on every call -- see their own docstrings for why that distinction
matters when `batch` is device-resident.
"""
function exchange_constant_coeffs(ff::ForceField{T}, batch::FrameworkBatch{T}, guest::Guest{T, N}) where {T, N}
    nsys = batch.nsys
    p = Vector{T}(undef, nsys)
    q = Vector{T}(undef, nsys)
    for n in 1:nsys
        p[n], q[n] = exchange_constant_coeffs(ff, batch, guest, n)
    end
    return p, q
end

# Deterministically reproduces chain `n`'s to-be-deleted guest index from `(rng_seed[n],
# rng_counter[n])` alone, exactly as `select_and_propose` does for the NVT moves: every work-item
# of every `evaluate_delete_kernel!` workgroup assigned to chain `n` calls this identically, and
# `decide_delete_kernel!` calls it again in its own, separate kernel launch to know which slot to
# possibly remove. Returns the ADVANCED `rng` state too (`select_and_propose`'s own convention), so
# `decide_delete_kernel!` can continue drawing its acceptance uniform from the same stream without
# reconstructing it. A zero-occupancy chain (`Ng` zero) has no valid slot to select; `gidx_local`
# clamps into `refpoints`'s valid range regardless (`rand_range` never returns below `1`), and
# `decide_delete_kernel!` forces `accept = false` whenever `Ng` is zero, so nothing selected for
# such a chain is ever actually removed.
@inline function select_delete_index(n::Integer, guest_offsets, occupancy, rng_seed, rng_counter, nrefpoints::Integer)
    gr0 = guest_offsets[n]
    Ng = occupancy[n]
    Ng_eff = max(Ng, one(Ng))
    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    gidx_local, rng = rand_range(rng, Ng_eff)
    i = clamp(gr0 + gidx_local, one(gr0), Int32(nrefpoints))
    return i, Ng, rng
end

"""
    evaluate_insert_kernel!(partial1, partial2, refpoints, orientations, Sk, batch, guest,
                            guest_types, guest_offsets, occupancy, k_offsets, rng_seed, rng_counter,
                            nblocks_per_chain, ::Val{G})

`evaluate_move_kernel!`'s own structure, for a candidate insertion: `nblocks_per_chain`
workgroups of `G` work-items evaluate one chain's candidate pose (`ndrange = (G, nblocks_per_chain,
nsys)`). Every work-item in every workgroup assigned to chain `n` independently redraws the SAME
deterministic candidate pose (`propose_reinsertion` from `ChainRNG(rng_seed[n], rng_counter[n], 0)`
— the same per-chain, per-attempt keying `select_and_propose` uses for the NVT moves, so an
insertion attempt is just another draw from the same per-chain move-attempt stream). Each work-item
then takes its own disjoint, stride-`L` slice (`L = nblocks_per_chain*G`) of the host-atom loop
(`host_guest_realspace_energy_range`), the guest-guest loop against every LIVE existing guest
(`guest_pair_realspace_energy_range`, `exclude = 0` since the candidate is not itself stored) and
the k-vector loop against the running field `Sk` (`reciprocal_exchange_energy`, over a strided
view — the same trick `evaluate_move_kernel!` plays on `reciprocal_move_delta_energy`). `buf1`
accumulates ONLY the host-guest term (`host_guest_realspace_energy_range` already returns it KE-
scaled where needed), matching `host_energy[i]`'s per-guest-cache convention so
`decide_insert_kernel!` can write `partial1`'s sum straight into `host_energy` on acceptance;
`buf2` accumulates the guest-guest and reciprocal terms together, which never feed the cache.
`decide_insert_kernel!` sums across blocks.
"""
@kernel function evaluate_insert_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(occupancy), @Const(k_offsets), @Const(rng_seed),
        @Const(rng_counter), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    pos, q, rng = propose_reinsertion(rng, A)

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    gr0 = guest_offsets[n]
    Ng = occupancy[n]
    a0 = batch.atom_offsets[fw]
    natoms = batch.atom_offsets[fw + 1] - a0
    host_new_partial = host_guest_realspace_energy_range(
        pos, q, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, (a0 + lane):L:(a0 + natoms), A, invA, alpha
    )

    gr_lo = gr0 + one(gr0); gr_hi = gr0 + Ng
    E_lj_partial, E_sr_partial = guest_pair_realspace_energy_range(
        guest, pos, q, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, zero(gr0),
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
    kr_lo_b = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kr_hi_b = batch.k_offsets[fw + 1]
    zero_sites = no_guest_sites(T, Val(N))
    test_sites = guest_sites_at(guest, pos, q)
    ΔU_recip_partial = reciprocal_exchange_energy(
        guest, zero_sites, test_sites, view(batch.ks, (kr_lo_b + lane - 1):L:kr_hi_b),
        view(batch.kprefactor, (kr_lo_b + lane - 1):L:kr_hi_b), view(Sk, (kr_lo + lane - 1):L:kr_hi)
    )

    buf1 = @localmem T (G,)
    buf2 = @localmem T (G,)
    buf1[tid] = host_new_partial
    buf2[tid] = E_lj_partial + T(KE) * E_sr_partial + ΔU_recip_partial
    @synchronize()
    for step in 1:nsteps
        stride = G >> step
        if tid <= stride
            buf1[tid] += buf1[tid + stride]
            buf2[tid] += buf2[tid + stride]
        end
        @synchronize()
    end
    grp2 = @index(Group, NTuple)
    b2, n2 = grp2[2], grp2[3]
    tid == 1 && (partial1[b2, n2] = buf1[1]; partial2[b2, n2] = buf2[1])
end

"""
    decide_insert_kernel!(refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum,
                          rng_counter, capacity_hits, accept_flag, move_newpos, move_newq, partial1,
                          partial2, batch, guest, guest_offsets, rng_seed, fugacity, kT, const_p,
                          const_q, nblocks_per_chain)

One work-item per chain (`ndrange = nsys`), `evaluate_insert_kernel!`'s decide stage. Redraws the
SAME candidate pose (cheap: a handful of RNG draws and flops, unlike `evaluate_insert_kernel!`'s
fanned-out scan), sums `nblocks_per_chain` partial sums into `ΔU` together with the pose-independent
term `const_p[n] + const_q[n]*occupancy[n]` (`exchange_constant_coeffs`), and applies
`metropolis_accept_muvt` with the acceptance ratio's insertion prefactor. At capacity
(`occupancy[n] == capacity(state, n)`) the draw and the energy evaluation still happen (keeping the
RNG stream advance identical to every other chain regardless of outcome, matching
`decide_move_kernel!`'s handling of a guest-less chain for the NVT moves) but `accept` is forced
`false`, so a slot beyond the reserved capacity block is computed but never written.

On acceptance: writes the new pose and its host-guest energy (`partial1`'s summed value, exactly
the cached-energy convention `decide_move_kernel!`'s own `host_energy[i] = e_new` uses) into slot
`guest_offsets[n] + occupancy[n] + 1` (`insert_guest!`'s own placement rule, inlined here since a
kernel may not call a throwing host function), increments `occupancy[n]`, and updates
`energy[n]`/`energy_abs_accum[n]`. On rejection, `refpoints`/`orientations`/`host_energy`/
`occupancy`/`energy` are all left untouched; `Sk` is never touched here at all —
`apply_exchange_sk_kernel!` applies an accepted move's structure-factor change separately, reading
`accept_flag`/`move_newpos`/`move_newq` written below. `rng_counter[n]` advances by one regardless
of the outcome.

`capacity_hits[n]` is set to `1` whenever the Metropolis test alone would have accepted (a genuine,
physical insertion) but the chain sat at capacity, so the move was forced to reject anyway — the
actual truncation event `mc_insert!` checks for and throws on — and to `0` otherwise (including an
ordinary physical rejection at capacity, which truncates nothing since the untruncated chain would
have rejected that attempt too). Without this signal, a kernel that merely forces `accept = false`
at capacity makes `audit_energy!`'s `occupancy <= capacity` check vacuous: nothing written by this
kernel can ever exceed capacity, so that check alone could never catch a saturating chain.
"""
@kernel function decide_insert_kernel!(
        refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum, rng_counter, capacity_hits,
        accept_flag, move_newpos, move_newq, @Const(partial1), @Const(partial2), batch, guest::Guest{T, N},
        @Const(guest_offsets), @Const(rng_seed), @Const(fugacity), kT::T, @Const(const_p), @Const(const_q),
        nblocks_per_chain::Int32
    ) where {T, N}
    n = @index(Global, Linear)
    fw = batch.framework_of[n]
    A = batch.cells[fw]
    V = batch.volumes[fw]
    gr0 = guest_offsets[n]
    Ng = occupancy[n]
    cap = guest_offsets[n + 1] - gr0

    rng = ChainRNG(rng_seed[n], rng_counter[n], zero(Int32))
    pos, q, rng = propose_reinsertion(rng, A)
    u, rng = rand_uniform(rng, T)

    e_new = zero(T); rest = zero(T)
    for blk in Int32(1):nblocks_per_chain
        e_new += partial1[blk, n]
        rest += partial2[blk, n]
    end
    const_term_n = const_p[n] + const_q[n] * T(Ng)
    ΔU = e_new + rest + const_term_n

    log_pref = log_insertion_prefactor(fugacity[n], V, kT, Ng)
    would_accept = metropolis_accept_muvt(ΔU, kT, log_pref, u)
    at_capacity = !(Ng < cap)
    accept = would_accept & !at_capacity
    capacity_hits[n] = UInt8(would_accept & at_capacity)

    if accept
        slot = gr0 + Ng + one(gr0)
        refpoints[slot] = pos
        orientations[slot] = q
        host_energy[slot] = e_new
        occupancy[n] = Ng + one(Ng)
        energy[n] += ΔU
        energy_abs_accum[n] += abs(ΔU)
    end
    accept_flag[n] = UInt8(accept)
    move_newpos[n] = pos; move_newq[n] = q
    rng_counter[n] = rng_counter[n] + one(eltype(rng_counter))
end

"""
    evaluate_delete_kernel!(partial1, partial2, refpoints, orientations, Sk, batch, guest,
                            guest_types, guest_offsets, occupancy, k_offsets, rng_seed, rng_counter,
                            nblocks_per_chain, ::Val{G})

`evaluate_insert_kernel!`'s deletion counterpart: no host-atom loop (deletion reads
`host_energy[i]`'s cache in `decide_delete_kernel!` instead, an O(1) lookup that needs no fan-out),
just the guest-guest loop against every OTHER live guest (`guest_pair_realspace_energy_range`,
`exclude = i`) and the k-vector loop (`reciprocal_exchange_energy`, `old_sites = sites_i`,
`new_sites = no_guest_sites`), fanned across `nblocks_per_chain` workgroups of `G` work-items
exactly as `evaluate_insert_kernel!` fans its own three loops. `select_delete_index` picks the
SAME guest index every work-item of every workgroup assigned to chain `n` would pick, since it is
keyed only on `(rng_seed[n], rng_counter[n])`. `buf1` accumulates the guest-guest term alone (the
piece `decide_delete_kernel!` negates together with the cached `host_energy[i]`); `buf2`
accumulates the reciprocal term, which is added rather than negated (`decide_delete_kernel!`'s own
docstring explains why).
"""
@kernel function evaluate_delete_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(occupancy), @Const(k_offsets), @Const(rng_seed),
        @Const(rng_counter), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    fw = batch.framework_of[n]
    i, Ng, _ = select_delete_index(n, guest_offsets, occupancy, rng_seed, rng_counter, length(refpoints))
    pos = refpoints[i]; q = orientations[i]

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    gr0 = guest_offsets[n]
    gr_lo = gr0 + one(gr0); gr_hi = gr0 + Ng
    E_lj_partial, E_sr_partial = guest_pair_realspace_energy_range(
        guest, pos, q, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, i,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
    kr_lo_b = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kr_hi_b = batch.k_offsets[fw + 1]
    zero_sites = no_guest_sites(T, Val(N))
    sites_i = guest_sites_at(guest, pos, q)
    ΔU_recip_partial = reciprocal_exchange_energy(
        guest, sites_i, zero_sites, view(batch.ks, (kr_lo_b + lane - 1):L:kr_hi_b),
        view(batch.kprefactor, (kr_lo_b + lane - 1):L:kr_hi_b), view(Sk, (kr_lo + lane - 1):L:kr_hi)
    )

    buf1 = @localmem T (G,)
    buf2 = @localmem T (G,)
    buf1[tid] = E_lj_partial + T(KE) * E_sr_partial
    buf2[tid] = ΔU_recip_partial
    @synchronize()
    for step in 1:nsteps
        stride = G >> step
        if tid <= stride
            buf1[tid] += buf1[tid + stride]
            buf2[tid] += buf2[tid + stride]
        end
        @synchronize()
    end
    grp2 = @index(Group, NTuple)
    b2, n2 = grp2[2], grp2[3]
    tid == 1 && (partial1[b2, n2] = buf1[1]; partial2[b2, n2] = buf2[1])
end

"""
    decide_delete_kernel!(refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum,
                          rng_counter, accept_flag, move_newpos, move_newq, partial1, partial2,
                          batch, guest, guest_offsets, rng_seed, fugacity, kT, const_p, const_q,
                          nblocks_per_chain)

One work-item per chain (`ndrange = nsys`), `evaluate_delete_kernel!`'s decide stage. Reselects the
SAME guest index (`select_delete_index`) and sums `nblocks_per_chain` partial sums into `ΔU`
together with the cached `host_energy[i]` (P2's cache, no host-atom rescan) and the pose-independent
term `const_p[n] + const_q[n]*(occupancy[n]-1)`:

    ΔU = -(host_energy[i] + guest_guest_term + const_term_n1) + reciprocal_term

`host_energy[i]`, the guest-guest term (`partial1`'s sum) and `const_term_n1` are each the SAME
positive-convention quantity `decide_insert_kernel!` would compute for this same guest at this same
pose, so removing the guest subtracts them (one shared negation). The reciprocal term (`partial2`'s
sum, `reciprocal_exchange_energy` with `old_sites = sites_i`, `new_sites = none`, against the
CURRENT `Sk`, which still includes this guest) already returns `-1` times the value the matching
insertion would return against the field with this guest excluded — the reciprocal energy is
quadratic in the total structure factor, not a per-guest additive term, so its delta is correctly
signed for THIS removal on its own and must be added, not folded into the shared negation; folding
it in would silently double-negate it, exactly what the reversibility test in
`test/exchange_tests.jl` checks for. Accepts with `metropolis_accept_muvt` and the deletion
prefactor, explicitly forced `false` whenever `occupancy[n]` is zero (R7) rather than relying on the
`-Inf` `log_deletion_prefactor(..., 0)` would otherwise produce — the prefactor itself is evaluated
at `max(occupancy[n], 1)` so `log(0)` is never actually computed.

On acceptance: copies the system's LAST occupied slot's pose and cached host energy into the freed
slot (`delete_guest!`'s own swap rule, a no-op when the removed slot already is the last one),
decrements `occupancy[n]`, and updates `energy[n]`/`energy_abs_accum[n]`. `Sk` is never touched
here; `move_newpos[n]`/`move_newq[n]` capture the REMOVED guest's pose BEFORE any swap, so
`apply_exchange_sk_kernel!` (`Val(false)`) reads the guest that actually left, regardless of
whether its slot was then overwritten. On rejection every mutated field above is left untouched.
`rng_counter[n]` advances by one regardless of the outcome.
"""
@kernel function decide_delete_kernel!(
        refpoints, orientations, host_energy, occupancy, energy, energy_abs_accum, rng_counter,
        accept_flag, move_newpos, move_newq, @Const(partial1), @Const(partial2), batch, guest::Guest{T, N},
        @Const(guest_offsets), @Const(rng_seed), @Const(fugacity), kT::T, @Const(const_p), @Const(const_q),
        nblocks_per_chain::Int32
    ) where {T, N}
    n = @index(Global, Linear)
    V = batch.volumes[batch.framework_of[n]]
    Ng = occupancy[n]
    Ng_eff = max(Ng, one(Ng))
    i, _, rng = select_delete_index(n, guest_offsets, occupancy, rng_seed, rng_counter, length(refpoints))
    u, rng = rand_uniform(rng, T)

    gg_term = zero(T); recip_term = zero(T)
    for blk in Int32(1):nblocks_per_chain
        gg_term += partial1[blk, n]
        recip_term += partial2[blk, n]
    end
    const_term_n1 = const_p[n] + const_q[n] * T(Ng - one(Ng))
    ΔU = -(host_energy[i] + gg_term + const_term_n1) + recip_term

    log_pref = log_deletion_prefactor(fugacity[n], V, kT, Ng_eff)
    accept = metropolis_accept_muvt(ΔU, kT, log_pref, u) & !iszero(Ng)

    gr0 = guest_offsets[n]
    pos = refpoints[i]; q = orientations[i]
    if accept
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
    accept_flag[n] = UInt8(accept)
    move_newpos[n] = pos; move_newq[n] = q
    rng_counter[n] = rng_counter[n] + one(eltype(rng_counter))
end

"""
    apply_exchange_sk_kernel!(Sk, sk_abs_accum, batch, accept_flag, pose_pos, pose_q, guest,
                              k_offsets, ::Val{INSERT})

Applies an accepted μVT exchange move's structure-factor change, fanned across `ndrange = (L,
nsys)` work-items with no workgroup reduction needed — the same disjoint-k-range parallelism
`apply_sk_kernel!` uses for the NVT moves, since each k-vector's `Sk` update is independent of
every other. `INSERT = true` (an accepted `decide_insert_kernel!`) treats `pose_pos[n]`/
`pose_q[n]` as the newly inserted guest's pose (`old_sites = no_guest_sites`, `new_sites` the real
pose); `INSERT = false` (an accepted `decide_delete_kernel!`) treats them as the REMOVED guest's
pose before removal (`old_sites` the real pose, `new_sites = no_guest_sites`) — the same
`reciprocal_exchange_energy` convention `evaluate_insert_kernel!`/`evaluate_delete_kernel!` already
use to score the move, fanned across work-items instead of read by one. An exchange move has only
ONE real pose to give this kernel, unlike `apply_sk_kernel!`'s old-and-new pair for a single guest
that moved, so only `move_newpos`/`move_newq` (`MoveWorkspace`'s docstring) are read here.
A rejected chain's `accept_flag` is `0` and this kernel does nothing for it.
"""
@kernel function apply_exchange_sk_kernel!(
        Sk, sk_abs_accum, batch, @Const(accept_flag), @Const(pose_pos), @Const(pose_q), guest::Guest{T, N},
        @Const(k_offsets), ::Val{INSERT}
    ) where {T, N, INSERT}
    idx = @index(Global, NTuple)
    lane, n = idx[1], idx[2]
    L = @ndrange()[1]
    if !iszero(accept_flag[n])
        sites = guest_sites_at(guest, pose_pos[n], pose_q[n])
        zero_sites = no_guest_sites(T, Val(N))
        old_sites = INSERT ? zero_sites : sites
        new_sites = INSERT ? sites : zero_sites
        ds_bound = 2 * sum(abs, guest.charges)
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
    mc_insert!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity, kT;
              backend = CPU(), groupsize = DEFAULT_GROUPSIZE,
              nblocks_per_chain = default_nblocks_per_chain(T, state.nsys)) -> nothing
    mc_delete!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity, kT;
              backend = CPU(), groupsize = DEFAULT_GROUPSIZE,
              nblocks_per_chain = default_nblocks_per_chain(T, state.nsys)) -> nothing

One μVT insertion (`mc_insert!`) or deletion (`mc_delete!`) attempt for every chain in `state`, at
fixed fugacity `fugacity[n]` per system in PASCALS — the units `peng_robinson_fugacity` returns.
Both functions convert to the eV·Å⁻³ convention `log_insertion_prefactor`/`log_deletion_prefactor`
need (`PASCAL`, `src/constants.jl`) exactly once, here, at this entry point; neither the kernels nor
the prefactor functions do any unit conversion of their own.

Three kernel launches each, `mc_step!`'s own structure: an evaluate kernel fans a chain's
host-atom (insertion only), guest-guest and k-vector loops across `nblocks_per_chain` workgroups of
`groupsize` work-items; a decide kernel (one work-item per chain) sums the partial sums and applies
`metropolis_accept_muvt`; `apply_exchange_sk_kernel!` fans back out to update `state.Sk` on an
accepted chain. `ws` (`MoveWorkspace`), `batch` and `state` must already be resident on `backend`,
matching `mc_step!`'s own contract; `groupsize` must be a power of two and `nblocks_per_chain` must
not exceed `ws.nblocks_max`, exactly as `mc_step!` requires. `const_p`/`const_q` are the
pose-independent term's affine coefficients (`exchange_constant_coeffs(ff, host_batch, guest)`, one
entry per system), already resident on `backend` — the CALLER computes these ONCE per run from a
host-resident `batch` (a scalar-indexing computation `FrameworkBatch`'s device storage cannot
support) and adapts them once, exactly as `run_nvt!` precomputes `insertion_constant_term` before
its cycle loop: `batch` and `guest` do not change between calls, so re-deriving `const_p`/`const_q`
on every attempt would force a host round-trip of a device-resident `batch` every time. `guest`
must already be reindexed to `batch`'s compact type (`compact_guest`). `kT` is Boltzmann's constant
times the temperature, in the same energy units as `batch`/`guest`.

Attempting a deletion on an empty chain is handled by forcing that chain's acceptance to `false`
(`decide_delete_kernel!`'s own docstring), not by throwing: a kernel may not throw, so only
`insert_guest!`/`delete_guest!` (`state.jl`, not called by this device path) do that. Exceeding a
chain's capacity is different: `mc_insert!` throws `ArgumentError` immediately if
`ws.capacity_hits` (written fresh by every `decide_insert_kernel!` launch) reports any system where
a physically-accepted insertion was forced to reject for lack of a free slot — a saturating chain
samples a silently truncated distribution, so this is fail-fast rather than a diagnostic left for a
later audit to notice.
"""
function mc_insert!(
        ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity, kT::T;
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = default_nblocks_per_chain(T, state.nsys)
    ) where {T, N}
    ispow2(groupsize) || throw(ArgumentError("mc_insert!: groupsize=$groupsize must be a power of two"))
    nblocks_per_chain >= 1 || throw(ArgumentError("mc_insert!: nblocks_per_chain=$nblocks_per_chain must be >= 1"))
    nblocks_per_chain <= ws.nblocks_max ||
        throw(ArgumentError("mc_insert!: nblocks_per_chain=$nblocks_per_chain exceeds the workspace's nblocks_max=$(ws.nblocks_max)"))
    G = Int(groupsize)
    nbpc = Int32(nblocks_per_chain)
    nsys = state.nsys
    dfug = adapt(backend, T(PASCAL) .* T.(fugacity))

    evaluate_insert_kernel!(backend)(
        ws.partial1, ws.partial2, state.refpoints, state.orientations, state.Sk, batch, guest, guest_types,
        state.guest_offsets, state.occupancy, state.k_offsets, state.rng_seed, state.rng_counter, nbpc, Val(G);
        ndrange = (G, nblocks_per_chain, nsys), workgroupsize = (G, 1, 1)
    )
    KernelAbstractions.synchronize(backend)

    decide_insert_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.occupancy, state.energy, state.energy_abs_accum,
        state.rng_counter, ws.capacity_hits, ws.accept_flag, ws.move_newpos, ws.move_newq, ws.partial1, ws.partial2,
        batch, guest, state.guest_offsets, state.rng_seed, dfug, kT, const_p, const_q, nbpc; ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)

    apply_exchange_sk_kernel!(backend)(
        state.Sk, state.sk_abs_accum, batch, ws.accept_flag, ws.move_newpos, ws.move_newq, guest, state.k_offsets, Val(true);
        ndrange = (nblocks_per_chain * G, nsys)
    )
    KernelAbstractions.synchronize(backend)

    hits = Array(ws.capacity_hits)
    any(!iszero, hits) && throw(
        ArgumentError(
            "mc_insert!: system(s) $(findall(!iszero, hits)) hit capacity on a move the Metropolis " *
                "test would otherwise have accepted; the chain would sample a truncated distribution " *
                "-- raise capacity, lower fugacity/pressure, or attempt fewer insertions per cycle"
        )
    )
    return nothing
end

function mc_delete!(
        ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity, kT::T;
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = default_nblocks_per_chain(T, state.nsys)
    ) where {T, N}
    ispow2(groupsize) || throw(ArgumentError("mc_delete!: groupsize=$groupsize must be a power of two"))
    nblocks_per_chain >= 1 || throw(ArgumentError("mc_delete!: nblocks_per_chain=$nblocks_per_chain must be >= 1"))
    nblocks_per_chain <= ws.nblocks_max ||
        throw(ArgumentError("mc_delete!: nblocks_per_chain=$nblocks_per_chain exceeds the workspace's nblocks_max=$(ws.nblocks_max)"))
    G = Int(groupsize)
    nbpc = Int32(nblocks_per_chain)
    nsys = state.nsys
    dfug = adapt(backend, T(PASCAL) .* T.(fugacity))

    evaluate_delete_kernel!(backend)(
        ws.partial1, ws.partial2, state.refpoints, state.orientations, state.Sk, batch, guest, guest_types,
        state.guest_offsets, state.occupancy, state.k_offsets, state.rng_seed, state.rng_counter, nbpc, Val(G);
        ndrange = (G, nblocks_per_chain, nsys), workgroupsize = (G, 1, 1)
    )
    KernelAbstractions.synchronize(backend)

    decide_delete_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.occupancy, state.energy, state.energy_abs_accum,
        state.rng_counter, ws.accept_flag, ws.move_newpos, ws.move_newq, ws.partial1, ws.partial2,
        batch, guest, state.guest_offsets, state.rng_seed, dfug, kT, const_p, const_q, nbpc; ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)

    apply_exchange_sk_kernel!(backend)(
        state.Sk, state.sk_abs_accum, batch, ws.accept_flag, ws.move_newpos, ws.move_newq, guest, state.k_offsets, Val(false);
        ndrange = (nblocks_per_chain * G, nsys)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    mc_exchange!(rng::AbstractRNG, ws, batch, state, guest, guest_types, const_p, const_q, fugacity,
                kT; backend = CPU(), groupsize = DEFAULT_GROUPSIZE,
                nblocks_per_chain = default_nblocks_per_chain(T, state.nsys)) -> Bool

The μVT exchange move: draws ONE fair coin from `rng` (`rand(rng, Bool)`) — shared across every
chain in the batch for this launch, exactly as `run_nvt!` draws one shared `movetype` per call to
`mc_step!` — and calls `mc_insert!` on `true`, `mc_delete!` on `false`. Returns which one ran.
`ws`, `const_p`/`const_q`, `groupsize` and `nblocks_per_chain` are forwarded unchanged to whichever
one runs; see `mc_insert!`'s docstring for why `const_p`/`const_q` are precomputed once per run
rather than derived here.

This is the ONLY sanctioned way to attempt insertion/deletion moves: `mc_insert!`/`mc_delete!`
individually do NOT satisfy detailed balance (this file's own opening comment on the μVT moves), so
calling them directly in a fixed or deterministic pattern samples the wrong equilibrium loading
while every existing energy audit still passes. `p_ins = p_del = 1/2` here is exactly what
`log_insertion_prefactor`/`log_deletion_prefactor` assume; passing an `rng` that is not fresh per
launch, or substituting a biased draw, breaks that assumption silently.
"""
function mc_exchange!(
        rng::AbstractRNG, ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N},
        guest_types::SVector{N, Int}, const_p, const_q, fugacity, kT::T;
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, nblocks_per_chain::Integer = default_nblocks_per_chain(T, state.nsys)
    ) where {T, N}
    do_insert = rand(rng, Bool)
    if do_insert
        mc_insert!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity, kT; backend, groupsize, nblocks_per_chain)
    else
        mc_delete!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity, kT; backend, groupsize, nblocks_per_chain)
    end
    return do_insert
end
