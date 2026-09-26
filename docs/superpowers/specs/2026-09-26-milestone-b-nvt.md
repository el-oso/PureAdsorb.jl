# PureAdsorb.jl Milestone B — Widom insertion along an NVT Monte Carlo run

Status: draft for approval. Nothing here is built.

Milestone A computes the excess chemical potential of a guest in an *empty* rigid framework.
Milestone B puts a fixed number of guests in the box, samples their configurations with
Metropolis Monte Carlo at constant volume and temperature, and measures Widom insertion along
that chain. The quantity stops being a Henry coefficient and becomes a chemical potential at
finite loading.

Validation target: kUPS `examples/mcmc_widom.yaml`, reproduced on the same framework, guest and
force field.

## 1. What changes, physically

In Milestone A the test particle interacts with the host only. Two properties follow from that,
and both are used throughout the current code:

- The host never moves, so its energy is a constant that cancels in every difference.
- There is exactly one guest, so there is no guest–guest interaction at all.

Milestone B keeps the first and destroys the second. With `N` guests present, the total
potential energy is

    U = U_host-host  +  Σ_i U_host-guest(i)  +  Σ_{i<j} U_guest-guest(i,j)

`U_host-host` stays constant and never needs evaluating. The second term is what
`insertion_energy` already computes, once per guest. The third term is new, and it is the whole
of the work in this milestone.

## 2. The Monte Carlo

### 2.1 Ensemble and acceptance

Fixed `N`, `V`, `T`. The chain samples the canonical distribution `P(x) ∝ exp(−U(x)/kT)`.
With a symmetric proposal, `q(x→x') = q(x'→x)`, detailed balance is satisfied by the Metropolis
acceptance probability

    A(x→x') = min(1, exp(−ΔU/kT)),   ΔU = U(x') − U(x)

Only `ΔU` is ever needed, so the chain never evaluates the total energy — except for the
periodic audit in §6.

### 2.2 Move set

One guest is chosen uniformly at random and one of three moves is applied, matching kUPS's
configuration (`translation_prob`, `rotation_prob`, `reinsertion_prob`, each 1/3 in the
reference case):

| Move | Proposal | Symmetric? |
|---|---|---|
| Translation | displace the guest's reference point by a uniform vector in a cube of half-width `δ` | yes |
| Rotation | rotate about the reference point by a uniform random rotation, or by a small random rotation of angle ≤ `θ_max` | yes |
| Reinsertion | discard the pose; draw a uniformly random position in the cell and a uniformly random orientation | yes |

All three are symmetric, so the plain Metropolis criterion above applies and no Jacobian or
proposal-density ratio enters. Uniform random orientations use the same Shoemake construction as
Milestone A, which is uniform on SO(3); a small-angle rotation composed from a uniformly random
axis and an angle drawn symmetrically about zero is also symmetric, since the reverse move has
the same density.

`δ` and `θ_max` control the acceptance rate, not the answer. They are inputs; this milestone
does **not** auto-tune them during sampling, because adapting a step size using the chain's own
history breaks detailed balance unless the adaptation stops. If tuning is wanted it belongs in a
separate equilibration phase whose samples are discarded.

### 2.3 Cycles, and where Widom fits

kUPS's unit of work is a *cycle*: `num_displacements_per_cycle` moves, then
`num_widom_per_cycle` test-particle insertions into the configuration as it then stands. The
Widom estimator is unchanged from Milestone A in form,

    μ_ex = −kT ln ⟨W⟩,    W = exp(−ΔU_test/kT)

but `ΔU_test` is now the energy of the test particle against the host **and** all `N` guests,
and the average runs over both the insertion points and the Markov chain's configurations.
Statistical error comes from block averaging over cycles, not over insertions, because
successive cycles are correlated — this is the substantive change to the error analysis, and
using Milestone A's per-insertion blocks here would understate the error.

## 3. Guest–guest energy

### 3.1 Lennard-Jones

Pairwise over guest sites, Lorentz–Berthelot mixing, the same plain truncation at `cutoff` as
the host term. With `N ≤ 200` (kUPS's `max_num_adsorbates`) and a handful of sites per guest,
the guest–guest pair count is small enough that an all-pairs loop over guests is the right
structure; a cell list over guests would cost more than it saves at this `N`. The analytic tail
correction gains a guest–guest contribution proportional to `N(N−1)/2` per unit volume, which
must change when `N` changes — in NVT it is constant, but the expression should be written so
Milestone C can vary `N` without reworking it.

### 3.2 Ewald with mobile charges — the crux

This is where the existing code does not extend, and the reason is worth stating precisely.

The reciprocal-space energy of the whole system is

    U_recip = Σ_k pref_k |S(k)|²,    S(k) = S_host(k) + Σ_i S_i(k)

where `S_i(k)` is the structure factor of guest `i`. Milestone A exploits two facts that both
fail here:

1. **E1's sparse k table.** The current `FrameworkBatch` stores only the k-vectors *coupled to
   the framework's replication* — 190 of 4587 for the reference case — because `S_host(k)`
   vanishes for every other k when the cell is an exact replication of a smaller one. That is a
   statement about the host alone. Guest positions are arbitrary, so `S_i(k)` is nonzero at
   **every** k, and both the guest–host cross term and the guest–guest term need the full set.
   Milestone B therefore restores the full k table. The reference case goes from 190 to 4587
   k-vectors, a factor of 24 in reciprocal work and in the per-framework memory for `S_host`.

2. **The guest self term as a constant.** Milestone A folds `Σ_k pref_k |S_g(k)|²` into
   `constant_offset` as an orientation-mean, valid because there is one guest and the term
   depends only on its orientation. With `N` guests it is neither constant nor separable, since
   the cross terms between different guests are exactly what we are trying to compute.

The intramolecular exclusion within each guest is unchanged from Milestone A: the reciprocal sum
counts interactions between sites of the same molecule, which are not physical bonds' worth of
Coulomb energy, so each guest subtracts `Σ_{s<t} q_s q_t (1 − erfc(α r_st)) / r_st` over its own
site pairs. For a rigid guest this is a constant per guest and cancels in every `ΔU`, so the
chain never evaluates it; it matters only for the absolute energy audit in §6.

### 3.3 Incremental update

Moving one guest `i` changes `S(k)` by `ΔS(k) = S_i^new(k) − S_i^old(k)`, so

    ΔU_recip = Σ_k pref_k ( 2 Re[ conj(S(k)) ΔS(k) ] + |ΔS(k)|² )

with `S(k)` the running total *before* the move. The chain therefore carries `S(k)` as mutable
state, updated to `S(k) + ΔS(k)` on acceptance and left alone on rejection. This is the standard
construction and it makes a move cost `O(n_k)` rather than `O(N n_k)`.

Cost estimate, to be confirmed by measurement: at 4587 k-vectors a move's reciprocal work is
comparable to or larger than its real-space work over roughly 360 neighbors within the cutoff.
The reciprocal sum was 1–2% of Widom kernel time on the GPU and 13.7% on the M6 CPU; in
Milestone B it is expected to dominate. **This is the number to measure first**, before any
optimization is designed, and it may well redirect the whole effort — the same discipline that
corrected three wrong bottleneck guesses in the efficiency campaign.

## 4. Parallelism

A Markov chain is sequential: move `n+1` depends on the outcome of move `n`, and no amount of
hardware changes that. Parallelism comes entirely from running many *independent* chains, which
is the axis this package is already built around:

- one chain per framework in the batch, and optionally several replicas per framework with
  different seeds;
- every chain advances one move per kernel launch, in lockstep across the batch;
- chains are independent, so there is no communication between them.

Lockstep raises one real issue. Different chains choose different move types and accept or
reject independently, so a single kernel handling all three move types will have divergent
branches. Two structures are possible: one kernel with a branch on move type, or a fixed
move-type schedule shared across the batch so every chain performs the same kind of move at the
same step. The second removes the divergence and is still valid, because the move *type* may be
chosen by any rule that does not depend on the configuration; only the guest choice and the
acceptance must be random per chain. This spec adopts the shared schedule for that reason, and
the decision should be revisited if measurement shows divergence was not the cost.

## 5. State and data structures

Per system, mutable and living on the device:

| Field | Type | Notes |
|---|---|---|
| guest reference points | `SVector{3,F}` × N | wrapped into the cell |
| guest orientations | `SVector{4,F}` × N | unit quaternions, same convention as Milestone A |
| running structure factor | `Complex{F}` × n_k | host + all guests |
| running energy | `F` | for the audit only; not used in acceptance |
| RNG state | per chain | device-side, one stream per chain |
| move counters | accepted/attempted per move type | for the acceptance-rate report |

`FrameworkBatch` stays immutable and keeps the host data; a new `SystemState` holds the mutable
part, so a batch can be reused across runs and the state can be serialized between them.

The ragged-offset layout of Milestone A carries over for guests, so systems may hold different
`N` — not needed in NVT, but Milestone C requires it and building it in now avoids a rewrite.

## 6. Correctness, and how we will know

The failure mode of incremental Monte Carlo is silent: a wrong `ΔU` still produces a plausible
chain, converging to the wrong distribution. Three independent checks:

1. **Energy audit.** Recompute `U` from scratch every `n_audit` cycles and compare against the
   running value accumulated from accepted `ΔU`s. They must agree to accumulated rounding. This
   catches almost every incremental-update error, and it is the single most valuable test here.
   Fail fast on a discrepancy rather than warning.
2. **Detailed balance on a tiny system.** For a system small enough to enumerate or sample
   exhaustively, check that the chain's occupancy distribution matches `exp(−U/kT)` directly.
3. **Reversibility.** Applying a move and its inverse returns `ΔU` values that sum to zero,
   and the structure factor returns to its previous value.

Then the external checks: the Milestone A limit (`N = 0` must reproduce Milestone A's `μ_ex`
exactly, not approximately, for the same seed and poses), and agreement with kUPS on
`mcmc_widom.yaml` within combined statistical error.

## 7. Requirements checklist

- ☐ Full k-vector table restored for systems carrying guests; `verify_replication`'s sparse path
      remains for Milestone A and is selected by whether guests are present.
- ☐ Guest–guest Lennard-Jones with Lorentz–Berthelot mixing and the same truncation as the host
      term; tail correction written as a function of `N`.
- ☐ Guest–guest and guest–host reciprocal terms via a running structure factor, updated
      incrementally on acceptance.
- ☐ Per-guest intramolecular exclusion, constant per guest, excluded from `ΔU` and included in
      the absolute-energy audit.
- ☐ Translation, rotation and reinsertion moves, each verifiably symmetric, with Metropolis
      acceptance; step sizes as inputs, no adaptation during sampling.
- ☐ Shared move-type schedule across the batch; guest choice and acceptance random per chain.
- ☐ Device-side RNG, one stream per chain.
- ☐ Widom insertion into the occupied configuration every cycle, with block averaging over
      cycles.
- ☐ Energy audit every `n_audit` cycles, fail-fast on mismatch.
- ☐ `N = 0` reproduces Milestone A exactly for a fixed seed.
- ☐ Agreement with kUPS `mcmc_widom.yaml` within combined statistical error.
- ☐ Reciprocal-versus-real-space cost measured before any optimization is designed.
- ☐ Generic over `Float32`/`Float64`; runs on CPU, CUDA, ROCm and Metal (Float32 only).
- ☐ Allocation-free and type-stable kernels under the StrictMode audit.

## 8. Open questions

- The exact kUPS semantics of `mcmc_widom.yaml` — how it interleaves Widom with moves, whether
  Widom insertions count toward the chain, its step sizes and how it reports uncertainty. These
  must be read from the kUPS source before implementation, since "reproduces exactly" is the
  acceptance criterion.
- Whether Float32 is sound for a long chain. Widom in Float32 was validated against Float64, but
  an accumulated running energy over millions of accepted moves is a different numerical
  question, and the audit in §6.1 is what will answer it.
- Whether `N` guests per system should be fixed at batch construction or allowed to vary from
  the start. Milestone C needs variable `N`; building it now costs little and avoids a rewrite.
