# PureAdsorb.jl Milestone B (NVT Monte Carlo) Implementation Plan

Design: `docs/superpowers/specs/2026-09-26-milestone-b-nvt.md` (approved 2026-09-26).
Status: approved 2026-09-26, including the change to the validation target recorded below.

## Change to the approved design — approved

The design names kUPS `examples/mcmc_widom.yaml` as the validation target. **That example sets
`init_adsorbates: [0]`.** It runs Widom insertion into an empty framework, which is Milestone A;
its 20 displacement moves per cycle act on nothing. It cannot validate anything in Milestone B.

The right targets, both shipped or trivially derived from what kUPS ships:

| Case | Configuration | What it validates |
|---|---|---|
| B0 | RUBTAK 3×3×3, `N = 0` | Must reproduce Milestone A *exactly* for a fixed seed |
| B1 | kUPS `examples/nvt_co2_pressure_test.yaml`: 50 CO2 in a 30 Å cubic box (`host/empty.cif`, one non-interacting dummy site), `exchange_prob: 0` | Guest–guest Lennard-Jones and Ewald **in isolation** — there is no host at all |
| B2 | RUBTAK 3×3×3 with `init_adsorbates: [N]`, `exchange_prob: 0`; a config we write in kUPS's own schema | Host and guests together, the real Milestone B case |

B1 is the valuable addition. A pure CO2 fluid in an empty periodic box exercises exactly the
term this milestone adds, with the host contribution identically zero, so a discrepancy cannot
hide behind a large correct host energy. Validating against the combined case alone would be
much weaker.

Everything else in the approved design stands unchanged.

## Global constraints

- Julia only; no Python anywhere in the package. kUPS runs via `uv` from `~/src/kups` as a
  reference stopwatch only, under the standing narrow exception, with `JULIA_GUARD=off` on those
  commands alone.
- Generic over `Float32`/`Float64`; kernels written once with KernelAbstractions and run on CPU,
  CUDA, ROCm and Metal. Metal is Float32 only.
- No `@inbounds` unless a measurement justifies it and a test under `--check-bounds=yes` covers it.
- Kernels stay allocation-free and type-stable under the StrictMode audit; no throwing branches,
  no multi-array `eachindex`, no `floor(Int, x)` or `mod`/`div` by a runtime value.
- Every claim about cost is measured, not asserted. Decompose before optimizing.
- TestItems.jl; Runic formatting; `Assisted-by` commit trailer.
- One writer in the tree at a time.

## kUPS conventions this plan reproduces

Read from the source at commit `e183c9a`; re-verify each before relying on it.

- **Move probability defaults** (`src/kups/application/mcmc/data.py:209-212`): translation 1/6,
  rotation 1/6, reinsertion 1/6, exchange 1/2. NVT is selected by `exchange_prob: 0`, leaving the
  three NVT moves. **Open: how the remaining probabilities are normalized when they sum to 1/2.**
  Task 1 must settle this from the source; it changes the move mix and therefore the chain.
- **Cycle length** (`core/propagator.py:474-521`): a cycle repeats its propagator a number of
  times that scales with the current particle count, floored by `min_cycle_length`. The exact
  rule must be read and mirrored, since "cycle" is the unit our block averaging uses.
- **Acceptance** (`core/propagator.py:325`): `accept = log_p_ratio > log(uniform)`, one draw per
  system. Note this is a strict inequality on logs; match it, including the treatment of a
  zero-probability proposal.
- **Incremental Ewald** (`potential/classical/ewald.py:440-490`): kUPS maintains the structure
  factor across moves, the same construction this milestone adopts. Their `custom_jvp` wrapper is
  for autodiff correctness and has no counterpart here; ignore it.
- **Lennard-Jones**: Lorentz–Berthelot, plain truncation, analytic tail correction for energy
  (and pressure, which we do not need), generalized per species pair.
- **Block averaging** (`core/utils/block_average.py::optimal_block_average`): automatic block-size
  selection by a plateau criterion with `rtol = 0.05`. Ours uses a fixed block count. For the
  comparison, report both codes' own uncertainty and do not claim agreement tighter than the
  looser of the two.

## Validation ladder

Internal checks first; they find bugs that cross-code agreement hides.

1. **Energy audit.** Recompute the total energy from scratch every `n_audit` cycles and compare
   against the value accumulated from accepted `ΔU`. Fail fast on mismatch beyond accumulated
   rounding. This is the highest-value test in the milestone.
2. **Reversibility.** A move and its inverse give `ΔU` summing to zero and restore `S(k)`.
3. **B0 exactness.** `N = 0` reproduces Milestone A bit-for-bit for a fixed seed.
4. **Tiny-system distribution.** One guest in a small box, sampled long enough to compare the
   occupancy histogram against `exp(−U/kT)` computed by brute force.
5. **B1, then B2**, against kUPS within combined statistical error.

## File structure

```
src/state.jl      SystemState: guest poses, running S(k), running energy, RNG, counters
src/guest.jl      guest–guest Lennard-Jones and Ewald, tail correction as a function of N
src/moves.jl      translation, rotation, reinsertion; proposal and ΔU
src/nvt.jl        cycle driver, Metropolis acceptance, Widom along the chain, block statistics
```

`src/energy.jl`, `src/ewald.jl`, `src/batch.jl` gain the full-k path; none of their existing
behavior changes, because Milestone A keeps using the sparse table.

## Tasks

Each task ends with its own tests passing and a commit. Run the filtered test items during
iteration and the full suite only as a pre-merge gate.

### Task 1 — kUPS convention extraction
Read `application/mcmc/`, `mcmc/moves.py`, `mcmc/probability.py`, `core/propagator.py`,
`core/utils/block_average.py`. Write down, with file and line: the move-probability
normalization, the cycle-length rule, the acceptance comparison, the RNG stream structure, and
how Widom insertions interleave with moves. Settle the open question above. Deliverable: a
section appended to this plan, not code.

### Task 2 — Full k-vector path
`FrameworkBatch` gains the full k table and `S_host` over all k, selected when a batch carries
guests; the sparse Milestone A path stays untouched and stays the default for `widom`. Acceptance:
the existing Milestone A tests are unchanged and still pass; a new test shows the full-k
`S_host` reproduces the sparse one on the coupled subset.

### Task 3 — Guest–guest energy
LJ over guest site pairs with Lorentz–Berthelot and plain truncation; tail correction written as
a function of `N`. Reciprocal term via the running structure factor. Per-guest intramolecular
exclusion, constant per guest, in the absolute energy only. Acceptance: total energy of a
configuration agrees with a brute-force reference (direct lattice sum at large cutoff) to 1e-10
relative, for the empty-box CO2 case and for RUBTAK with guests.

### Task 4 — `SystemState`
Mutable per-system state as specified, with ragged offsets so `N` may differ per system, even
though NVT holds it fixed. Adapt to device. Acceptance: round-trips to device and back
unchanged; StrictMode audit clean.

### Task 5 — Moves and acceptance
Translation, rotation, reinsertion, each with its `ΔU` computed incrementally. Metropolis
acceptance per chain, shared move-type schedule across the batch. **Document in the source why a
fixed schedule is valid**: each move individually satisfies detailed balance, so their
composition preserves the target distribution; what the fixed order gives up is reversibility of
the composite, which a palindromic ordering would restore. Acceptance: reversibility test;
acceptance rate varies sensibly with step size.

### Task 6 — Device-side RNG
One stream per chain, counter-based, reproducible for a given seed independent of chunking and
of the number of chains. Acceptance: identical results across two different batch sizes
containing the same system.

### Task 7 — Cycle driver and Widom along the chain
Warmup cycles discarded; `num_displacements_per_cycle` moves then `num_widom_per_cycle`
insertions into the current configuration; block averaging over cycles. Acceptance: B0
reproduces Milestone A exactly.

### Task 8 — Energy audit
From-scratch recomputation every `n_audit` cycles, fail fast. Acceptance: a deliberately
corrupted `ΔU` is caught.

### Task 9 — Validation
The ladder above, B1 then B2, with kUPS run under the standing exception. Acceptance: agreement
within combined statistical error, reported as a number of combined standard errors.

### Task 10 — Benchmark and documentation
Measure the reciprocal-versus-real-space split **before** proposing any optimization. Benchmark
on the 4070 and the R9700, save JSON, and extend the docs with the theory and the results.

## Amendment — throughput is a requirement, not an outcome (owner, 2026-09-26)

Milestone B is judged on moves per second as well as on correctness. This is a change in kind:
the plan as written measured cost at task 10 and left optimization to a later stage. It does not
license guessing — measurement still precedes design — but it does mean the cost measurement
moves early and the structure is chosen with the measurement in hand.

**The dominant cost is known by construction.** A move changes one guest, so the real-space work
touches only its neighbors, while the reciprocal update touches every k-vector:
`ΔU_recip = Σ_k pref_k (2 Re[conj(S) ΔS] + |ΔS|²)`. At 4587 k-vectors against roughly 360
neighbors inside the cutoff, the reciprocal sum is expected to dominate a move outright. That
expectation is a hypothesis, not a finding, and task 3 must measure the split before task 5
commits to a kernel structure.

**Consequent changes to the task order:**
- Task 3 gains a requirement: measure per-move reciprocal cost against real-space cost, both
  precisions, on the 4070 and the R9700, and record it in `bench/results/`. This measurement,
  not task 10's, is what the design responds to.
- Task 5 must not be designed until that number exists.

**Levers, in the order they should be considered.** None is approved; each needs its own
measurement and, where it changes the physics, the owner's sign-off.
1. **Choose α to minimize total work.** The Ewald splitting parameter trades real-space against
   reciprocal work at fixed accuracy: raising α shrinks the real-space cutoff's reach and grows
   `n_k`, lowering it does the reverse. kUPS picks α from a closed form in the real cutoff and
   the precision, with no reference to cost. We are free to pick the α that minimizes *our*
   measured cost at the same accuracy, which is a legitimate optimization and changes no physics.
   This is the first thing to try because it is free and principled.
2. **Amortize the launch.** Several moves per chain per kernel launch, which is sound because
   moves within a chain are sequential anyway and the batch is what provides parallelism.
3. **Exploit the structure of ΔS.** For a translation, the phase factors of the moved guest's
   sites change by a common factor per k-vector; for a rotation, the reference point is
   unchanged. Whether either admits a cheaper update than recomputing the guest's structure
   factor is a question for the measurement, not for assertion here.
4. **A different electrostatics scheme** (Wolf or damped shifted force) would make guest–guest
   electrostatics O(N) with no reciprocal sum at all. It changes the physics and therefore breaks
   exact comparability with kUPS, so it belongs in the design's Milestone D as an accelerator
   gated by measurement, never as a silent substitution.

**What throughput does not license.** Not `@inbounds` without a measurement and a bounds-checked
test; not dropping the energy audit; not weakening the validation ladder. A fast chain that
samples the wrong distribution is worth nothing, and the audit is what tells the two apart.

## Open questions carried from the design

- Whether Float32 is sound for a long chain; the energy audit answers it.
- Whether the reciprocal sum dominates once the k set grows 24×; measured in task 10, and it may
  redirect the work.

## Task 1 findings — kUPS NVT conventions (commit `e183c9a`)

Read-only source extraction. All citations `path:line` against kUPS commit `e183c9a`. Items
marked **UNRESOLVED** need a second look before Task 5/6 rely on them; items marked
**CONTRADICTS DESIGN** are places where the approved design's description of kUPS differs from
what the source actually does — flagged, not silently corrected.

### 1. Move probability normalization — RESOLVED

`translation_prob`/`rotation_prob`/`reinsertion_prob`/`exchange_prob` are **unnormalized
selection weights**, not literal probabilities (`make_gcmc_mcmc_propagator` docstring,
`src/kups/mcmc/moves.py:1140-1143`). A move with weight `≤ 0` is **excluded from the candidate
list entirely** — it is not folded in as a "null move" chance:
`src/kups/mcmc/moves.py:1201,1216,1231,1251` (`if <weight> > 0:` gates whether the move and its
weight are appended to `propose_fns`/`wts`). The surviving weights are then renormalized to sum
to one by `propose_mixed`: `probs = jnp.array(weights) / sum(weights)`, `jax.random.choice(key,
n, p=probs)` (`src/kups/core/propagator.py:258-262`). With `exchange_prob: 0`
(`examples/nvt_co2_pressure_test.yaml:41`), the exchange move is dropped from the candidate list
and the three remaining weights `(1/6, 1/6, 1/6)` (defaults, `src/kups/application/mcmc/data.py:
209-211`) renormalize to `(1/3, 1/3, 1/3)`. Confirmed end to end: `application/simulations/
mcmc_rigid.py:370-378` passes `config.translation_prob` etc. straight through as
`*_weight` with no other transformation.

### 2. Step sizes and adaptation — RESOLVED, and it never stops

Translation step (`step_width`, a per-system scalar) and rotation step (`step_width` in
`[0,1]`, see §3) are both driven by a `ParameterSchedulerState`
(`src/kups/core/parameter_scheduler.py:61-158`), defaults `initial_value=0.1`,
`multiplicity=1.1`, `target=0.5` acceptance, `tolerance=0.05`, `history_length=100` samples.
`acceptance_target_schedule` (`src/kups/core/parameter_scheduler.py:161-208`) is a classic
Robbins–Monro-style multiplicative tuner: every `history_length` acceptance samples it compares
the rolling average acceptance to `target`; outside `±tolerance` it multiplies (or divides) the
step by `multiplicity`, clipped to `bounds`. Translation's upper bound is `min_half_box` (half
the shortest cell perpendicular length); rotation and reinsertion have no upper bound
(`application/simulations/mcmc_rigid.py:473-485`).

**This scheduler updates on every single propagator call, unconditionally** —
`MCMCPropagator.__call__` always writes back the selected move's updated scheduler state
(`src/kups/core/propagator.py:331-335`, `tree_map(lambda *cs: select_n(which, *cs), *candidates)`
runs regardless of accept/reject). `run_mcmc` (`application/mcmc/simulation.py:53-63`) and
`make_cycle_function`/`run_warmup_cycles`/`run_simulation_cycles`
(`application/utils/propagate.py:36-104`) compile and reuse **the same propagator** for both the
warmup and the production phase; nothing freezes or removes the scheduler for production. So
kUPS's own step sizes keep adapting for the entire run, warmup and production alike, with no
mechanism to stop. This **CONTRADICTS the design's own stated rationale** (spec §2.2: "this
milestone does not auto-tune them during sampling, because adapting a step size using the
chain's own history breaks detailed balance unless the adaptation stops") — that is a
deliberate design choice for *our* code, correctly reasoned, but it means our chain and kUPS's
chain are not literally the same stochastic process; agreement can only be statistical, and kUPS
is (by the design's own logic) not running an exact Metropolis chain in the first place. Worth
flagging to the user; not something to "fix" in the plan's other sections.

One more wrinkle: `ReinsertionMove`/`propose_reinsertion` (`src/kups/mcmc/moves.py:268-296`)
takes no `step_width` argument at all — reinsertion is always a full random reinsertion
regardless of `reinsertion_params.value`. The reinsertion scheduler still exists and still gets
updated (`src/kups/mcmc/moves.py:1000-1017`, `mcmc_rigid.py:480-482`) but its tuned value is
never read back by anything — pure acceptance-rate bookkeeping.

### 3. The three NVT moves, exactly — RESOLVED, with one description mismatch

All three select one group (=one guest molecule) uniformly per system
(`random_select_groups`, `src/kups/mcmc/moves.py:122-147`, one uniform draw of bits per system
via `select_per_label`). All three return a proposal log-ratio of exactly zero
(`src/kups/mcmc/moves.py:358,396,429`) — i.e. kUPS's own code treats all three as symmetric,
confirming the design's claim.

- **Translation** (`propose_group_translation`, `moves.py:222-245`): draws
  `distribution(key, (n_sys, 3)) * step_width[:, None]` with **`distribution` defaulting to
  `jax.random.normal`** (`moves.py:229,379-381`), not a uniform cube as the design's table
  states (spec §2.2: "uniform vector in a cube of half-width δ"). **CONTRADICTS DESIGN**: the
  actual proposal is an isotropic Gaussian displacement of the whole group by one shared vector
  per system, added in Cartesian coordinates to every particle in the group, then
  `cell.wrap`-ped. Still symmetric (normal(0,σ) is symmetric about zero), so this doesn't affect
  correctness of the target distribution, only which exact proposal shape we'd need to match for
  a step-by-step (not just statistical) comparison.
- **Rotation** (`propose_group_rotation` → `random_rotate_groups`, `moves.py:158-197,247-266`):
  computes the group's center of mass (periodic-aware `center_of_mass`), converts to
  COM-relative positions, applies `Quaternion.random(key, (n_sys,)) ** step_width`, converts
  back. `Quaternion.random` (`src/kups/core/utils/quaternion.py:70-99`) is the Shoemake (1992)
  construction, uniform on SO(3). `Quaternion.__pow__`/`_pow_quaternion`
  (`quaternion.py:184-210,242-264`) scales the rotation **angle** by `step_width` while keeping
  the **same random axis** (`0 ≤ step_width ≤ 1`; `step_width=0` → identity, `=1` → the full
  uniform rotation). This is symmetric because the axis is uniform on the sphere and independent
  of angle, so axis `u` and `-u` are equally likely and give the same rotation with `θ → -θ`.
  **Description mismatch, not a correctness issue**: the design's table describes "a small
  random rotation composed from a uniformly random axis and an angle drawn symmetrically about
  zero" — mechanically different from "one Shoemake-uniform quaternion raised to a fractional
  power," though both are valid symmetric proposals. Reproducing kUPS's actual step-size
  trajectory (not just its target distribution) requires the power construction specifically.
- **Reinsertion** (`propose_reinsertion`, `moves.py:268-296`): always uses `step_width=1`
  (full random rotation) about the *old* center of mass, then adds a translation
  `triangular_3x3_matmul(cell.vectors, uniform(0,1)^3)` (a vector uniform over the whole cell in
  fractional coordinates) and wraps. Because the final position is `wrap(old_position +
  cell_uniform_offset)`, the result is uniformly distributed over the cell **independent of the
  starting position** — algebraically equivalent to "draw a fresh uniform COM + fresh uniform
  orientation," matching the design's description, and trivially symmetric since the proposal
  density does not depend on the current state at all.

### 4. Cycle definition — RESOLVED, and the two example YAMLs are not comparable the way the design implies

`LoopPropagator` (`src/kups/core/propagator.py:470-522`) repeats its wrapped propagator a
`jax.lax.while_loop`-counted number of times, either fixed or from a state-dependent view.
`application/simulations/mcmc_rigid.py:379-384` (the entry point that actually runs
`nvt_co2_pressure_test.yaml`) sets:

    repetitions = max(groups.data.system.counts.data.max(), config.min_cycle_length)

i.e. **one scalar repetition count shared by the whole batch**: the maximum number of guest
groups over all systems in the batch, floored by `min_cycle_length`. For
`nvt_co2_pressure_test.yaml` (`init_adsorbates: [50]`, `min_cycle_length: 1`): one cycle = 50
move attempts (each attempt = one random group + one random move type, translation/rotation/
reinsertion at 1/3 each per §1). **There is no Widom insertion anywhere in this run** — see §8.

**CONTRADICTS DESIGN's implicit framing (via the design's citation of `mcmc_widom.yaml`)**:
`num_displacements_per_cycle` and `num_widom_per_cycle` are fields of a *different* config —
`WidomRunConfig` in `application/simulations/mcmc_widom.py:109-120`, used only by
`mcmc_widom.py`'s own `run()`. `mcmc_rigid.RunConfig` (`application/mcmc/data.py:199-216`, the
schema `nvt_co2_pressure_test.yaml`'s `run:` block actually matches — it has
`min_cycle_length`/`exchange_prob`, which `WidomRunConfig` does not have) **has no such fields at
all**, and `mcmc_rigid.py` never constructs a Widom probe. So "how `num_displacements_per_cycle`
interacts with [the dynamic cycle length]" (question 4) has no answer for B1: that field simply
does not apply to the code path that runs `nvt_co2_pressure_test.yaml`. It only exists in the
separate Widom entry point, whose own cycle is `SequentialPropagator((LoopPropagator(nvt_moves,
num_displacements_per_cycle), LoopPropagator(widom_probe, num_widom_per_cycle)))` — a **fixed**
per-cycle move count, not the dynamic particle-count formula above. Our own cycle driver (Task 7)
composes both ideas (dynamic NVT move count *and* Widom-per-cycle) in a way **no single kUPS
example exercises together** — see §8's closing note.

### 5. Acceptance — RESOLVED

Confirmed exactly as cited in the plan: `src/kups/core/propagator.py:325`,

    accept = log_p_ratio > jnp.log(jax.random.uniform(next(chain), (n_sys,)))

`log_p_ratio = move_log_ratio + density.data` (`propagator.py:323`), where `move_log_ratio` is
identically zero for all three NVT moves (§3) and `density` is `BoltzmannLogProbabilityRatio`
(`src/kups/mcmc/probability.py:100-111`): `(U_old − U_new) / (k_B T)`, computed via
`KahanSummand.difference` so the compensated accumulators are subtracted exactly rather than
losing `ΔU` to rounding against a large total. **No explicit non-finite/zero-probability
handling was found** (`grep` for `isnan`/`isinf` in `core/propagator.py`, `mcmc/moves.py`,
`mcmc/probability.py` returns nothing): an overlapping insertion drives `ΔU → +∞`, hence
`log_p_ratio → -∞`, which fails the `>` comparison under ordinary IEEE-754 semantics and is
rejected; a `NaN` would also fail any `>` comparison and so is also (silently) rejected. This is
an emergent property of floating-point comparison, not a designed safety branch — flagging this
since **it is easy to reproduce accidentally-correct behavior in Julia and mistake it for a
verified invariant; write an explicit test for the overlapping-insertion case rather than relying
on the same accident.**

### 6. Random numbers — PARTIALLY UNRESOLVED

`key_chain` (`src/kups/core/utils/jax.py:455-487`) is a generator that folds an incrementing
integer counter into one key per `next()` call (`jax.random.fold_in(key, i)`, `i` starting at 0).
Nested composition (`MCMCPropagator`, `propose_mixed`, `LoopPropagator`'s own inner
`jax.lax.while_loop` body which instead calls raw `jax.random.split` once per iteration,
`propagator.py:511-514`) builds a deterministic tree of `split`/`fold_in` calls whose shape is
fixed by the *static* structure of the composed propagator, not by `n_sys`. **Per-system
randomness within one move step comes from a single JAX key used with shape `(n_sys, ...)`**
(e.g. `jax.random.normal(key, (n_sys, 3))` in `propose_group_translation`,
`jax.random.uniform(key, (n_sys,))` for the accept draw) — kUPS does **not** explicitly split one
key per system; it relies on JAX's counter-based (threefry, by default) generator to decorrelate
array elements from one key+shape call.

**UNRESOLVED**: whether `jax.random.uniform(key, (n,))[i] == jax.random.uniform(key, (m,))[i]`
for `i < min(n, m)` — i.e. whether this construction is actually reproducible independent of
batch size, which is exactly what our Task 6 acceptance criterion requires — is a property of
JAX's threefry-2x32 bit generator's shape handling, not something kUPS's own source asserts,
tests, or documents anywhere (no test in `test/` checks batch-size invariance). This should be
verified empirically against JAX's own semantics (or treated as a property our device-side RNG
must establish on its own terms) rather than assumed from reading kUPS's source.

### 7. Energy conventions with N guests — RESOLVED, with one formula CONTRADICTS DESIGN

**Lennard-Jones.** Guest–guest and guest–host pairs are computed by the exact same
`make_lennard_jones_potential` call over all particles together (`src/kups/potential/classical/
lennard_jones.py:352-386`), same Lorentz–Berthelot mixing (`lennard_jones.py:118-136`), same
per-system cutoff. Intramolecular (same-group) pairs are excluded from the neighbor list itself
via `MCMCParticles.exclusion = group.to_cls(ExclusionId)` (`application/mcmc/data.py:276-278`)
feeding the generic "drop pairs sharing an exclusion segment" neighbor-list mask
(`src/kups/core/neighborlist/masks.py:147-161`); confirmed `make_lennard_jones_from_state` passes
`state.particles` through unmodified (`application/potential/classical/lennard_jones.py:243`), so
no LJ energy or tail-correction contribution from bonded pairs at all — no separate exclusion
term needed for LJ, unlike Ewald (below).

**Tail correction — CONTRADICTS DESIGN.** `global_lennard_jones_tail_correction_energy`
(`lennard_jones.py:299-314`, using `_global_tail_correction_common`, `lennard_jones.py:273-296`)
is a single global sum over **all** species pairs present (host site labels and guest atom
labels together, from `inp.parameters.labels`), with `density[i,j] = counts_i · counts_j / V`
summed over the full `n_species × n_species` grid (not restricted to `i ≤ j`). This is the
standard Allen & Tildesley multi-component analytical tail formula, and for a guest self-pair
it scales as **`N²`** (`counts_guest · counts_guest`), not `N(N−1)/2` as spec §3.1 states. At
`N = 50` this is a ~2 % difference in that term alone (`2500` vs `1225`), not negligible for an
exact-agreement test. `tail_corrected` is masked to `epsilon > 0` per species pair
(`lennard_jones.py:250-256`), so `X1` (epsilon 0, see §10) never contributes here regardless.

**Ewald α / k_max.** Both example YAMLs set `ewald.real_cutoff` explicitly (`12.0`), which routes
`estimate_ewald_parameters` (`src/kups/potential/classical/ewald.py:1060-1191`) through its
"fast path" (no numerical optimization): `alpha` solves `erfc(alpha·rc) = rc·(precision/2)`, then
`k_max = 2·alpha·sqrt(-ln(precision/2))` — **a function of `real_cutoff` and `precision` only**;
charges/`N` are not used in this branch even though they're passed in. `EwaldParameters.make`
(`ewald.py:175-241`) runs once at `init_state`, so `alpha`/`k_max` are fixed for the whole NVT
run; `kvecs_from_kmax(cell, k_max)` (`ewald.py:1223-1246`) depends only on cell geometry and
`k_max`, so **the k-set is identical whether or not guests are present**, for both shipped
examples. (N-dependence would only appear via `estimate_ewald_parameters`'s general/optimizing
branch, which neither example uses.)

**Ewald intramolecular exclusion — CONTRADICTS DESIGN (spec §3.2's formula).** kUPS's real-space
(short-range) Ewald sum uses `atomic_view` with inclusion at the **system** level only
(`ewald.py:971-991`, `_system_inclusion`, `ewald.py:972-974`) and gives every particle a *unique*
exclusion id via `_convert_particles`'s `Index.arange(...)` (`ewald.py:906-915`) — so
intramolecular (bonded) pairs are **not** filtered out of the real-space sum; they receive the
ordinary `erfc(α r)/r` term like any other pair. Separately, a dedicated `exclusion_correction`
graph (`ewald.py:1012-1039`) selects exactly the same-molecule pairs (inclusion = the *original*
group/exclusion id, `all_connected` neighbor list) and subtracts their **plain, undamped**
Coulomb energy `q_i q_j / r` (`_pairwise_coulomb_energy`, `src/kups/potential/classical/
coulomb.py:69-77` — no `erf`/`erfc` factor anywhere in that function), scaled by `-1`
(`ewald.py:1040`). Since real-space already counts bonded pairs at `erfc(αr)/r` and reciprocal
space implicitly supplies `erf(αr)/r` for every pair including bonded ones (the structure factor
has no concept of exclusion), the two together sum to the full `1/r` for a bonded pair —
subtracting the *full* `q_i q_j/r` is the correct way to zero it out **given kUPS's choice to
include bonded pairs in real space**. The design's spec formula,
`Σ_{s<t} q_s q_t (1 − erfc(α r_st)) / r_st`, is the correct correction only under the *opposite*
convention (real space already excludes bonded pairs) and is **not numerically equal** to
kUPS's actual term at finite α (they agree only as α→∞). Reproducing kUPS exactly means copying
its actual pairing — real space includes bonded pairs, correction subtracts full Coulomb — not
the formula written in the spec. This is load-bearing for B1 (no host, so guest-guest Ewald is
the entire electrostatic energy) and should get a second, independent look before Task 3 is
implemented.

**Structure factor incrementality** — confirmed as the plan already states.
`_structure_factor_update` (`ewald.py:440-488`, matches the plan's cited range) computes
`S'(k) = S(k) + Σ_changed[ρ_new(k) − ρ_old(k)]` via `_frequency_response`/`segment_sum`,
Kahan-compensated, cached in `EwaldCache.structure_factor` (`ewald.py:104-121`), and patched into
state conditionally on `accept` by `EwaldCachePatch.__call__` (`ewald.py:142-171`,
`where_broadcast_last(mask, new, old)` — reverted on rejection). The `custom_jvp` wrapper
(`ewald.py:440-443,489-548`) is autodiff-only, exactly as the plan already notes; no counterpart
needed.

### 8. Widom along the chain — RESOLVED, but no kUPS example exercises our combination

Widom mechanics live only in `src/kups/mcmc/widom.py` and
`application/simulations/mcmc_widom.py` — a **separate application entry point** from
`mcmc_rigid.py` (the one that runs `nvt_co2_pressure_test.yaml`, see §4). In
`mcmc_widom.py::make_propagator` (lines 293-347), one cycle is
`SequentialPropagator((nvt_loop, widom_loop))`: `nvt_loop` runs all
`num_displacements_per_cycle` moves first, **then** `widom_loop` runs all
`num_widom_per_cycle` ghost insertions against the configuration as it stands after the moves —
not interleaved move-by-move.

`widom_test` (`src/kups/mcmc/widom.py:65-99`) proposes a ghost insertion
(`ExchangeMove.propose_insertion`, the same uniform-position/uniform-orientation mechanism as
reinsertion, §3), builds the fully patched state (so the ghost **does** see the host and all `N`
real guests through the same potential the production moves use), evaluates the bare Boltzmann
log-ratio, and explicitly discards the resulting patch (`# result.patch is intentionally
discarded --- state is NOT modified`, `widom.py:98`) — confirmed non-mutating.
`WidomStatistics.update` (`widom.py:143-160`) accumulates `Σ exp(ln α)` and
`Σ ΔU·exp(ln α)` per ghost insertion (`ΔU = −k_B T·ln α` exactly,
`mcmc_widom.py:250-260`); `μ_ex`/`K_H`/`q_st` are reduced from these sums post-hoc by
`analyze_widom_file` (not read in this pass — worth a quick confirmatory read before Task 9 if
exact block-averaging of `μ_ex` matters).

**Flag for the user, not something to fix in the plan**: no kUPS example ever runs "N-guest NVT
+ Widom-along-the-chain" together. `nvt_co2_pressure_test.yaml` (`N=50`) goes through
`mcmc_rigid.py`, which never invokes a Widom probe at all; `mcmc_widom.yaml` (the only file that
does Widom) sets `init_adsorbates: [0]`. The combination this milestone's own abstract describes
("measures Widom insertion along that chain" with guests present) is structurally supported by
kUPS's code (both pieces exist and compose the way §8 describes) but is not itself exercised or
validated by anything kUPS ships. B1 as chosen (per the design's own already-approved correction)
validates guest-guest energetics via the NVT chain alone, not any Widom estimate.

### 9. Uncertainty — RESOLVED

`optimal_block_average` (`src/kups/core/utils/block_average.py:245-304`). `block_transform`
(lines 180-242) sweeps `n_blocks` from `n_samples // 2` down by successive halving to the
largest value `≥ min_blocks` (default 4), discarding leftover samples that don't fill a whole
block (`compute_block_means`, lines 40-84) — i.e. from many small blocks toward few large blocks.
`optimal_block_average` walks that list in that same order, computes the relative change in SEM
between consecutive block sizes, and returns the block average at the **first (smallest-block)
transition** whose relative SEM change falls below `rtol` (default `0.05`), taking the
**larger-block-size member of that pair** (`optimal_idx = plateau_indices[0] + 1`); absent any
such plateau, it falls back to the single largest block size tested (fewest blocks, most
conservative). Our fixed-block-count estimate is a genuinely different rule (no plateau search),
so per the plan: report both codes' own SEM and never claim agreement tighter than the looser of
the two.

### 10. B1's actual physics — RESOLVED, confirms B1 is a pure CO2 fluid

`host/empty.cif` (verbatim): a 30×30×30 Å, 90°/90°/90°, `P 1` cell with one site, `X1`, at
fractional `(0, 0, 0)`. In `nvt_co2_pressure_test.yaml`, `lj.parameters.X1: [null, null]` maps
through `LennardJonesParameters.from_dict` (`src/kups/potential/classical/lennard_jones.py:
99-108`, docstring: "`None` values default to `sigma=1.0`, `epsilon=0.0`") to `epsilon = 0.0` —
X1 contributes exactly zero LJ energy to every pair it's in (energy ∝ epsilon), and is
automatically excluded from the tail correction too (`tail_corrected = epsilon > 0`,
`lennard_jones.py:250-256`). Charge: `_particles_from_atoms`
(`src/kups/application/utils/particles.py:292-320`) reads
`_atom_type_partial_charge`/`_atom_site_charge` from the CIF's parsed `atoms.info`, defaulting to
`jnp.zeros(...)` when neither key is present; `empty.cif` has no charge column at all, so X1's
charge is `0.0`. X1 is therefore confirmed a fully non-interacting dummy site — zero LJ epsilon
**and** zero charge — so B1 is exactly the pure-CO2-in-vacuum-with-PBC case the plan requires,
with the host contributing identically zero to every energy term.

### Confidence summary

High confidence, verified by direct code reading with no remaining ambiguity: §1 (move
normalization), §3 (move mechanics), §4 (cycle-length formula), §5 (acceptance comparison), §7's
LJ/tail-correction/α-k_max/structure-factor-incrementality claims, §9 (block averaging), §10
(B1 physics).

Needs a second look before implementation depends on it: §2's exact adaptation formula is solid,
but its consequence (kUPS never freezes step sizes) deserves the user's explicit sign-off since
it changes what "reproduces kUPS" can mean statistically. §7's Ewald intramolecular-exclusion
finding is the most consequential single item in this report — it contradicts the approved
spec's own formula and is load-bearing for B1 — re-derive it independently before Task 3 encodes
either version. §8's Widom mechanics are clear, but `analyze_widom_file`'s exact reduction
(`application/mcmc/analysis.py`) was not read in this pass.

Genuinely unresolved, not just under-read: §6, whether kUPS's key-threading is actually
reproducible independent of batch size, is a property of JAX's PRNG internals that kUPS's own
source neither states nor tests — do not assume it without an independent check against JAX's
documented semantics (or verify empirically).

## Rulings on the task 1 findings

**R1 — the Ewald exclusion is not a contradiction; keep ours.** Both codes are correct and give
the same total, because the exclusion term must match whatever the real-space sum did. kUPS
includes intramolecular pairs in its real-space sum at `erfc(αr)/r` and subtracts the full
`q_i q_j / r`; PureAdsorb omits them from the real-space sum and subtracts only the reciprocal
sum's leftover `q_i q_j erf(αr)/r`. Since `erf = 1 − erfc` identically, `erf(αr)/r` and
`(1 − erfc(αr))/r` are the same number, and the two schemes differ only in bookkeeping. This
pairing is already implemented and documented (`src/ewald.jl:174-176`, `docs/src/theory.md:219`).
The equivalence does require every intramolecular distance to lie inside the real-space cutoff
(1.16 Å against 12 Å for CO2); **add an assertion** rather than leaving it as an assumption.

**R2 — tail correction scales as N², superseding spec §3.1.** kUPS uses `N²`, which is also the
standard mean-field form: the correction integrates a uniform pair density beyond the cutoff, and
that counts `N²` rather than the exact pair count `N(N−1)/2`. The difference is 2% at `N = 50`,
large enough to show up in a comparison. Use `N²` and say so in the docstring.

**R3 — adopt kUPS's proposal shapes, superseding spec §2.2's table.** Translation displacements
are Gaussian, not uniform in a cube; rotations are a uniform random quaternion raised to a
fractional power, not a small perturbation of a fixed form. Both are symmetric, so the acceptance
rule is unchanged and the spec's *requirement* (symmetry, verified) stands. Matching their shapes
keeps acceptance rates comparable, which makes a disagreement easier to localize.

**R4 — we adapt step sizes during warmup only, and freeze for production. This is a deliberate
divergence from "reproduces kUPS" and needs the owner's sign-off.** kUPS's scheduler targets 50%
acceptance and never stops: the same compiled propagator runs in warmup and production and writes
back an updated step width on every call. Adapting from the chain's own history without ever
freezing means the chain is not exactly reversible and does not exactly sample the target
distribution. The bias is usually small and shrinks as the step width settles, but it is real, and
this project prefers an exact result to a matching one. Freezing after warmup costs nothing here
because we compare equilibrium averages, not trajectories.

**R5 — the milestone's headline combination has no kUPS counterpart; validate it in pieces.**
kUPS ships no configuration that runs N-guest NVT and Widom together: its Widom entry point uses
a different config class from the one that runs the NVT example, and the only Widom example has
`N = 0`. So the comparison splits:
- B1 compares the *chain*, through the mean energy of 50 CO2 in the empty box, which is a direct
  test of guest–guest Lennard-Jones and Ewald with no host contribution at all;
- B0 compares the *estimator*, exactly, against Milestone A, which is already validated against
  kUPS;
- what neither covers is the test particle seeing the other guests. That piece gets a
  brute-force reference (direct lattice sum at a large cutoff) rather than a kUPS comparison.
State this limit in the docs rather than implying the whole milestone was cross-validated.

**R6 — build our RNG so the question in §6 does not arise.** Use explicit counter-based
per-chain streams keyed by (seed, chain index, cycle, move index). That is reproducible
independent of batch size and chunking by construction, so nothing depends on resolving how
JAX's threefry behaves across shapes.

**R7 — test the overlap case explicitly.** kUPS has no designed branch for a non-finite or
zero-probability proposal; rejection of an overlapping insertion falls out of IEEE-754 comparison
semantics. Relying on that by accident is exactly the kind of thing this project does not do: add
a test that a proposal with infinite energy is rejected, and make the code path deliberate.

## Measured: the reciprocal sum does not dominate on GPU

`bench/results/pureadsorb_guestmove_*_20260926_80ea256.json`, RUBTAK 3×3×3 with 50 CO2,
`nk = 4587`, 65,536 move proposals per launch:

| Backend | Precision | Real-space (ns/move) | Reciprocal (ns/move) | Reciprocal share |
|---|---|---|---|---|
| RTX 4070 | Float32 | 733 | 364 | 33% |
| RTX 4070 | Float64 | 11,149 | 6,806 | 38% |
| CPU | Float64 | 17,828 | 69,153 | 79% |

**This refutes the amendment's premise on the backend production runs on.** Real space is roughly
twice the reciprocal sum on the GPU; only on the CPU does the reciprocal term dominate. The
Amdahl ceilings follow directly: a reciprocal sum reduced to zero cost buys 1.50× (Float32) and
1.61× (Float64) on the GPU. Every Ewald-side lever, α included, is bidding for that ceiling, so
lever 1 of the amendment is demoted to last.

**Caveat on the measurement itself.** It uses a batch of ONE framework with 65,536 independent
proposals sharing a single `Sk`, which is an L1 broadcast, not a chain. It therefore measures
kernel cost at full occupancy — what roughly sixty-five thousand concurrent chains would see —
and says nothing about a realistic batch. The ratio is sound because both kernels were measured
the same way; the absolute numbers are not the production figure.

## Rulings from the throughput design panel (2026-09-26)

A read-only panel of seventeen agents (four readers, four designs, eight adversarial judges, one
synthesis) examined the measurement above. Its findings, ruled on:

**P1 — the energy audit is blind to the error it exists to catch. Fix before anything else.**
`audit_energy!` compares the running energy against `total_energy`, but `total_energy` draws its
reciprocal term from the same running `Sk`. Since `ΔU_recip` is identically the change in
`Σ_k pref·|Sk|²` under `Sk ← Sk + ΔS` for *any* `ΔS`, a `ΔS` with the wrong sign, phase or guest
leaves both quantities self-consistently wrong and the audit passes. The task 8 test that caught
a "corrupted ΔU" perturbed a scalar, not a structure factor. The audit must rebuild `Sk` from the
poses, compare element-wise, **overwrite** `state.Sk`, and only then recompute the energy and
reset `state.energy` — in that order, or it certifies a value derived from the thing under test.
Overwriting also removes Float32 drift rather than merely detecting it.

**P2 — a per-guest host-energy cache is an unconditional win, about 1.45×.** The host is rigid,
so guest `i`'s host interaction depends only on its own pose and is invalidated only by its own
move; a move then scans the host once instead of twice. Host–guest work is about 97% of the
real-space outer loop (2 poses × 3078 atoms × 3 sites, against 2 × 49 × 9 for guest–guest). It is
exact arithmetic on an exact identity, and self-checking, because `total_energy` recomputes from
poses and never reads the cache.

**P3 — hoist the rotations out of the k-loop.** `reciprocal_move_delta!` and
`reciprocal_move_delta_energy` call `rotate` inside `for i in eachindex(ks)`, where
`host_guest_realspace_energy` hoists it. LLVM very likely does this already, but "very likely" is
not a measurement, and it contaminates every reciprocal number until settled.

**P4 — a neighbour list is worth about 1.18×, not 4×, and is not approved.** Only 368 of 3078
atoms fall inside the cutoff, which looks like an 8.4× redundancy, but a min-image visit costs
about 42 ps against 519 ps for an in-cutoff pair, so removing visits removes little. A cell list
is dead outright by geometry: the supercell is 36.1 Å across against a 12 Å cutoff, so
cutoff-sized cells give a 3×3×3 grid whose stencil is the whole box. Do not build either until
the visit-versus-pair split is measured on the move kernel specifically.

**P5 — three measurements gate task 5, and none of them is optional.**
1. *Transcendental fraction.* In a bench-local copy, replace `cis(x)` with a deliberately wrong
   `Complex(1 − x²/2, x)` of comparable cost and re-run. The ratio is the sincos fraction. Run it
   after P3 or it measures unhoisted rotations.
2. *Visit versus pair.* Decompose the move kernel's real-space half by setting the Ewald cutoff,
   then both cutoffs, near zero and differencing. This decides P4. The existing E2b shares come
   from an insertion kernel on a different card and must not be borrowed.
3. *Chain sweep.* Sweep the number of systems with one chain each and a private `Sk` per chain,
   reporting cost per move against chain count. This is the production shape, and the number it
   produces decides whether any of the kernel work is worth doing.

## Second-opinion review, 2026-09-27 — corrections and the ranked plan

An independent review read the source, the kUPS source and every measurement. It corrected two
things this plan asserts and found two structural defects nobody had looked for.

### Correction to spec §3.2 — E1's sparsity is only half lost

The design says guests force the full k-vector set because their structure factors are nonzero
everywhere. That is right for the guest–guest term `|S_g|²` and **wrong for the host–guest cross
term**, which is `2·Re(conj(S_host)·ΔS)` and still vanishes wherever `S_host` does. The two
therefore split:

- host–guest cross term: the 190 replication-coupled k-vectors, at the host's `α`, exactly as
  Milestone A already does;
- guest–guest: its own Ewald sum at its own `α_gg`, whose real-space cutoff may run out to half
  the supercell's perpendicular length (about 17.5 Å here against 12 Å), so `k_max` and hence
  `n_k` shrink with it — roughly 1,480 k-vectors.

Net k-work per move: about 1,670 against 4,587, a factor of 2.7, changing no physics. The
cross term of two charge sets splits exactly under Ewald with no self term; the self, exclusion
and net-charge terms attach to the guest–guest sum at `α_gg`.

### Defect 1 — the host is stored once per system

`FrameworkBatch` recomputes and stores `Shost`, the orientation-averaged self term, the hard-core
bound and `kmin` for every system, even when every system holds the same framework. At 4,096
systems that is 4,096 identical copies: about 586 MB on the device and fourteen million `cis`
evaluations per system at build time. This is the whole of the 0.18 s per system build cost.
A `framework_of` indirection with a per-framework memoized build takes the build from 756 s to
seconds and the device footprint to 143 KB, which is small enough to stay in L2 where every
workgroup reads it.

### Defect 2 — Float64 on a GeForce is a hardware mismatch, and the CPU sweep never ran threaded

An RTX 4070 delivers roughly 0.46 TFLOPS of double precision; this laptop's CPU delivers about
0.6. The measured CPU aggregate already matches or beats the GPU at every chain count. The flat
CPU curve in the chain sweep (367–382 ns/move from 1 to 1,024 systems) is the signature of the
KernelAbstractions CPU backend running the whole sweep in one workgroup — **the CPU has never
been measured threaded on the chain shape**, so that row is not yet evidence of anything.

Direction: Float64 on the threaded CPU backend, Float32 on the GPU. The open question is whether
Float32 rounding in `ΔU` biases acceptance; the estimate is about 0.005 kT per move, which the
B1 comparison of `⟨U⟩` in Float32 against Float64 will settle.

### The curve's shape was diagnostic and nobody read it

Cost per move that keeps falling without plateauing is the signature of a latency-bound kernel,
not of insufficient work. The throughput floor for this move is about 0.15 µs; 8.24 µs at 4,096
chains is fifty times off it, and 5.79 ms at one chain is four orders off. Both are the same
defect.

### Addendum, 2026-09-27 — the GPU figures above were not warmed to boost clock

`bench/chain_sweep_bench.jl` (the source of every GPU number in this section, and of the CPU row
below) times each kernel with `Chairmarks.@be(...; seconds = 10, samples = 10, evals = 1)` and no
warm-up loop of its own. Reading `Chairmarks.jl`'s `@be` implementation: because `evals` is given
explicitly, its calibration phase is skipped, and its sampling loop stops at the *first* of "ran
out of time" or "collected `samples` samples" — not the maximum of the two. Every kernel this
script measures finishes 10 calls in well under the 10-second budget, so the loop always exits on
the sample count; the nominal multi-second budget never actually binds. Total recorded launches
per data point are 1 discarded warm-up call plus 10 timed samples, on top of one untimed call the
script makes before `@be` runs — about a dozen launches regardless of `nsys`, on an RTX 4070 that
idles at 210 MHz against a 3,105 MHz boost clock.

That a dozen launches is not enough is independently confirmed by `bench/mc_step_bench.jl`
(written after this plan; see `docs/src/benchmarks.md`), which measured this same card's
clock-ramp directly: `mc_step!` at `nsys=1`, Float64, drops from about 2.8 ms on the first calls
to 220–250 µs at steady state, with the ramp itself taking 100–150 calls. `chain_sweep_bench.jl`
has no wall-clock warm-up loop at all (`mc_step_bench.jl` added one — loop on wall-clock time
until 0.5 s elapses, immediately before timing — specifically because of this measurement), so
every GPU figure quoted above from it (the "5.79 ms"/"8.24 µs" pair the curve-shape argument and
item 1's "Expected" column are built on, and the GPU-vs-CPU comparison in Defect 2) was almost
certainly measured near the idle clock and is **unreliable** — plausibly off by close to the
2.8 ms/220 µs ≈ 11–13× ratio measured above, which matches the roughly 8–10× gap between what
this section quotes and the corresponding rows actually committed in
`bench/results/pureadsorb_chainsweep_neuromancer4070_cuda_{f32,f64}_20260927_0eba49b.json` (e.g.
`f64`, `nsys=1`: 44.9 ms recorded there, not the 5.79 ms — itself the `f32` row — quoted in the
prose above). The CPU row ("367–382 ns/move") is unaffected by the clock question but is a
separate unit error: the committed CPU JSON gives 367–382 *µs*/move, a thousand-fold larger.

None of this changes any conclusion the plan actually acted on: item 1 (the workgroup-per-chain
kernel) was built regardless, and its replacement, `mc_step!`, was re-measured with a correct
wall-clock warm-up and gives 187.2 µs/move at one chain, Float64 (`docs/src/benchmarks.md`) — the
current, trustworthy baseline. This addendum exists so the "5.79 ms → 10–30 µs" and "8.24 µs"
figures above are not mistaken for validated measurements; they were never re-measured warm, and
should not be cited as such.

### Ranked plan

| # | Item | Physics | Expected | Deciding measurement |
|---|---|---|---|---|
| 0 | Measure kUPS NVT at one system | — | sets the target | µs per move |
| 1 | Workgroup per move, strided atoms and k, `@localmem` tree reduction | exact | 5.79 ms → 10–30 µs at one chain; 8.24 → 1–2 µs at 4,096 | chain sweep rerun |
| 2 | `framework_of` indirection, per-framework memoized build | exact | build 756 s → seconds; host bytes ÷4,096 | `t_build`, bytes, 4,096-chain cost |
| 3 | Float64 on the threaded CPU, Float32 on GPU | exact | CPU ~50 µs/move aggregate | CPU sweep with real threading; Float32 vs Float64 `⟨U⟩` in σ |
| 4 | Split `Sk`: host at 190 k, guest–guest at `α_gg` | exact to precision | k-work 2.7× | lattice-sum reference at 1e-10 |
| 5 | Factorized phase tables replacing `cis` | exact | 2–3× on the reciprocal term | swap methodology; `ΔU_recip` agreement |
| 6 | Persistent chains, `Sk` in registers, several moves per launch | exact | removes per-move launch and sync | empty-kernel launch cost first |
| 7 | Unit-cell energy grids, tricubic, clamped wall | **changes the energy model — gated** | host term → ~0 | grid-vs-exact over 10⁵ poses; μ_ex shift in σ |
| 8 | Per-voxel image list (exact fallback to 7) | exact | re-measure first | visit-vs-pair on the new kernel |

Rejected: multiple-try Metropolis (preserves the distribution but buys effective samples, not
throughput, and only when the device is already idle); waste recycling (a variance lever, not a
throughput one); rejection-free and n-fold-way methods (need a finite event set, which continuous
rigid-body moves do not have); event-chain Monte Carlo (exact and better-mixing, but inherently
sequential and its Ewald treatment is a research project).

**Every share measured so far — the `cis` fraction, the visit-versus-pair split — was measured on
the kernel item 1 replaces. Re-derive them on the new kernel; do not carry them forward.**

### Where a durable advantage lives

Each stage of a kUPS move is a separate kernel and a separate round trip through memory, with
ragged data padded to a fixed capacity and masked — which is why sixteen frameworks attempt a
10 GiB allocation. A traced array language cannot express chain state resident in registers
across sequential moves, a per-work-item early exit at the cutoff, ragged per-system loops
without padding, a random number generator fused into the energy kernel, or Float32 at all.
Those asymmetries, not out-tuning them at their own game, are where the advantage is.

## Sixteen-bit storage for host positions — candidate, unmeasured

Half-precision *arithmetic* is not viable here. Float16 carries about eleven mantissa bits, so
roughly 1e-3 relative error; against `kT = 0.0257 eV` that is about 0.04 kT of error per move on
a 1 eV energy, eight times the Float32 figure that already needs validating. The reciprocal sum
is worse: 4,587 terms with heavy cancellation accumulated at 1e-3 per term is meaningless.
BFloat16, with eight mantissa bits, is worse again.

Float16 is also the wrong sixteen-bit format for positions: eleven bits across a 44 Å cell is
0.02 Å, and Lennard-Jones goes as r^-12, so at a 3 Å contact that is an 8% energy error.

**Sixteen-bit fixed point does work.** Host atom positions are rigid and bounded by the cell, so
each coordinate can be stored as a 16-bit integer fraction of the cell vector: 65,536 levels over
44 Å is 0.0007 Å, finer than the CIF's own coordinates. Decode to the working precision in the
kernel; arithmetic is unchanged.

The payoff is memory traffic, which is where the second-opinion review located the real limit
(about 10,700 global loads per work-item). Halving the host position array halves the dominant
stream and doubles what fits in L2 — and once the host is deduplicated to one copy per framework
(item 2 of the ranked plan), cutting 143 KB to 72 KB makes L2 residency comfortable rather than
marginal.

Ordering: worth nothing until the workgroup rewrite makes the kernel throughput-bound instead of
latency-bound, so it sits behind items 1 and 2. Deciding measurement: bytes moved per move and
cost per move before and after, on the restructured kernel, plus energy agreement against the
Float64 positions to the existing 1e-10 oracle tolerance.
