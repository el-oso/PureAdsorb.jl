# PureAdsorb.jl Milestone B (NVT Monte Carlo) Implementation Plan

Design: `docs/superpowers/specs/2026-09-26-milestone-b-nvt.md` (approved 2026-09-26).
Status: draft. One change to the approved design needs sign-off before task 1 starts.

## Change to the approved design — needs approval

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

## Open questions carried from the design

- Whether Float32 is sound for a long chain; the energy audit answers it.
- Whether the reciprocal sum dominates once the k set grows 24×; measured in task 10, and it may
  redirect the work.
