# PureAdsorb.jl Milestone C (grand canonical Monte Carlo) Implementation Plan

Design: `docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md` (approved 2026-09-27).

## Global constraints

Unchanged from Milestone B, restated because they are the contract:

- Julia only; no Python in the package. kUPS runs via `uv` from `~/src/kups` as a reference
  stopwatch and oracle under the standing narrow exception, with `JULIA_GUARD=off` on those
  commands only.
- Single-source KernelAbstractions kernels, generic over `Float32`/`Float64`, compiling for CPU,
  CUDA, ROCm and Metal. Metal is Float32 only.
- No throwing branches in kernels; no multi-array `eachindex`, no `floor(Int, x)`, no `mod`/`div`
  by a runtime value. No `@inbounds` without a measurement and a bounds-checked test.
- Allocation-free and type-stable under the StrictMode audit, both precisions.
- **GPU timing warms on wall-clock time**, never a fixed call count: the card idles at 210 MHz
  against a 3,105 MHz boost and needs 100–150 sustained calls. A single warm-up call once gave a
  15× too-slow reading here and sent a whole dispatch after a phantom bottleneck.
- Every benchmark datapoint saved as JSON in `bench/results/` and recorded in its README.
- TestItems.jl; Runic; `Assisted-by` trailer; one writer in the tree at a time.
- Docs and comments state present-tense facts, never history or task numbers.

## kUPS conventions to reproduce

Read from the source before relying on any of it; `mcmc_rigid.py`, `mcmc/moves.py`,
`mcmc/probability.py` (`make_muvt_probability_ratio`, `probability.py:389`), `mcmc/fugacity.py`,
`core/data/buffered.py`. Milestone B's "Task 1 findings" section already records the move
weights, the cycle rule and the acceptance comparison; C adds `exchange_prob` (default 1/2, and
`ExchangeMove` splits 50/50 between insertion and deletion), `max_num_adsorbates`, and their
multicomponent Peng–Robinson implementation.

## Tasks

### Task 1 — Peng–Robinson fugacity
`src/fugacity.jl`. Fugacity coefficient from `tc`, `pc`, `omega`, already carried by `Guest` and
currently unused. Host-side; it runs once per system per run, so clarity beats speed.

**Say explicitly what happens when the cubic has three real roots** (below the critical point):
pick the vapor root and document it, or refuse. Do not let the root choice be an accident of the
solver.

Acceptance: agreement with kUPS's own fugacity module over a grid of pressures and temperatures
spanning sub- to supercritical for CO2 and methane, to 1e-10 relative; the ideal-gas limit
`φ → 1` as `P → 0`; and a test at a state point with three real roots that pins the choice.

### Task 2 — variable occupancy
Extend `SystemState` with a per-system capacity and occupancy count. Insertion writes at slot
N+1 and increments; deletion swaps the last occupant into the freed slot and decrements, which
keeps occupancy contiguous without shifting. Every energy loop runs to the occupancy, not the
capacity, for now — §7 of the design leaves the divergence question to measurement.

**Exceeding capacity fails loudly**, and the audit gains an occupancy-against-capacity check. A
chain that silently saturates samples a truncated distribution and looks healthy.

Acceptance: round-trips to device unchanged; the swap-delete preserves the multiset of occupied
poses; StrictMode clean; existing Milestone B tests unaffected.

### Task 3 — insertion and deletion moves
`src/moves.jl` gains the two μVT moves with the acceptance ratios from design §1.1. `ΔU` for an
insertion is Milestone A's insertion energy against host and all present guests; for a deletion
it is the negative of the removed guest's interaction with everything else. The structure factor
changes by exactly `±S_guest(k)`.

**The combinatorial factors are the whole risk.** `V/((N+1)kT)` and `N kT/(fV)` must be each
other's inverse. Write them once, derive the second from the first in a comment, and test the
inverse relation directly rather than trusting two hand-typed expressions.

Acceptance: a move and its inverse give `ΔU` summing to zero and restore `Sk`; the existing
energy audit passes over a long chain with insertions and deletions and still catches an injected
corruption; StrictMode clean; runs on CUDA.

### Task 4 — the ideal-gas limit
**Do this before the driver, not after.** With guest–host and guest–guest interactions switched
off, the chain must reproduce `⟨N⟩ = fV/kT` and Poisson fluctuations `⟨N²⟩ − ⟨N⟩² = ⟨N⟩`.

This isolates the combinatorial factors from the physics and is the single most valuable test in
the milestone — a GCMC chain with correct energies and a wrong acceptance ratio converges
smoothly to the wrong loading, and nothing in our existing audit would notice. Choose the
tolerance from the sampling error rather than a round number, and make the test fail if the
factors are perturbed.

### Task 5 — the GCMC driver
`src/gcmc.jl`. Warmup then production; the Milestone B move set plus insertion and deletion, with
`exchange_prob` selecting between them; block averaging over cycles. Reports per system: mean
loading, mean energy, the fluctuation `q_st` from design §4, each with a block-averaged error.

Acceptance: disabling insertion and deletion reproduces Milestone B **exactly** for a fixed seed;
the audit runs and passes; the capacity diagnostic reports the maximum loading reached.

### Task 6 — isotherms on the batch axis
A batch indexed by (framework, pressure, replica), with framework deduplication meaning a
multi-pressure sweep on one host costs one host. A convenience constructor for "this framework,
these pressures, this many replicas", and a result type carrying loading against pressure.

Acceptance: a 50-point isotherm on one framework builds in the time one framework takes (the
dedup work makes this about 0.2 s, so check it) and runs as one batch; per-point errors come from
replicas as well as from blocks.

### Task 7 — Henry's law crosscheck
At low pressure the isotherm must be linear with slope equal to Milestone A's Henry coefficient,
computed by a completely independent route. Two methods, one number.

Acceptance: agreement within combined statistical error, with the pressure range chosen so the
isotherm is actually in its linear regime — and state how that range was chosen.

### Task 8 — detailed balance between adjacent loadings
On a small system, the ratio of time spent at N and N+1 must match the analytic ratio. This
catches an insertion and deletion pair that are individually plausible but not each other's
inverse. Demonstrate the test has discriminating power by perturbing a factor and showing it
fails, as the Milestone B detailed-balance test does.

### Task 9 — validation against kUPS and RASPA
kUPS GCMC (`examples/mcmc_rigid.yaml`) on the same case: loading and energy in combined standard
errors. Then the published RASPA isotherm for methane in IRMOF-1 — an external check against
neither our code nor kUPS. Check first whether we have an IRMOF-1 CIF our P1-only reader accepts;
if not, say so rather than quietly substituting a different material.

### Task 10 — throughput and documentation
Measure against kUPS GCMC, warm, with a chain-count sweep, on the 4070. Then the docs: a
grand-canonical section in `theory.md` derived rather than asserted — the ensemble, why fugacity
removes the awkward constants, where the combinatorial factors come from, the fluctuation
formula for `q_st` and why it converges slowly — plus benchmarks and validation sections and an
isotherm figure.

## Ordering

Tasks 1 and 2 are independent and can run in either order. Task 3 needs both. **Task 4 gates task
5**: do not build the driver until the ideal-gas limit passes. Tasks 6, 7 and 8 need 5. Task 9
needs 6. Task 10 is last.

## Risks

- **The combinatorial factors.** Mitigated by task 4 running before the driver exists.
- **Capacity truncation.** Mitigated by the loud failure and the diagnostic.
- **Slow convergence of the fluctuation `q_st`.** It may simply need more cycles than is practical;
  if so, report it with its error and say it is slow rather than hiding it.
- **The IRMOF-1 comparison may be blocked** by file format rather than physics.
