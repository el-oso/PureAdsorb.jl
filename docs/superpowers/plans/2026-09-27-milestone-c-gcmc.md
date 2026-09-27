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

## Second-opinion review, 2026-09-27 — five errors, and the ideal-gas gate is not sufficient

An independent review read the design, the plan, the landed code, the in-progress task-3 diff and
the kUPS source. Findings, in the order they should be fixed.

**E1 — Units. There is no pascal conversion anywhere, and it is a factor of 1.6e11.**
`peng_robinson_fugacity` returns `f` in pascals; `batch.volumes` is Å³ and `kT` is eV, so
`log_insertion_prefactor(f, V, kT, N)` in the working tree mixes all three. kUPS converts before
the equation of state (`application/mcmc/data.py:427-431`, `PASCAL = 1/(METER³·e)`). The factor is

    PASCAL = 1 / (1e30 * 1.6021766208e-19) = 6.241509e-12  eV Å⁻³ Pa⁻¹

Left uncorrected the insertion prefactor is 1.6e11 too large: **every insertion is accepted, the
chain climbs to capacity, and the energy audit passes.** Add `PASCAL` to `src/constants.jl` beside
`KB`, convert exactly once at the μVT entry point, and document the unit on the `f` argument.
Peng–Robinson is scale-invariant in pressure (only `P/pc` enters), so the conversion belongs at
the acceptance boundary, not inside `fugacity.jl`.

**E2 — Insertion and deletion are not individually detailed-balanced, and the spec says they are.**
Design §1.1 and `src/moves.jl`'s header carry Milestone B's argument that each move satisfies
detailed balance individually, so any fixed schedule preserves the target. That holds for
translation, rotation and reinsertion and **fails for insertion or deletion alone** — an
insertion cannot be reversed by an insertion. Only the mixture `½K_ins + ½K_del` is
π-preserving. A deterministic alternation samples the wrong loading with a passing audit
(worked counterexample: ideal gas at `fV/kT = 0.1` gives a stationary ratio of 0.009 instead of
0.1). Requirements: the choice between insertion and deletion is a **fresh fair random draw every
launch**, `p_ins = p_del` is an invariant the prefactors depend on, and the "any fixed sequence"
argument must be restated as applying to the NVT moves and to the exchange pair *as one move*.
Task 8's discriminating test should perturb exactly this.

**E3 — `q_st` sign: kUPS's GCMC analyzer is the negative of its own Widom analyzer.**
`application/mcmc/analysis.py:122-126` computes `cov/var − kT`; its Widom path
(`analysis.py:326-332`) computes `kT − ⟨ΔU·W⟩/⟨W⟩`. Ours follows the Widom convention, so task 9
must compare against `−hoa_kUPS`. `U` is the configurational energy, exactly `total_energy`.
Use a pooled estimator with a jackknife over blocks, not kUPS's per-block ratio (which carries
the Jensen bias its own docstring warns about). Convergence goes as `(1−ρ²)/(n ρ²)` with
`ρ = corr(U,N)`; **measure ρ per system** — below 0.5 expect more than 4× the cycles the loading
needs.

**E4 — `insertion_constant_term` is cached once per run and depends on N.**
`src/nvt.jl:44-63` folds the tail correction (N² convention) and the net-charge term into a
constant evaluated once before the cycle loop. Correct for NVT, wrong the moment N changes.
Insertion needs `tail(N+1) − tail(N)` at the chain's current N. Compute it in-kernel from
`occupancy[n]`; for CO2 the net-charge piece is zero, so the tail term is what would silently
drift.

**E5 — kUPS's own example is mislabelled.** `examples/mcmc_rigid.yaml` says
`pressure: 10_000  # Pa (10 bar)`; 1e4 Pa is 0.1 bar. That run is also 100% exchange moves
(the other three probabilities are zero). Task 9 must match both. At that pressure the fugacity
coefficient is about 0.9995, so **the kUPS GCMC comparison does not exercise the equation of
state at all** within statistical error.

**P1 — Deletion at N = 0 and insertion at capacity.** `log_deletion_prefactor(…, 0)` evaluates to
`-Inf`, so rejection happens by the IEEE accident ruling R7 forbids relying on; reject explicitly.
Worse: a kernel cannot throw at capacity, so if it merely rejects, the chain samples a truncated
ensemble and **`audit_energy!`'s `occupancy <= capacity` check is vacuous** — nothing can ever
exceed it. Replace with a per-system `capacity_hits` counter written by the kernel and checked
host-side, throwing on nonzero. Test: an ideal-gas chain with `fV/kT = 20` and capacity 25 must
abort, not finish.

**P2 — Cycle length must be recomputed each cycle** from the live maximum occupancy, as kUPS does
(`mcmc_rigid.py`'s `LoopPropagator` evaluates it on the current state). `run_nvt!` fixes it once
before the loop; copying that into the GCMC driver biases sampling density as N drifts.

**P3 — `insert_guest!`'s `host_energy_new` defaults to zero** (`src/state.jl:212`). A caller that
forgets it corrupts the cache silently until the next audit. Make it required.

**R1 — The ideal-gas gate as specified cannot catch E1.** Asserting `⟨N⟩ = fV/kT` with the
expected value computed from the same `f`, `V` and `kT` the code consumes is self-referential: it
pins the combinatorial factors, and a uniform scale error in `f` passes it. **Assert a physical
anchor instead** — Loschmidt (1 atm, 273.15 K, ⟨N⟩ = 1 in 37,219 Å³), or the kUPS example
(1e4 Pa, 298.15 K, RUBTAK 3×3×3 at V = 61,457 Å³ gives ⟨N⟩ = 0.1493, kT = 0.025693 eV).
"Interactions off" must also zero the tail, net-charge and per-guest self/exclusion terms.

**R2 — A sharper test that merges tasks 7 and 8.** With interactions on,
`P(N+1)/P(N) = (fV/((N+1)kT))·⟨exp(−ΔU_ins/kT)⟩_N`, where the average is Milestone B's
Widom-along-chain at loading N. At N = 0 this is Henry's law exactly, so one test has a
closed-form target at every loading and ties C to B under the real potential.

**R3 — To exercise the equation of state**, run the ideal-gas chain at CO2, 298 K, 5e6 Pa, where
the fugacity coefficient is well below 1, and assert `⟨N⟩ = φPV/kT`. A 20–30% effect is a
many-σ discriminator in a short run.

**Sound, confirmed:** the acceptance ratios themselves are correct and exact inverses, matching
kUPS's `LogFugacityRatio`, *given* E1 and E2. The rigid-body orientational factor **cancels**,
because the equation-of-state fugacity refers to the same rigid molecule's ideal gas and the
orientation proposal is exactly uniform on SO(3) — both Λ and the rotational partition function,
symmetry number included, drop out. **Warning attached**: if the hard-core stage is ever used to
*resample* an insertion pose until it clears the host, the proposal stops being uniform and the
ratio needs a correction; using it only to short-circuit the energy of a pose that is then
rejected as a normal attempt is fine. `state.jl`'s swap-delete is correct in every field.

**Throughput:** looping to `occupancy` rather than capacity is warp-uniform under
workgroup-per-chain, so the divergence worry is probably moot — measure at `nsys = 1024` with N
uniform over 0..50 against all-50 and all-0. The real costs are that half of all exchange attempts
are insertions with no host-energy cache, and that a batch-wide cycle length set by the maximum N
wastes work on low-pressure chains in a mixed-pressure isotherm batch.
