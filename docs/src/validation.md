# Validation

## Oracles

Each test item below is a `@testitem` under `test/`, run through TestItemRunner.

| What | Where | Check |
|---|---|---|
| Madelung constant of NaCl | `ewald_tests.jl` | Ewald energy of an ionic NaCl lattice against the analytic Madelung constant −1.747565 (per ion pair), at two cutoff/precision combinations; rtol 1e-5 |
| α-independence | `ewald_tests.jl` | The same charge configuration's Ewald energy, computed at three different cutoffs (hence three different α), agrees to rtol 1e-6 |
| `erfc_dev` vs `SpecialFunctions.erfc` | `ewald_tests.jl` | The Chebyshev-fit `erfc_dev` matches `SpecialFunctions.erfc` for x ∈ [0, 6] (rtol 1e-11), at x = 8 (rtol 1e-9), and for a negative argument |
| k-vector enumeration vs brute force | `ewald_tests.jl` | On the RUBTAK triclinic cell, `kvectors`' lattice-vector-length bound is checked against an explicit brute-force search over a wider integer range: equal vector count and equal total weight; also confirms the perpendicular-length bound would under-count on this cell |
| Minimum image vs brute-force image search | `cell_tests.jl` | `minimum_image` against the minimum over the ±2 image shell, 2000 random displacements in a scaled triclinic cell, whenever the true minimum is inside the cutoff |
| Insertion energy vs full-system Ewald difference | `energy_tests.jl` | `insertion_energy` for one random pose against a from-scratch Ewald energy difference (system + guest) − (system alone), decomposed by hand into LJ + Coulomb − self − exclusion − net terms; rtol 1e-8. A second item repeats this with independent LJ (6 Å) and Ewald (12 Å) cutoffs |
| Empty box | `widom_tests.jl` | In an empty framework, μ_ex ≈ 0 (atol 1e-12), K_H ≈ V/k_BT and q_st ≈ k_BT (ideal-gas limits); a Float32 variant checks every result field stays `Float32` |
| Remainder-sized insertion counts | `widom_tests.jl` | `ninsert` values that don't divide evenly into `nblocks`, for one and for two systems, leave every block with at least one sample and finite standard errors |
| Single LJ atom vs radial integral | `widom_tests.jl` | One fixed LJ atom, 4,000,000 insertions (20 blocks); the resulting ⟨W⟩ is compared to `QuadGK`'s numerical integral of `1 - exp(-βu(r))` over r, requiring agreement within 4 standard errors |
| CPU vs GPU agreement | `gpu_tests.jl` (tag `:gpu`) | The same random poses (same seed) run through `widom` on the CPU backend and on a functional CUDA or AMDGPU backend must agree to rtol 1e-10 in μ_ex, K_H and q_st |
| Generic indexing | `generic_tests.jl` | `ForceField`, `tail_delta`, `kvectors`/`structure_factor` and `ewald_energy` accept `OffsetArray`- and `view`-wrapped inputs; mismatched-length inputs raise `DimensionMismatch` |
| StrictMode audit | `bench/audit.jl` | `insertion_energy` (Float64 and Float32), `minimum_image`, `rotate` and `erfc_dev` are gated on `:typestable` and `:noalloc`; `STRICT_MODE=fast` (default) is a value-free heuristic scan, `STRICT_MODE=full` backs the same guarantees with AllocCheck and JET as the pre-merge gate |
| Cross-code reference | `reference_tests.jl` (tag `:slow`) | RUBTAK + CO2 against the kUPS numbers below, within 3 combined standard errors |
| Peng–Robinson fugacity vs kUPS | `fugacity_tests.jl` | CO2 and methane, temperatures 0.6–2× critical, pressures 1 Pa to 5× critical pressure: `Z` and `f` agree with kUPS's own module to 1e-10 relative; the ideal-gas limit `φ → 1` as `P → 0`; a three-real-root state point (below the critical temperature) pins the LIQUID-branch (smallest-root) selection against kUPS |
| Ideal-gas anchors (Loschmidt, kUPS's own example) | `ideal_gas_tests.jl` | With guest–host and guest–guest interactions switched off (a single ε=0 LJ type, `tail=false`, zero guest charges — checked directly, not assumed), an ideal-gas μVT chain's mean and variance of `N`, from 1500–3000 independent single-system replicas, match `⟨N⟩ = fV/k_BT` and Poisson `\mathrm{Var}(N)=⟨N⟩` within 5 standard errors at two PHYSICALLY anchored state points computed independently of the code under test: Loschmidt's number (1 atm, 273.15 K, ⟨N⟩=1 in 37,219 ų) and kUPS's own example point (1e4 Pa, 298.15 K, RUBTAK 3×3×3's 61,457 ų giving ⟨N⟩=0.1493) |
| The fugacity path exercises the equation of state | `ideal_gas_tests.jl` | Same ideal-gas chain, real CO2 critical constants, 298 K and 5e6 Pa (φ well below 1): the chain's mean `N` matches `φPV/k_BT` within 5 SE and misses the "used raw `P` instead of `f`" alternative by more than 5 SE — a state point where the two predictions differ by about 20%, not a regime where they coincide by chance |
| R2 merged Henry's-law / detailed-balance test | `henry_detailed_balance_tests.jl` (tag `:gpu`) | With real interactions on, `P(N+1)/P(N) = (fV/((N+1)k_BT))\langle\exp(-\Delta U_{\mathrm{ins}}/k_BT)\rangle_N` (`docs/src/theory.md`'s acceptance-ratio derivation) checked at every well-sampled loading `N` in one run, RUBTAK 3×3×3 + CO2 at 500 Pa (inside the Henry-linear regime, confirmed by a separate two-pressure isotherm check): every loading agrees within 4 combined standard errors, and the `N=0` value agrees with Milestone A's independently-computed Henry coefficient within 5 combined SE. A companion test perturbs the chain's own insertion/deletion fugacity by ×1.5 (leaving the comparison's target fugacity unperturbed) and confirms every well-sampled loading then fails by more than 3σ, with the recovered ratio matching the injected 1.5× factor — demonstrating the test's own discriminating power, not just that it can pass |

### CPU-vs-GPU relative differences

Measured on the RTX 3050 (CUDA backend), same seed, same insertion count, against the CPU
backend:

| Quantity | Relative difference |
|---|---|
| μ_ex | 4.0e-16 |
| K_H | 1.4e-15 |
| q_st | 2.2e-16 |

All at the level of a few floating-point epsilons — the two backends run the same generic
Julia energy and reduction code, compiled per backend by KernelAbstractions, so this is
agreement to the last bit rather than a statistical tolerance.

## Cross-code comparison against kUPS

RUBTAK 3×3×3, CO2, 298.15 K, 12 Å real-space and Ewald cutoffs, Ewald precision 1e-6, one
million insertions, seed 42, 20 blocks. The host CIF, force field and guest YAML files are
copied unchanged (byte-identical) from kUPS's own examples (`data/NOTICE`).

| Quantity | PureAdsorb | kUPS | Combined SE | Deviation |
|---|---|---|---|---|
| μ_ex (eV) | −0.1436470 ± 0.0002190 | −0.1432013 ± 0.0002226 | 0.0003122 | 1.43 σ |
| K_H (Å³/eV) | 6.410616e8 ± 5.4634e6 | 6.300389e8 ± 5.4584e6 | 7.723e6 | 1.43 σ |
| q_st (eV) | 0.2629071 ± 0.0002836 | 0.2623186 ± 0.0003429 | 0.0004450 | 1.32 σ |

PureAdsorb's numbers above come from running the exact case the package's own `:slow`-tagged
reference test runs; kUPS's are `test/reference/rubtak_co2_kups.json` (kUPS commit
`e183c9aae820b8c98333f8f9ac27a7ac9cfa213d`, RTX 3050, driver 615.71.09).

All three quantities agree within about 1.4 combined standard errors. Because all three are
computed from the same set of insertions in each code, their deviations from each other are
correlated rather than independent draws — a single set of Widom samples that happens to run
slightly high or low in one code moves μ_ex, K_H and q_st together. The reference test's
acceptance criterion is 3 combined standard errors, so this run passes with room to spare.

## Milestone B (NVT) validation ladder

Internal checks (the energy audit, `audit_energy!`, and reversibility) come first; they find
bugs that cross-code agreement can hide. `audit_energy!`'s tolerance is derived from Higham's
recursive-summation bound applied to the actual per-move rounding scale that accumulates in
`SystemState.sk_abs_accum`/`energy_abs_accum`, not to the running total's own magnitude, which
cancellation can make far smaller than the terms that produced it (`src/audit.jl`). Measured
false-positive rate on a clean chain over 15 fresh seeds at 10, 100, 500, 3,000, 5,000 and 10,000
accepted moves, both precisions: 0/15 at every checkpoint.

**B0 — `N = 0` reproduces Milestone A.** `run_nvt!` at `N=0` agrees with `widom_singlephase` to
`rtol = 1e-10` in μ_ex, K_H and q_st — not bit-for-bit: this milestone's cycle-based block
averaging groups the same per-insertion Boltzmann weights differently from Milestone A's
insertion-based blocking, so the two estimators' totals agree to floating-point rounding rather
than bit for bit, the same standard this project already applies to other independently-grouped
floating-point sums.

**B1 — pure guest–guest, kUPS's `examples/nvt_co2_pressure_test.yaml`.** 50 CO2 in a 30 Å cubic
box whose only host site is non-interacting (`exchange_prob: 0`), so the host contributes
identically zero and the comparison isolates guest–guest Lennard-Jones and Ewald. Mean total
energy agrees to 0.05 combined standard errors (both codes' own block-averaged SEM; kUPS's
`optimal_block_average` and this package's fixed block count are different rules, so neither
claims agreement tighter than the looser of the two). Acceptance rates differ in translation
(82.8% here vs 54.6% for kUPS) because kUPS's step sizes adapt toward a 50% target continuously
(R4) while this package freezes them after warmup — a known, deliberate divergence, not an error;
rotation and reinsertion rates land close (kUPS never actually tunes reinsertion's step, Task 1
finding #2).

**Float32 vs Float64 (B1).** Same case, same seed: mean energy differs by 0.68 combined standard
errors — no statistically significant Float32 bias detected. The design's own estimate was about
0.005 kT of rounding error per move; this measurement, not the estimate, is what decides whether
the Float32 row is publishable.

**B2 — host and guests together.** RUBTAK 3×3×3 with 50 CO2, `exchange_prob: 0`, written in
kUPS's own config schema. kUPS's own reported energy is the FULL system energy (it computes
`U_host-host`; this package's `total_energy` never does, since that term is a constant that
cancels in every difference and plays no role in an absolute value either). Comparing against
kUPS's own `energy(N=50) - energy(N=0)` for the same host, mean guest-dependent energy agrees to
0.05 combined standard errors.

**Widom along the chain has no kUPS counterpart (R5).** No kUPS example runs N-guest NVT and
Widom together: its Widom entry point (`mcmc_widom.py`) uses a different config class from the
one that runs the NVT example (`mcmc_rigid.py`), and its only Widom example has zero guests. This
piece is validated instead against an independent, non-incremental oracle: a literal Ewald sum
(`ewald_energy`) over every host-plus-guest position, computed once before and once after
appending the test guest — never touching the incremental `Sk` machinery under test. A single
fixed pose does not match exactly: `FrameworkBatch.constant_offset` folds in an
orientation-AVERAGED reciprocal self term (`self_mean`), not the exact per-orientation value, so
any one insertion carries a real error bounded by `self_term_halfrange`. Since `self_mean` is
defined as exactly the average of the true per-orientation value, this error has zero mean and
washes out in a Boltzmann-weighted average over enough random poses — the same quantity Widom's
own μ_ex accumulates — so that average, not a single pose, is what this validates: agreement to
better than 0.001 combined standard errors for both an empty-box-plus-guests and a
RUBTAK-plus-guests configuration (`test/nvt_tests.jl`).

Every number and its provenance is in `bench/results/README.md`'s "Milestone B validation ladder"
section and the JSON files it cites.

## Milestone C (GCMC) validation ladder

A grand canonical chain can be wrong in a way no energy check detects: the energies are computed
correctly and only the ACCEPTANCE RATIO is wrong, which produces a stable, audit-passing chain
that converges to the wrong loading (`docs/src/theory.md`'s "Units: two systems meeting at one
boundary" and "Insertion and deletion are not individually detailed-balanced" both describe a
real instance of exactly this). The ladder below is built around the particle number rather than
around energy for that reason.

**C0 — disabling exchange reproduces Milestone B exactly.** `run_gcmc!` at `exchange_prob=0`
against `run_nvt!`, same seed, RUBTAK 3×3×3 + CO2 with 5 and 3 initial guests in two systems:
`refpoints`, `orientations`, `Sk`, `energy`, `accepted` and `attempted` agree bit for bit, and both
drivers' own reported energy/energy-error agree bit for bit too (`gcmc_tests.jl`). This pins the
NVT-move machinery `run_gcmc!` shares with `run_nvt!` before any μVT-specific check runs.

**C1 — the ideal-gas limit against physical anchors, not the code's own inputs.** With guest–host
and guest–guest interactions switched off — a single ε=0 Lennard-Jones type, `tail_correction =
false`, zero guest charges, confirmed directly (every term `exchange_constant_term` adds is
exactly zero, and a 5000-attempt chain's running energy stays exactly `0.0` throughout, not merely
small) — 1500–3000 independent single-system replicas' final occupancy, treated as iid draws of
the stationary distribution, give a sample mean and variance matching `⟨N⟩ = fV/k_BT` and the
Poisson relation `\mathrm{Var}(N)=⟨N⟩` within 5 standard errors, at TWO state points computed as
literal, hand-transcribed numbers rather than by calling any of this package's own code
(`ideal_gas_tests.jl`'s own comment: a self-referential target would be blind to a uniform scale
error in `f`, exactly the class of bug a missing `PASCAL` factor is): Loschmidt's number (1 atm,
273.15 K, ⟨N⟩=1 in 37,219 ų) and kUPS's own `examples/mcmc_rigid.yaml` state point (1e4 Pa,
298.15 K, RUBTAK 3×3×3's 61,457 ų giving ⟨N⟩=0.1493, `k_BT=0.025693` eV, pinned as a literal
too). A worked, one-off demonstration (not a permanent test, to avoid a scratch bug living on in
`src/`) confirms this ladder rung has teeth: substituting `N` for `N+1` in
`log_insertion_prefactor` and rerunning the Loschmidt case with identical seeds shifts the sample
mean by an EXACT `+1.0` (a ~39σ effect) while leaving the variance bit-for-bit unchanged — an
algebraic consequence of evaluating the correct prefactor one guest count too low, not a vague
"gets worse".

**C2 — the fugacity path exercises the equation of state.** Every other exchange test in this
ladder runs at a pressure low enough that the fugacity coefficient `φ ≈ 1`, so a caller that
accidentally passed raw pressure instead of `peng_robinson_fugacity`'s `f` would pass unnoticed.
CO2 at 298 K and 5e6 Pa has `φ` well below 1 (Peng–Robinson, confirmed `< 0.8` before the chain
even runs), so the ideal (`PV/k_BT`) and real (`φPV/k_BT`) predictions for `⟨N⟩` differ by more
than 20% — a many-σ discriminator. The same ideal-gas replica machinery, real CO2 critical
constants, matches the REAL prediction within 5 standard errors and misses the ideal one by more
than 5 (`ideal_gas_tests.jl`).

**C3 — R2, the merged Henry's-law / detailed-balance test.** With interactions on, RUBTAK 3×3×3 +
CO2 at 500 Pa (confirmed inside the isotherm's Henry-linear regime by a separate two-pressure
check: `loading/pressure` at 200 and 500 Pa matches the `N→0` slope `widom`'s own Henry
coefficient predicts, within 5 combined standard errors), 64 replica chains run together with
Widom test-particle insertion along each chain (Milestone B's own trick, generalized to whatever
occupancy the chain currently holds). Every sample is binned by the chain's occupancy `N` at the
moment it was taken, giving, per `N`, both the visit-count ratio `P(N{+}1)/P(N)` and the
Widom-along-chain average `⟨\exp(-\Delta U_{\mathrm{ins}}/k_BT)⟩_N`
(`docs/src/theory.md`'s own R2 formula, `P(N{+}1)/P(N) = (fV/((N{+}1)k_BT))⟨\exp(-\Delta
U_{\mathrm{ins}}/k_BT)⟩_N`): every well-sampled loading agrees within 4 combined standard errors
(a pooled-ratio-plus-jackknife estimator throughout, avoiding the Jensen-inequality bias a
per-block ratio would carry), and the `N=0` value — expressed as a Henry coefficient — agrees with
Milestone A's `widom`, run independently on the pristine framework with its own RNG stream, within
5 combined standard errors. This one test ties Milestone C to Milestone B under the real
potential, with a closed-form target at every loading rather than only at `N=0`.

A companion test demonstrates C3's own discriminating power (mirroring how the Milestone B ladder
demonstrates the energy audit's): running the identical setup with the chain's OWN
insertion/deletion fugacity multiplied by 1.5, while the comparison's target fugacity stays the
true, unperturbed value, makes every well-sampled loading fail by more than 3σ, with the
recovered `\mathrm{LHS}/\mathrm{RHS}` ratio matching the injected 1.5× factor to within 0.15 — the
test does not merely pass on correct code, it fails in the expected, quantitative way on
incorrect code (`henry_detailed_balance_tests.jl`).

**Audit false-positive rate on CUDA.** The same `audit_energy!` tolerance Milestone B's own ladder
validates on the CPU was re-measured on CUDA specifically, where the device-resident chain state
is a genuinely separate allocation from the host's rather than an alias of it: 224 (Float64) and
256 (Float32) trials per checkpoint, at 1,000/5,000/20,000 accepted moves, RUBTAK 3×3×3 + CO2. A
missing device-to-host sync of the audit's own running-error accumulators gave false-positive
rates up to 39.1% at 1,000 moves (dropping toward 0% as more accumulated moves diluted the effect
of the stale, all-zero accumulator); after syncing them and scaling the audit's recompute-side
tolerance by the number of terms `total_energy` actually sums, the false-positive rate is 0.0% at
every checkpoint, both precisions
(`bench/results/pureadsorb_audit_tolerance_falsepositive_neuromancer4070_cuda_20260927_7f1032f.json`).

Every number above and its provenance is in `bench/results/README.md` and the JSON/test files it
cites.
