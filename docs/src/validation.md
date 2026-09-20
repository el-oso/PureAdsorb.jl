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
