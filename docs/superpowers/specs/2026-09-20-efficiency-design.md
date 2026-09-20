# PureAdsorb.jl — kernel efficiency design

Make one Widom insertion cheaper without changing its physics, in ways that rely on
data-dependent work sizes and ragged per-framework data.

## Measurements this design rests on

CO2 in RUBTAK 3×3×3 (3078 atoms, 12 Å cutoffs, Ewald precision 1e-6, 298.15 K), commit c910867.

| Quantity | Value | How measured |
|---|---|---|
| Reciprocal sum's share of kernel time | 21% (Float32), 23% (Float64) on an RTX 3050; 43% on one CPU thread | kernel timed with the full and with an empty reciprocal table |
| k-vectors in the reciprocal table | 4587 | `length(batch.ks)` |
| k-vectors whose host structure factor is nonzero | 190 (those with every integer coefficient divisible by the replication factor 3); `|S_host| ≤ 1.8e-12` on the other 4397 | integer coefficients `round(inv(B)·k)` |
| Cross-term energy carried by the other 4397 | ≤ 1.7e-11 kT | `KE Σ pref·2|S_host|·Σ|q|` |
| Guest self term `KE Σ_k pref_k |S_g(k)|²` over orientations | CO2: mean 0.2032 kT, max−min 2.0e-5 kT; water-like 3-site guest: mean 0.9335 kT, max−min 2.9e-5 kT | 2000 random orientations |
| Host atoms inside the 12 Å cutoff of a site | about 12% | sphere volume over supercell volume |
| Insertions with Boltzmann weight exactly 0.0 | 68% (Float64), 78% (Float32) | 40,000 insertions |
| Insertions with any site within 1.5 Å / 2.0 Å of a host atom | 61% / 78%, carrying 1e-231 / 1.8e-14 of the summed weight | same run |

Consequences: 96% of the reciprocal loop multiplies by zero; the real-space loop visits eight
times more atoms than the cutoff needs; most insertions are decided by one close contact.

## Requirements (status: ☐ open, ☑ done)

### E1 — sparse reciprocal table
- ☑ `replicate` records its replication factors in the `Framework` (field `replication::NTuple{3,Int}`, `(1,1,1)` for an unreplicated framework; `replicate` of a replicated framework multiplies them).
- ☑ `FrameworkBatch` stores only the host-coupled k-vectors: those whose integer coefficients in the stored cell's reciprocal basis are all divisible by the corresponding replication factor. `ks`, `kprefactor`, `Shost`, `k_offsets` keep their meaning and shrink. The integer coefficients come from the enumeration in `kvectors`, not from rounding a matrix product.
- ☑ The guest self term `KE Σ_k pref_k |S_g(k)|²` over the full k-vector set of the stored cell depends on the orientation only. `FrameworkBatch` evaluates it at a fixed set of 64 orientations (deterministic, seeded), adds its mean to `constant_offset`, and stores the half-range as `self_term_halfrange` (per system, energy units). It throws if `2·self_term_halfrange / kT_ref > 1e-3` with `kT_ref = KB·300 K`, naming the half-range estimate and the guard factor: that situation (a strongly polar guest in a small periodic cell) needs the per-insertion sum, which this design does not provide.
- ☑ `insertion_energy`'s reciprocal loop computes only the cross term `2 Re(conj(S_host) S_g)` over the coupled k-vectors.
- ☑ The neutral-guest shortcut (no reciprocal table at all) is unchanged.
- ☑ Oracle: a test-only reference `insertion_energy_reference` keeps the full sum (all k-vectors, cross + self) and the brute-force real-space loop. The production energy agrees with it within `2·self_term_halfrange` plus 1e-12 relative, over random poses, for CO2 and for a polar three-site guest.
- ☑ The kUPS cross-code test passes unchanged in tolerance.
- ☑ Measured and recorded: kernel time before/after on the RTX 3050 and the R9700, both precisions; bytes per framework.

  RUBTAK 3×3×3 + CO2, kernel-only time for a 65,536-insertion chunk on an RTX 3050 (`bench/gpu`,
  Julia 1.13.0):

  | Precision | k-vectors kept | Bytes/framework | Kernel time before (c910867) | Kernel time after |
  |---|---|---|---|---|
  | Float32 | 190 (was 4587) | 66,120 B (was 171,648 B) | 96.4 ms | 77.2 ms |
  | Float64 | 190 (was 4587) | 119,928 B (was 330,984 B) | 2099.8 ms | 1621.9 ms |

  Radeon AI PRO R9700 (ROCm, AMDGPU 2.8.0), kernel-only insertions per second with runs of 256
  insertions per framework (`bench/widom_scaling.jl`, 262,144-insertion chunk), from
  `bench/results/pureadsorb_widom_scaling_galen_rocm_{f32,f64}_run256_20260920.json` (c910867)
  and `..._20260920_584b806.json`:

  | Precision | Frameworks | Before (c910867) | After (584b806) |
  |---|---|---|---|
  | Float32 | 1 | 3.32M | 4.52M |
  | Float32 | 32,768 | 3.22M | 4.49M |
  | Float64 | 1 | 169k | 234k |
  | Float64 | 32,768 | 169k | 234k |

### E2 — cell list for the real-space loop
- ☑ Per framework, a grid in fractional coordinates of the stored cell with `n_i = max(1, floor(L_i / w))` cells along axis i (`L_i` perpendicular lengths, `w` the target width, default chosen by benchmark among 2, 3, 4, 6 Å and recorded).
- ☑ Atoms are stored sorted by cell, so a cell is a contiguous range of `positions`/`types`/`charges`; the batch gains `cell_offsets` (concatenated, ragged), `cellgrid_offsets` (where each system's cells start) and `ncells::SVector{3,Int32}` per system. No per-atom index indirection.
- ☑ One stencil per insertion, centered on the guest's reference point, with reach `m_i = ceil((r_c + r_guest) · n_i / L_i)` cells, `r_c = max(cutoff, ewald_cutoff)`, `r_guest` the largest site distance from the reference point. Along an axis with `2m_i + 1 > n_i` every cell is visited exactly once. Each atom is read once and evaluated against all guest sites; the minimum image is taken once per atom for the reference point and the site offsets are added, which is valid because construction requires `min_multiplicity(cell, r_c + r_guest) == (1,1,1)`.
- ☑ Cell index wrapping is branch-free integer arithmetic (no `mod`/`div` by a runtime value in kernel code; no throwing branches).
- ☑ Oracle: agrees with `insertion_energy_reference` to 1e-12 relative over random poses, for the triclinic RUBTAK cell, a cubic cell, a cell where `2m_i + 1 > n_i` on some axis, sites on cell faces, and every tested `w`.
- ☐ Generic over `Float32`/`Float64` (done — oracle tests pass at both precisions); allocation-free and type-stable under the StrictMode audit (done — `bench/audit.jl` full mode, AllocCheck+JET, 0 failures for `insertion_energy` at both precisions); compiles on CUDA (done — `:gpu` test item passes on this machine's RTX 3050); ROCm compilation is NOT verified — galen was not touched, per this task's instructions, so this box stays open for the controller.
- ☐ Measured and recorded: kernel time against `w` on the RTX 3050, both precisions; fraction of atoms visited; bytes per framework — table below. The R9700 measurement stays open for the controller (galen not touched).
- ☐ Gated option NOT built. The measurement below shows its gate condition IS met at the fastest tested width: the stencil visits 100% of atoms against 15.5% inside the cutoff-plus-guest-reach sphere, a 6.4x overshoot, well past the 2x trigger. Reported per the design's instruction ("do not build it without reporting first"), left for a future stage.

  RUBTAK 3×3×3 + CO2, kernel-only time for a 65,536-insertion chunk on an RTX 3050 (`bench/gpu`,
  Julia 1.13.0), measured with a one-off script mirroring `bench/widom_bench.jl`'s kernel-only
  path (same `@be` harness, same warm-up), since the full `widom_bench.jl` grid sweep is not
  needed for this table:

  | cellwidth (Å) | Kernel F32 | Kernel F64 | Mean atoms visited | Fraction inside r_c+r_guest sphere | Bytes/framework F32 | Bytes/framework F64 |
  |---|---|---|---|---|---|---|
  | 2 | 190.2 ms | 1318.8 ms | 58.0% | 15.5% | 89,476 | 143,284 |
  | 3 | 130.2 ms | 1024.1 ms | 77.1% | 15.5% | 73,060 | 126,868 |
  | 4 | 74.1 ms | 1396.2 ms | 100% | 15.5% | 69,064 | 122,872 |
  | 6 (chosen default) | 72.0 ms | 1383.2 ms | 100% | 15.5% | 67,012 | 120,820 |

  At 4 and 6 Å the stencil's reach already spans the whole grid on every axis for this system
  (`2m_i+1 >= n_i`), so those two widths visit every atom — the same set brute force would — yet
  both are still faster than the E1 baseline below, from evaluating all `N` guest sites against
  one `minimum_image` per host atom instead of one per site per atom. The narrower widths (2, 3
  Å) visit measurably fewer atoms but are markedly slower: with `natoms/n_i^3` this low, most
  cells hold very few atoms, so the added cost of iterating many near-empty cells outweighs the
  saved atom evaluations. `cellwidth = 6` is the fastest F32 measurement and within 5% of the
  fastest F64 measurement (1318.8 ms at `cellwidth = 2`), so it is the default.

  Speedup against the E1 baseline (commit 9a0771f: Float32 77.2 ms, Float64 1621.9 ms,
  66,120 / 119,928 bytes/framework): at `cellwidth = 6`, Float32 1.07x, Float64 1.17x.

### Kernel time decomposition after E2 (RTX 3050, 65,536-insertion chunk, commit da327a6)

Measured by running the production kernel on batches with parts disabled (empty reciprocal
table; Ewald cutoff set to 1e-3 Å so no pair evaluates the screened Coulomb term; both cutoffs
at 1e-3 Å so only the distance checks remain):

| Part | Float32 | Float64 |
|---|---|---|
| Whole kernel | 70.2 ms | 1378 ms |
| Reciprocal cross term (190 k-vectors) | 1.4 ms (2%) | 13 ms (1%) |
| Screened Coulomb in real space: `sqrt`, `erfc_dev`, division, for the ~12% of pairs inside the cutoff | 29.3 ms (42%) | 1039 ms (75%) |
| Lennard-Jones for the same pairs | 14.0 ms (20%) | 154 ms (11%) |
| Minimum image and distance checks over all atoms | 25.5 ms (36%) | 173 ms (13%) |

Float32 against Float64 on the same case (RTX 3050, seed 42): `μ_ex` −0.143435(100) eV from
4·10⁶ Float32 insertions against −0.143647(219) eV from 10⁶ Float64 insertions, 0.9 combined
standard errors apart; `K_H` and `q_st` within 0.9 and 0.5.

Consequences. The pair arithmetic inside the cutoff, not the visit to far atoms, dominates: 62%
of the Float32 kernel and 86% of the Float64 kernel. A stencil that skips far atoms can remove at
most the 36% / 13% spent on distance checks, and the measured cell traversal overhead at narrow
widths exceeds that saving for a 3078-atom framework. The gated pruning option is therefore not
built. The cell list stays, because a short-reach test needs it (E3).

### E2b — cheaper screened Coulomb pair term
- ☑ `erfc_dev` is a 28-term Chebyshev series valid for every argument at 1e-12. The pair term only needs `x = α r < α·ewald_cutoff` (3.20 for RUBTAK 3×3×3 + CO2 at the default precision). Measured on the GPU (RTX 3050, `bench/gpu`, 65,536-insertion chunk, whole kernel time), against `SpecialFunctions.erfc`, max relative error over `[0, 4]` (dense grid plus random points):

  | Candidate | Kernel F32 | Kernel F64 | Max rel. error F32 | Max rel. error F64 | Extra bytes/batch |
  |---|---|---|---|---|---|
  | today (`erfc_dev`, 28-term global series) | 69.8 ms | 1382.9 ms | 1.8e-6 | 3.7e-15 | 0 |
  | A: restricted-range series (`pair_erfc_dev`, N=8 (F32) / N=17 (F64) terms, domain [0,4]) | 63.4 ms | 1210.9 ms | 1.4e-6 | 9.5e-15 | 0 |
  | B: shared Hermite table (`erfc(α√s)/√s`, s_min=4 Å², rc²=144 Å², 900 (F32) / 16,000 (F64) nodes) | 77.8 ms | 638.2 ms | 1.8e-6 | 8.2e-13 | 7,200 B (F32) / 256,000 B (F64) |

  A wins outright in Float32 (69.8 → 63.4 ms) and improves Float64 (1382.9 → 1210.9 ms, 12%). B is dramatically faster in Float64 (638.2 ms, 54% faster than today) because it removes the `exp`/`sqrt`/division this GPU's Float64 path pays for in software, but B is a *regression* in Float32 (69.8 → 77.8 ms, confirmed with an interleaved value/slope table layout too: 70.5 ms) — the table-lookup memory latency it adds costs more than the (already hardware-fast) transcendentals it removes. No single algorithm wins both precisions.

  **Kept: A only, for both precisions.** The alternative — B for Float64, A for Float32, dispatched once per `T` inside one generic function — would capture B's larger Float64 win, but needs a per-batch table field threaded through `FrameworkBatch`, `widom_scaling.jl`'s tiling, and the docs, and stretches "one kernel source, per-precision constants selected by dispatch on `T`" (this design's literal wording) from selecting constants to selecting an entire algorithm. Given the added surface area for a Float64-only gain, A is kept as the simpler, literal-reading choice; B's numbers are recorded above as an open option if the controller wants the larger Float64 win built out.
- ☑ `pair_erfc_dev`'s accuracy is a tested bound against `SpecialFunctions.erfc` over `[0, PAIR_ERFC_XMAX] = [0, 4]` in both precisions (`test/ewald_tests.jl`, dense grid plus random points): Float64 rtol 1e-12 (measured 9.5e-15), Float32 rtol 2e-6 (measured 1.4e-6; Float32 cannot reach a literal 1e-7 bound with this construction — rounding in the polynomial evaluation and the final `exp` floors the achievable error near 1.3e-6, already an improvement on `erfc_dev`'s own Float32 accuracy of 1.8e-6 on this range). The Madelung and α-independence tests are unaffected (they use `ewald_energy`/`erfc_dev`, unchanged). The kUPS `:slow` cross-code test passes unchanged (3σ criterion).
- ☑ Kernel code stays allocation-free, branch-safe for GPU compilation and generic over the float type: `_pair_erfc_coef(::Type{T})` dispatches per-precision coefficient tuples, one `pair_erfc_dev(z::T)` source.
- ☑ Measured and recorded: the decomposition table below, re-measured, on the RTX 3050. The R9700 measurement stays open for the controller (galen not touched).

  RUBTAK 3×3×3 + CO2, kernel-only time for a 65,536-insertion chunk on an RTX 3050 (`bench/gpu`,
  Julia 1.13.0), same decomposition method as after E2 (empty reciprocal table; Ewald cutoff at
  1e-3 Å for the Coulomb-free runs; both cutoffs at 1e-3 Å for the distance-only run):

  | Part | Float32 | Float64 |
  |---|---|---|
  | Whole kernel | 63.4 ms | 1245.9 ms |
  | Reciprocal cross term (190 k-vectors) | 0.9 ms (1%) | 13.6 ms (1%) |
  | Screened Coulomb in real space: `sqrt`, `pair_erfc_dev`, division | 22.9 ms (36%) | 903.0 ms (72%) |
  | Lennard-Jones for the same pairs | 14.1 ms (22%) | 156.2 ms (13%) |
  | Minimum image and distance checks over all atoms | 25.5 ms (40%) | 173.1 ms (14%) |

  Speedup against the E2 baseline (commit da327a6: Float32 70.2 ms, Float64 1378 ms): Float32
  1.11x, Float64 1.11x. Bytes/framework are unchanged from E2 (no new batch fields).

### E3 — hard-core rejection before the energy
- ☐ The cell list serves this test only. Its reach is the largest core radius (about 1.5 Å), so with cells of about 3 Å a site checks 27 cells instead of every atom; the full energy keeps a plain loop over the system's atoms with one minimum image per atom. `cellwidth` and the full-energy path change accordingly (the full-energy stencil walk of E2 is replaced by the linear loop, which the width table above shows is the fastest form for this system size).
- ☐ Phase 0 tests, through the cell list, whether any site lies inside a per-type-pair core radius `ρ_ab` of a host atom. `ρ_ab` is the radius at which the pair's own energy (LJ repulsion minus the largest possible Coulomb attraction of that pair) exceeds the underflow threshold of the working float type plus a per-framework bound `B_s` on every other term, so a rejected insertion has weight exactly 0.0 and the statistics are unchanged. Pairs with `ε_ab = 0` never reject.
- ☐ Survivors are compacted (host-side at first: the chunk's flags are already copied per chunk; a device-side scan only if the copy is measured to matter) and phase 1 runs with the survivor count as its launch size.
- ☐ `B_s` is rigorous (pair-minimum and absolute-charge bounds); its looseness costs rejection fraction, not correctness. Expected rejection with the crude bound: about 55–60% of insertions for CO2 in RUBTAK.

### Not in this design
Storing the unit cell with explicit image shifts instead of the supercell (a further ~27× cut in
positions); retuning the Ewald split; device-side random numbers; importance sampling; adaptive
per-framework sampling; ahead-of-time binaries. Each needs its own measurement first.

## Why the earlier "reject, then reciprocal" idea is not here

Rejecting after the real-space energy and skipping the reciprocal sum for the rejected 67% is
exact, but on a GPU the reciprocal sum is 22% of the time, so it saves 15%. E1 removes 96% of the
reciprocal loop for every insertion, after which there is nothing left to skip.

## Expected effect (arithmetic on the measured shares, to be replaced by measurements)

| After | Relative kernel time (GPU) | Bytes per framework (Float32) |
|---|---|---|
| today | 1.00 | 172 KB |
| E1 | about 0.79 | about 66 KB |
| E2 (measured, RTX 3050: one minimum image per atom; the stencil visits every atom) | 0.73 Float32, 0.66 Float64 | 67 KB |
| E2b (screened Coulomb pair term 3–4× cheaper; arithmetic) | about 0.5 Float32, about 0.25 Float64 | same, plus one shared table if a table is chosen |
| E3 (57% rejected in phase 0 at a few percent of the kernel's cost; arithmetic) | about 0.25 Float32, about 0.12 Float64 | same |

The first version of this table expected 0.25 from the cell list alone. That estimate assumed the
visit to far atoms dominated the kernel; the decomposition above shows it is 36% (Float32) and 13%
(Float64).

## Interfaces that change

- `Framework` gains `replication`. `read_cif` sets `(1,1,1)`.
- `FrameworkBatch` gains `self_term_halfrange` (E1) and the cell-list fields (E2); `ks`/`kprefactor`/`Shost` shrink. Atom order inside a framework is no longer the CIF order (E2).
- `insertion_energy` takes the cell-list arrays (E2). The reference implementation keeps the current signature's semantics for tests.
- `widom`'s signature and results' meaning are unchanged. Results change by about `self_term_halfrange` in energy (E1; a 64-orientation estimate that sampled continuous orientations exceed by up to about 1.3×) and by floating-point summation order (E2).
- bench scripts and `bench/audit.jl` follow the kernel signature.

## Testing and gates

TestItems as before; every stage keeps the brute-force reference as its oracle; the `:gpu` item
and the kUPS `:slow` item run at each stage; StrictMode audit fast in the loop and full as the
gate; cold `Pkg.test()` before each merge. Benchmarks saved as JSON per stage; docs numbers
updated per stage, stating each configuration in the present tense.
