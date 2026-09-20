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
- ☑ Generic over `Float32`/`Float64` (done — oracle tests pass at both precisions); allocation-free and type-stable under the StrictMode audit (done — `bench/audit.jl` full mode, AllocCheck+JET, 0 failures for `insertion_energy` at both precisions); compiles on CUDA (done — `:gpu` test item passes on this machine's RTX 3050); compiles and runs on ROCm (R9700, AMDGPU 2.8.0, commit da327a6).
- ☑ Measured and recorded: kernel time against `w` on the RTX 3050, both precisions; fraction of atoms visited; bytes per framework — table below. R9700 at the default width, kernel-only insertions per second with runs of 256 (`bench/results/pureadsorb_widom_scaling_galen_rocm_{f32,f64}_run256_20260920_da327a6.json`): Float32 4.81M (1 framework) and 5.43M (32,768 frameworks), against 4.52M and 4.49M at 584b806; Float64 253k at both sizes, against 234k.
- ☐ Gated option not built. The measurement below shows its gate condition is met at the fastest tested width: the stencil visits 100% of atoms against 15.5% inside the cutoff-plus-guest-reach sphere, a 6.4x overshoot, well past the 2x trigger. Left open for a future stage.

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
  | A: restricted-range series (`pair_erfc_dev`, N=8 (F32) / N=17 (F64) terms, domain [0,4]) | 63.4 ms | 1210.9 ms | 1.51e-6 | 9.5e-15 | 0 |
  | B: shared Hermite table (`erfc(α√s)/√s`, s_min=4 Å², rc²=144 Å², 900 (F32) / 16,000 (F64) nodes) | 77.8 ms | 638.2 ms | 1.8e-6 | 8.2e-13 | 7,200 B (F32) / 256,000 B (F64) |

  A wins outright in Float32 (69.8 → 63.4 ms) and improves Float64 (1382.9 → 1210.9 ms, 12%). B is dramatically faster in Float64 (638.2 ms, 54% faster than today) because it removes the `exp`/`sqrt`/division this GPU's Float64 path pays for in software, but B is a *regression* in Float32 (69.8 → 77.8 ms, confirmed with an interleaved value/slope table layout too: 70.5 ms) — the table-lookup memory latency it adds costs more than the (already hardware-fast) transcendentals it removes. No single algorithm wins both precisions.

  **Kept: A only, for both precisions.** The alternative — B for Float64, A for Float32, dispatched once per `T` inside one generic function — would capture B's larger Float64 win, but needs a per-batch table field threaded through `FrameworkBatch`, `widom_scaling.jl`'s tiling, and the docs, and stretches "one kernel source, per-precision constants selected by dispatch on `T`" (this design's literal wording) from selecting constants to selecting an entire algorithm. Given the added surface area for a Float64-only gain, A is kept as the simpler, literal-reading choice; B's numbers are recorded above as an open option for a future stage that wants the larger Float64 win built out.
- ☑ `pair_erfc_dev`'s accuracy is a tested bound against `SpecialFunctions.erfc` over `[0, PAIR_ERFC_XMAX] = [0, 4]` in both precisions (`test/ewald_tests.jl`, dense grid plus random points): Float64 rtol 1e-12 (measured 9.5e-15), Float32 rtol 2e-6 (measured 1.51e-6 on a 20,001-point grid; Float32 cannot reach a literal 1e-7 bound with this construction — rounding in the polynomial evaluation and the final `exp` floors the achievable error near 1.3e-6, already an improvement on `erfc_dev`'s own Float32 accuracy of 1.8e-6 on this range). The Madelung and α-independence tests are unaffected (they use `ewald_energy`/`erfc_dev`, unchanged). The kUPS `:slow` cross-code test passes unchanged (3σ criterion).
- ☑ Kernel code stays allocation-free, branch-safe for GPU compilation and generic over the float type: `_pair_erfc_coef(::Type{T})` dispatches per-precision coefficient tuples, one `pair_erfc_dev(z::T)` source.
- ☑ Measured and recorded: the decomposition table below, re-measured, on the RTX 3050. The R9700 measurement stays open (galen not touched).

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
- ☑ The cell list serves this test only. Its reach is the largest core radius (about 0.9–1.2 Å, measured below), so with cells of about 2 Å a site checks a handful of cells instead of every atom; the full energy keeps a plain loop over the system's atoms with one minimum image per atom. `cellwidth` and the full-energy path change accordingly (the full-energy stencil walk of E2 is replaced by the linear loop).
- ☑ **Rejection rule (exact).** `widom` computes `W = exp(−ΔU/kT)` in the float type `F`; `W` is exactly `0.0` when `ΔU/kT > θ_F`, with `θ_F` the smallest value for which `exp(−F(θ_F)) == 0` (tested, not hard-coded from memory: 745.1332191019412 in Float64, 103.972084 in Float32). An insertion is rejected only if a rigorous lower bound on `ΔU` exceeds `(θ_F + 2)·kT + 1e-5·B_s + B_s − c_s` (the `1e-5·B_s` term covers the pair term's approximation error and floating-point summation error, both proportional to the magnitude of the summed terms); its `W` and `ΔU·W` are then recorded as exactly zero, so `μ_ex`, `K_H`, `q_st` and their errors are those of the single-phase calculation (verified `==` in the two-phase/single-phase test).
- ☑ **Lower bound.** `ΔU = Σ_{a,h} u_ah(r_ah) + U_recip + c_s`, with `u_ah(r) = LJ_ah(r)·[r < r_lj] + KE q_a q_h · pair_erfc(αr)/r·[r < r_ew]`. For a trigger pair at distance `r`: `u_ah(r) ≥ LJ_ah(r) − |K_ah|/r` when `K_ah = KE q_a q_h < 0`, else `≥ LJ_ah(r)`. For every other pair: when `K_ah ≥ 0`, `u_ah ≥ −ε_ah`; when `K_ah < 0`, let `r0_ah` be the zero of `u_ah` below its minimum (bisection on the host), then `u_ah ≥ 0` on `(0, r0]` and `u_ah ≥ −ε_ah − |K_ah|·erfc(α r0)/r0` on `[r0, r_c]`. `B_s` is the sum of those magnitudes over all sites and all host atoms of system `s`, plus the reciprocal bound `R_s = KE Σ_k kprefactor_k · 2|S_host,k| · Σ_a|q_a|`. `B_s` is computed at batch construction, stored per system (`hardcore_bound` in `src/reject.jl`), and is `Inf` (no rejection in that system) if any pair has `ε_ah = 0` with `K_ah < 0` (tested).
- ☑ **Core radii.** Per `widom` call (they depend on `T`), per system, per (guest site `a`, host LJ type `t`): `ρ_at` is the first root, scanning up from `r → 0`, of `LJ_at(r) − |K_min(a,t)|/r = (θ_F + 2)·kT + 1e-5·B_s + B_s − c_s` (`find_rho2`/`build_rejection_tables`), where `K_min(a,t)` is the most negative `K_ah` over the system's atoms of type `t` (zero if none is negative, `kmin_table`). Every `r < ρ_at` satisfies the rejection condition. Stored as `ρ²`, in a table indexed by a batch-local compact type index. (An earlier draft of this bullet omitted the `1e-5·B_s` term, inconsistent with the "Rejection rule" bullet's margin; the ruling was that the rejection rule is the authority, and both bullets now state the same target.)
- ☑ **Compact types.** `FrameworkBatch` remaps LJ types to the types actually present in the batch (framework atoms and guest sites); `types`, `sigma`, `epsilon` use the compact index (`compact_to_orig`, `guest_types`). The mapping is kept for error messages built at construction. The guest's own `guest_types_orig`/`guest_sites_orig`/`guest_charges_orig` are also kept, and `widom` throws a clear error if a later call passes a guest that does not match all three (not just types — every precomputed quantity depends on sites and charges too).
- ☑ **Phase 0 kernel.** One work-item per insertion (`hardcore_kernel!`). For each site, the neighborhood of the site's own home cell (reach from `max ρ`, computed per system per call) is scanned with one minimum image per atom; the flag is set if any `r² < ρ²_at`. Output: one `UInt8` per insertion. Same GPU-safety rules as the energy kernel; compiles and runs on this machine's RTX 3050 (`:gpu` test item, CUDA).
- ☑ **Compaction and phase 1.** The host reads the chunk's flags, builds the survivor index list in order, uploads it, and launches the energy kernel (`widom_kernel!`) over `nsurvivors` work-items, each reading its pose through the index and writing `ΔU` back at its original position. Rejected insertions add to the sample counts with zero weight (`boltzmann_weight`). A chunk with no survivors launches phase 1 not at all. The host pass was not replaced by a device-side scan: not measured to matter (phase 0 is 0.4–4% of the kernel-path time at the chosen width, see below).
- ☑ **Full energy path.** The energy kernel (`insertion_energy`) loops linearly over the system's atoms with one minimum image per atom; the cell arrays serve phase 0 only. `cellwidth` default chosen by measurement among 2, 3, 4 Å for phase-0 cost and bytes per framework (table below): 2 Å, the fastest at both precisions.
- ☑ **Zero-weight accumulation.** The host accumulates `ΔU·W` as zero whenever `W == 0` (`boltzmann_weight`), also on the single-phase path: in Float32 a site within about 10⁻³–10⁻⁴ Å of a host atom overflows the LJ term to `Inf`, and `Inf · 0` is `NaN`. Tested with a pose placed near a host atom (own commit, applies to the pre-E3 single-phase code path too).
- ☑ **Gate before building kernels:** measured on the CPU (JuliaMCP session), for CO2 in RUBTAK 3×3×3 at 298.15 K, 40,000 poses: rejection fraction 41.34% (Float64) / 41.37% (Float32), next to the fraction of exact zeros 68.58% / 78.19% (matching the design's recorded 68%/78% baseline). `B_s` = 127,118 kT in both precisions (order 10⁵ kT as expected); `ρ_at` in [0.9182, 1.2016] Å (Float64) / [0.9186, 1.2021] Å (Float32) (`ρ ≈ 0.3–0.4σ`, as expected; the `1e-5·B_s` margin term widens the lower end of this range by about 3e-4 Å against the value first recorded before that term was added). Both rejection fractions are above the 25% stop threshold and close to the ~40% estimate — continued to the kernels.
- ☑ **Tests.** `θ_F` by construction (`test/reject_tests.jl`); `B_s` against a brute-force evaluation of the bounded quantity on 2,000 random poses (the bound holds for every sampled pose: `ΔU − u_trigger ≥ −B_s + c_s`); no rejected pose has `W > 0` when its full energy is computed, 10⁵ poses per precision on the actual `find_rho2` radii (host math; the analogous production-kernel check is the `:gpu` item plus the two-phase/single-phase exact-equality test below, both of which ran with real rejections and no discrepancy); two-phase `widom` results equal single-phase results (`widom_singlephase`, non-exported) exactly (`==`) for fixed seeds on the CPU backend, for one framework, several frameworks, a neutral guest, and a framework where `B_s = Inf` (`test/widom_tests.jl`; small chunk sizes relative to the ~41% rejection fraction make both a zero-survivor and an all-survivor chunk overwhelmingly likely across the run, rather than being individually forced); GPU results bit-equal to CPU on this machine (RTX 3050, `:gpu` item, `rtol = 1e-10`, same tolerance as before E3 — not re-tightened); kUPS cross-code `:slow` item unchanged (passed); `bench/audit.jl` covers `insertion_energy`'s new (linear) signature and phase 0's own callees (`home_cell_dev`, `stencil_start_count`, `wrap_cell`, `cell_linear`) not already covered by `insertion_energy`, both precisions, full mode (24 findings, 0 failing).
- ☑ Measured and recorded (RTX 3050, `bench/gpu`, RUBTAK 3×3×3 + CO2): rejection fraction (kernel path) 40.5–40.8% both precisions; phase 0 / phase 1 / total kernel-path time for a 65,536-insertion chunk, cellwidth 2 Å: Float64 3.7 ms / 836.9 ms / 840.6 ms; Float32 0.68 ms / 38.7 ms / 39.4 ms; end-to-end `widom` time for 10⁶ insertions, one framework: Float64 13.02 s (76,792 ins/s); Float32 0.674 s (1.484M ins/s). Speedup against the E2b baseline (commit 082c656: whole kernel 63.4 ms Float32, 1245.9 ms Float64): Float32 1.61×, Float64 1.48×. R9700 measurement stays open for the controller (galen not touched).

  Cellwidth sweep for phase 0 (RTX 3050, 65,536-insertion chunk, `bench/gpu/cellwidth_sweep.jl`, mirroring `bench/widom_bench.jl`'s kernel-only path):

  | cellwidth (Å) | Phase 0 F64 | Phase 0 F32 | Bytes/framework F64 | Bytes/framework F32 |
  |---|---|---|---|---|
  | 2 (chosen default) | 4.8 ms | 0.8 ms | 143,436 | 89,552 |
  | 3 | 9.7 ms | 1.3 ms | 127,020 | 73,136 |
  | 4 | 19.1 ms | 2.5 ms | 123,024 | 69,140 |

  Phase 0's short reach (about 0.9–1.2 Å) reverses E2's finding: a narrower cell width is faster
  here, since fewer, smaller cells around each pose's home cell mean fewer atoms scanned before
  a stencil that no longer spans the whole grid the way E2's wider, cutoff-sized cells did.

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
| E3 (measured, RTX 3050: 40.5% rejected in phase 0 at 1–4% of the kernel-path cost) | 0.41 Float32, 0.40 Float64 | 90 KB (F32) / 143 KB (F64) at cellwidth 2 |

The first version of this table expected 0.25 from the cell list alone. That estimate assumed the
visit to far atoms dominated the kernel; the decomposition above shows it is 36% (Float32) and 13%
(Float64).

## Interfaces that change

- `Framework` gains `replication`. `read_cif` sets `(1,1,1)`.
- `FrameworkBatch` gains `self_term_halfrange` (E1) and the cell-list fields (E2); `ks`/`kprefactor`/`Shost` shrink. Atom order inside a framework is no longer the CIF order (E2). `types`/`sigma`/`epsilon` use a compact type index (E3), with `compact_to_orig` to map back; `guest_types`/`guest_types_orig`/`guest_sites_orig`/`guest_charges_orig`, `bs` and `kmin` are new (E3). `reach` (E2, the cutoff-plus-guest-reach stencil half-width) is removed: the cell list now serves phase 0 only, whose reach depends on temperature and is rebuilt per `widom` call instead of stored (E3).
- `insertion_energy` took the cell-list arrays (E2); E3 replaces them with a plain `atom_base`/`natoms` pair (linear loop, no stencil). The reference implementation keeps the current signature's semantics for tests.
- `widom`'s public signature and results' meaning are unchanged. Results change by about `self_term_halfrange` in energy (E1; a 64-orientation estimate that sampled continuous orientations exceed by up to about 1.3×), by floating-point summation order (E2), and are exactly (`==`) unchanged by E3's rejection (tested). `widom` now throws if the passed `guest` does not match the one `FrameworkBatch` was built from (E3). A new non-exported `widom_singlephase` (E3, test-only) skips rejection.
- bench scripts and `bench/audit.jl` follow the kernel signatures, including the new phase-0 kernel (E3).
- `FrameworkBatch` throws when `α·ewald_cutoff` exceeds `PAIR_ERFC_XMAX = 4` (Ewald precision tighter than about 1e-9 at a 12 Å cutoff); the message names both values and the two ways out, a looser `ewald.precision` or a shorter `ewald.cutoff`.

## Testing and gates

TestItems as before; every stage keeps the brute-force reference as its oracle; the `:gpu` item
and the kUPS `:slow` item run at each stage; StrictMode audit fast in the loop and full as the
gate; cold `Pkg.test()` before each merge. Benchmarks saved as JSON per stage; docs numbers
updated per stage, stating each configuration in the present tense.
