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
- ☑ The guest self term `KE Σ_k pref_k |S_g(k)|²` over the full k-vector set of the stored cell depends on the orientation only. `FrameworkBatch` evaluates it at a fixed set of 64 orientations (deterministic, seeded), adds its mean to `constant_offset`, and stores the half-range as `self_term_halfrange` (per system, energy units). It throws if `self_term_halfrange / kT_ref > 1e-3` with `kT_ref = KB·300 K`, naming the value: that situation (a strongly polar guest in a small periodic cell) needs the per-insertion sum, which this design does not provide.
- ☑ `insertion_energy`'s reciprocal loop computes only the cross term `2 Re(conj(S_host) S_g)` over the coupled k-vectors.
- ☑ The neutral-guest shortcut (no reciprocal table at all) is unchanged.
- ☑ Oracle: a test-only reference `insertion_energy_reference` keeps the full sum (all k-vectors, cross + self) and the brute-force real-space loop. The production energy agrees with it within `self_term_halfrange` plus 1e-12 relative, over random poses, for CO2 and for a polar three-site guest.
- ☑ The kUPS cross-code test passes unchanged in tolerance.
- ☑ Measured and recorded: kernel time before/after on the RTX 3050, both precisions; bytes per framework.

  RUBTAK 3×3×3 + CO2, kernel-only time for a 65,536-insertion chunk on an RTX 3050 (`bench/gpu`,
  Julia 1.13.0):

  | Precision | k-vectors kept | Bytes/framework | Kernel time before (c910867) | Kernel time after |
  |---|---|---|---|---|
  | Float32 | 190 (was 4587) | 66,120 B (was 171,648 B) | 96.4 ms | 77.2 ms |
  | Float64 | 190 (was 4587) | 119,928 B (was 330,984 B) | 2099.8 ms | 1621.9 ms |

  The R9700 number from the design's "Expected effect" table is not measured here: this run
  used the RTX 3050 only, per the task that produced these numbers.

### E2 — cell list for the real-space loop
- ☐ Per framework, a grid in fractional coordinates of the stored cell with `n_i = max(1, floor(L_i / w))` cells along axis i (`L_i` perpendicular lengths, `w` the target width, default chosen by benchmark among 2, 3, 4, 6 Å and recorded).
- ☐ Atoms are stored sorted by cell, so a cell is a contiguous range of `positions`/`types`/`charges`; the batch gains `cell_offsets` (concatenated, ragged), `cellgrid_offsets` (where each system's cells start) and `ncells::SVector{3,Int32}` per system. No per-atom index indirection.
- ☐ One stencil per insertion, centered on the guest's reference point, with reach `m_i = ceil((r_c + r_guest) · n_i / L_i)` cells, `r_c = max(cutoff, ewald_cutoff)`, `r_guest` the largest site distance from the reference point. Along an axis with `2m_i + 1 > n_i` every cell is visited exactly once. Each atom is read once and evaluated against all guest sites; the minimum image is taken once per atom for the reference point and the site offsets are added, which is valid because construction requires `min_multiplicity(cell, r_c + r_guest) == (1,1,1)`.
- ☐ Cell index wrapping is branch-free integer arithmetic (no `mod`/`div` by a runtime value in kernel code; no throwing branches).
- ☐ Oracle: agrees with `insertion_energy_reference` to 1e-12 relative over random poses, for the triclinic RUBTAK cell, a cubic cell, a cell where `2m_i + 1 > n_i` on some axis, sites on cell faces, and every tested `w`.
- ☐ Generic over `Float32`/`Float64`; allocation-free and type-stable under the StrictMode audit; compiles on CUDA and ROCm.
- ☐ Measured and recorded: kernel time against `w`, both GPUs, both precisions; fraction of atoms visited; bytes per framework.
- ☐ Gated option, built only if the measurement shows the cubic stencil visits more than twice the atoms inside the cutoff sphere: prune stencil cells whose nearest point is farther than `r_c + r_guest` from the home cell (a per-framework list of cell offsets).

### E3 — hard-core rejection before the energy (outline; its own design after E2 is measured)
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
| E2 (stencil visits ~30% of atoms) | about 0.25 | about 73 KB |
| E3 (57% rejected in phase 0) | about 0.11 | same |

## Interfaces that change

- `Framework` gains `replication`. `read_cif` sets `(1,1,1)`.
- `FrameworkBatch` gains `self_term_halfrange` (E1) and the cell-list fields (E2); `ks`/`kprefactor`/`Shost` shrink. Atom order inside a framework is no longer the CIF order (E2).
- `insertion_energy` takes the cell-list arrays (E2). The reference implementation keeps the current signature's semantics for tests.
- `widom`'s signature and results' meaning are unchanged. Results change by at most `self_term_halfrange` in energy (E1) and by floating-point summation order (E2).
- bench scripts and `bench/audit.jl` follow the kernel signature.

## Testing and gates

TestItems as before; every stage keeps the brute-force reference as its oracle; the `:gpu` item
and the kUPS `:slow` item run at each stage; StrictMode audit fast in the loop and full as the
gate; cold `Pkg.test()` before each merge. Benchmarks saved as JSON per stage; docs numbers
updated per stage, stating each configuration in the present tense.
