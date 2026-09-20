# Efficiency branch — whole-branch review findings and rulings

Working notes for the fix wave on branch `efficiency` (review of master..3653c6f). Delete this
file in the last commit of the fix wave; the design file keeps the durable record.

## Critical

**C1. Core radii and pair zeros are not clamped to the Lennard-Jones cutoff.**
`src/reject.jl` (`find_rho2`, `find_r0`), `src/widom.jl` (`build_rejection_tables`). The bound's
trigger term `LJ_at(r) − |K_min|/r` and the per-pair envelope `u_ah ≥ 0 on (0, r0]` both assume the
LJ term is present, but `insertion_energy` adds it only for `r² < cutoff²`. With `cutoff < ρ_at`
every pose with the trigger pair in `(cutoff, ρ_at)` is rejected while its true energy is purely
attractive. Reproducer: cubic 30 Å cell, one host atom `q = +1` at the center, types `G_`, `B_`
with `σ = 3`, `ε = 0.01`, `cutoff = 1.0`, `tail = false`; one-site guest `q = −1`;
`EwaldParams(cutoff = 12, precision = 1e-6)`. Then `ρ = 1.74 Å > cutoff`, a pose at `r = 1.2 Å` is
flagged with `W = 2.3e191`, and `widom` disagrees with the single-phase result (`mu_ex` −7.47 eV
against −13.39 eV; `ninsert = 200_000`, `seed = 17`).
*Ruling:* clamp both: `ρ_at ← min(ρ_at, r_lj)` and `r0_ah ← min(r0_ah, r_lj)` (pass the LJ cutoff
in). For `r < min(ρ, r_lj)` the LJ term is included, so the rejection condition holds; for
`r ≤ min(r0, r_lj)` the envelope holds. Only the LJ cutoff matters: dropping a negative Coulomb
term can only raise the energy. Tests: the reproducer above must give two-phase `==` single-phase
and zero rejected poses with `W > 0` over 5·10⁵ poses; a second case with `r0 > r_lj`. The spec's
"Lower bound" and "Core radii" bullets state the clamp.

## Important

**I2. Phase 0 uses one stencil around the reference point with reach from ρ alone**, so an atom
within ρ of a site but farther than the reach from the reference point is missed (17 of 200,000
poses for CO2 in RUBTAK; under-rejection only, never a wrong answer). The spec says each site
scans the neighborhood of its own home cell; the box was ticked.
*Ruling:* implement the spec: each site computes its own home cell (site positions can lie outside
the cell: wrap the fractional coordinate into [0,1) without `mod`/`floor` throwing branches —
`f − floor(f)` on floats is fine; then clamp and `unsafe_trunc`) and scans its own stencil with
reach from `max ρ`. New test: the kernel's flags equal a per-site brute-force scan over all atoms
with the same `ρ²`, in BOTH directions, over ≥ 2·10⁵ poses, triclinic RUBTAK and a small cubic case
with a guest whose `r_guest` exceeds the cell width. Docstring, `docs/src/design.md` and the spec
say the same thing.

**I3. Host cost of the rejection tables.** `build_rejection_tables` runs serially on every `widom`
call at 479 µs per system (15.7 s at 32,768 systems); `hardcore_bound` costs 60 ms per framework at
construction; `bench/widom_scaling.jl` times neither.
*Ruling:* (a) bisections stop at a relative interval width of `4·eps(T)` with an iteration cap
(see I6) instead of a fixed 100 iterations; (b) memoize within one call: `find_r0` on
`(σ, ε, K)` and `find_rho2` on `(σ, ε, kmin, margin)` (a `Dict` keyed on the tuple; identical
systems then cost one lookup); (c) the per-system loops in `build_rejection_tables` and the
per-framework bound in `FrameworkBatch` run under `Threads.@threads` (each iteration writes only
its own system's slots; the memo is per-thread or built before the threaded loop); (d)
`bench/widom_scaling.jl` and `bench/widom_bench.jl` record the table-build time per call
(`rejection_tables_s`) next to the kernel-path time, and the batch construction time per framework.
Measure before/after and record in the spec.

**I4. The `1e-5·B_s` safety term is not a floating-point bound in Float32.** Recursive summation
of `n` terms has error at most `(n−1)·u·Σ|x_i|` with `u = eps(F)/2`, and
`Σ|x_i| ≤ ΔU + 2B_s + |c_s|`.
*Ruling:* the safety term becomes `2·n·eps(F)·(B_s + |c_s|) + 4e-6·B_s`, with
`n = N_sites·natoms_s + nk_s + 8` per system (`4e-6` covers `pair_erfc_dev`'s relative error on
every Coulomb term). The rejection target is `(θ_F + 2)·kT + safety + B_s − c_s`. Spec, theory page
and docstrings state this derivation; nothing calls the term empirical or asserts more than it
proves.

**I5.** `FrameworkBatch` throws when `α·ewald_cutoff > PAIR_ERFC_XMAX = 4` (Ewald precision tighter
than about 1e-9 at a 12 Å cutoff). *Ruling:* keep the guard; list it under the spec's "Interfaces
that change" and in `docs/src/design.md`'s error list; the message names the two ways out (looser
precision or shorter Ewald cutoff).

**I6. Unbounded loops.** `find_rho2` with `margin ≤ 0` never terminates; with `ε = 0` and
`kmin = 0` it halves `lo` about 1080 times and exits through a `NaN`. *Ruling:* `margin > 0` is
checked with an `ArgumentError` naming the system and the numbers; `ε == 0` returns `ρ² = 0`
explicitly (no repulsion, no rejection) before any loop; every bracketing and bisection loop has an
iteration cap that throws when exceeded; `find_r0` likewise.

## Minor (fix all)
- `docs/src/design.md`: `cellwidth` default is 2 Å, not 3; "A chunk with no survivors launches
  phase 1 at all" lacks "not".
- `docs/src/theory.md`: LaTeX inside single backticks renders literally (three places); the stated
  `u_ah(r)` omits the cutoff indicators and the `ρ ≤ r_lj` condition.
- `src/batch.jl` docstring: "about 1.5 Å" → the measured 0.9–1.2 Å; `compact_to_orig` is described
  as "kept for error messages" but no message uses it — describe it as the public mapping it is.
- `src/cell.jl`: comment says `home_cell`/`home_cell_dev` serve "`insertion_energy`'s stencil";
  the consumer is the core-test kernel.
- History narration in shipped text (rewrite as present-tense facts): `src/energy.jl` ("now that
  the hard-core rejection stage … has already screened"), `src/batch.jl` ("now serves only", "the
  opposite of E2's full-cutoff finding", "no longer spans"), `docs/src/design.md` ("no longer walks
  a stencil", "replaced a cell-list stencil walk once measurement showed"), `docs/src/theory.md`
  ("now serves only").
- `bench/plot_scaling.jl`: the legend label omits `meta.commit`, so two files for the same
  host/backend/precision/run draw indistinguishable lines; add the commit, and plot only the latest
  commit per series unless `PA_PLOT_ALL=1`.
- `src/ewald.jl`/spec: Float32 max relative error of `pair_erfc_dev` on [0, 4] is 1.51e-6 on a
  20,001-point grid (recorded as 1.4e-6).
- `src/widom.jl`: the mismatched-guest error prints both guests (batch's and passed).

## Spec file (docs/superpowers/specs/2026-09-20-efficiency-design.md)
- Remove workflow language: "the controller" (three places), "Reported per the design's
  instruction", and the parenthetical about an earlier draft and a ruling. State requirements and
  measured facts only.
- E2b's and E3's measured tables name their commit (re-measure after this fix wave and record the
  new commit).
- The E3 "Tests" bullet must not claim kernel coverage that the listed tests cannot provide; after
  I2's new test it can.
- "docs numbers updated per stage" stays open: `docs/src/benchmarks.md` is updated in a separate
  step after the re-measurement.
