# PureAdsorb.jl — design

Batched GPU Widom insertion, and later Monte Carlo adsorption, for porous crystals in pure
Julia. The first milestone reproduces one number from CuspAI's kUPS engine (Henry coefficient
of CO2 in a Zr-MOF) and benchmarks it head to head on the same GPU.

## Decisions made without asking (override any of them)

| Decision | Choice | Alternative considered |
|---|---|---|
| Package name | `PureAdsorb.jl` | `PureWidom.jl` (too narrow once GCMC lands) |
| Location | `~/Documents/claude/PureAdsorb.jl` | |
| License | MIT | |
| Julia floor | 1.10 for the core; a cuTile extension (Milestone D) would need 1.11 | |
| CIF reader | own minimal parser for P1 CIF with fractional coordinates and `_atom_site_charge` | Chemfiles.jl (native library, allowed) once non-P1 files matter |
| Random numbers | generated on the host with `Xoshiro`, uploaded per chunk of insertions | device-side counter RNG if the upload is measured to matter |
| Units | eV, Å, K, elementary charge; matches kUPS so energies compare directly | |
| kUPS reference runs | on the 4070 in the eGPU enclosure, Python as a stopwatch only, never a dependency; numbers committed under `bench/results/` | **needs explicit approval, same exception as PyTorch-on-galen** |

## Scope

### Milestone A — Widom insertion in the empty host (this spec's deliverable)

Input: a batch of frameworks (P1 CIF with partial charges), one rigid guest (sites, charges,
LJ types), a force field (per-type σ, ε with Lorentz–Berthelot mixing and a cutoff with tail
correction), Ewald parameters (real-space cutoff, precision), temperature, number of
insertions.

Output per framework: excess chemical potential μ_ex, Henry coefficient K_H, zero-loading
isosteric heat q_st, each with a standard error, using the same definitions as kUPS
(`WidomStatistics`; Vlugt et al. 2008).

Not in A: any move that changes the framework state, adsorbate–adsorbate interactions,
flexible guests, blocking spheres, energy grids, TMMC.

### Milestone B — Widom along an NVT Monte Carlo run

Per-system Metropolis chains with translation, rotation and reinsertion moves on a set of
already-adsorbed guests, with Widom sampled every cycle. Reproduces kUPS's
`examples/mcmc_widom.yaml` exactly. Adds guest–guest energy and a per-system mutable state.

### Milestone C — Grand canonical Monte Carlo

Insertion and deletion moves with μVT acceptance, fugacity from the Peng–Robinson equation of
state, isotherms as output. Validation against kUPS GCMC and published RASPA isotherms
(methane in IRMOF-1).

### Milestone D — optional accelerators, each gated by a measurement

cuTile.jl backend for the reciprocal-space sum; precomputed energy grids with interpolation
(RASPA style); cell-list neighbor search if brute force over host atoms is measured to
dominate. None is built until a benchmark shows the current path is the bottleneck.

## Architecture (Milestone A)

Four units. Each can be tested alone.

### 1. Structures — `src/structure.jl`

- `Framework{T}`: cell matrix (3×3, columns are lattice vectors), fractional positions,
  LJ type index per atom, charge per atom, element labels. Constructed from a CIF path.
- `replicate(fw, (na, nb, nc))`: supercell. kUPS's example uses 3×3×3.
- `Guest{T}`: site positions in the molecular frame, LJ type index per site, charge per site,
  critical constants (used only from Milestone C).
- `ForceField{T}`: per-type σ and ε, cutoff, tail-correction flag. Pair parameters come from
  Lorentz–Berthelot mixing at construction, stored as an `ntype × ntype` table.

Plain structs, CPU only, no GPU types. The generic-indexing rule applies: nothing here assumes
1-based indexing without declaring it.

### 2. Batch — `src/batch.jl`

`FrameworkBatch{T, V<:AbstractVector}`: all frameworks concatenated into structure-of-arrays,
plus an offsets vector marking where each framework's atoms start. Fields:

- `positions` (3 × Natoms_total, Cartesian, Å), `types`, `charges`
- `cell`, `invcell` (3 × 3 × Nsys)
- `atom_offsets` (Nsys + 1)
- Ewald per system: `alpha`, `kvectors` and their prefactors, host structure factors
  `S_host` (complex, per k-vector), `k_offsets` (Nsys + 1)

Built on the CPU, moved to any backend with `Adapt.adapt`. This one type is what replaces
kUPS's tables, indices, lenses and patches: every kernel takes the batch and a system index.

### 3. Energy — `src/energy.jl`, `src/ewald.jl`

Guest–host insertion energy ΔU for one rigid guest at one pose (position + quaternion):

- **Lennard-Jones**: brute-force loop over the host atoms of that system, minimum image in the
  triclinic supercell, cutoff, analytic tail correction per guest site. No neighbor list in A.
- **Ewald real space**: same loop, erfc-screened Coulomb inside the real-space cutoff.
- **Ewald reciprocal space**: guest structure factor per k-vector (a handful of sites), dotted
  against the precomputed host structure factor. Host-only and guest-only self terms are
  constant per insertion and are included so ΔU matches kUPS's energy definition, not just its
  Boltzmann average.

Analytic expressions only. No autodiff anywhere in the package.

Ewald parameters (α, k-space cutoff) are derived from the real-space cutoff and the requested
precision with the same formulas kUPS uses, so the two codes evaluate the same sum.

### 4. Widom kernel and statistics — `src/widom.jl`

One `KernelAbstractions` kernel. Work-item = one insertion. It reads its system index from a
lookup (insertion → system), reads its random position and quaternion from the uploaded random
arrays, calls the energy unit, and writes `ΔU` to an output array.

Reduction on the host or with a second kernel: per system, `⟨e^{−βΔU}⟩` via log-sum-exp,
`⟨ΔU e^{−βΔU}⟩`, and their standard errors by blocking (kUPS uses Welford moments; blocks of
insertions give the same standard error and are simpler). Then

```
μ_ex  = −kT · ln ⟨e^{−βΔU}⟩
K_H   = ⟨e^{−βΔU}⟩ / (R T)          per unit volume; converted to mol/(kg·Pa) with the framework mass
q_st  = ⟨ΔU e^{−βΔU}⟩ / ⟨e^{−βΔU}⟩ − kT   reported as −q_st in kJ/mol, kUPS's sign convention
```

Insertions are processed in chunks (default 2^16 per launch) so the random arrays and output
stay bounded; chunk results are accumulated on the host.

Backends: CPU (KernelAbstractions' CPU backend) always; CUDA and AMDGPU through the package
extensions mechanism (`ext/`), with `Adapt` doing the data movement.

### Public API

```julia
fw   = Framework("RUBTAK.cif")                      # P1 CIF with charges
ff   = ForceField("trappe.yaml")                    # or a Dict of σ, ε per type
co2  = Guest("co2.yaml")
batch = FrameworkBatch([replicate(fw, (3,3,3))]; ff, guest = co2, ewald = (cutoff = 12.0, precision = 1e-6))
res  = widom(batch, co2, ff; T = 298.15, ninsert = 1_000_000, backend = CUDABackend(), seed = 42)
res[1].K_H, res[1].μ_ex, res[1].q_st, res[1].K_H_err
```

Input files use the same YAML shapes as kUPS's examples so the two codes run from identical
inputs. YAML parsing through `YAML.jl`.

## Data flow

```
CIF, YAML ─► Framework / Guest / ForceField (CPU structs)
          ─► FrameworkBatch (SoA + offsets + Ewald tables, CPU)
          ─► adapt(backend, batch)                         one upload per batch
          ─► for chunk in insertions:
                 host RNG ─► positions, quaternions ─► upload
                 widom_kernel!(ΔU, batch, guest, ff, randoms)
                 download ΔU ─► accumulate per-system moments
          ─► WidomResult per system
```

## Error handling

Fail fast, at construction:

- a guest LJ type missing from the force field, a CIF without charges when Ewald is requested,
  a non-P1 CIF, a cell whose minimum image is shorter than twice the cutoff (kUPS's
  `min_multiplicity` check), an `ninsert` not divisible into at least two blocks (no standard
  error possible).

Kernels never throw; anything they would need to check is checked before launch.

## Testing

TestItems.jl + TestItemRunner.jl. `max_workers` always passed. Items:

- CIF parser round trip on the RUBTAK file (117 atoms, charges, triclinic cell).
- Supercell replication: atom count, cell matrix, charge neutrality preserved.
- Minimum image against a brute-force search over ±1 images, triclinic cell.
- Lennard-Jones pair energy and tail correction against closed-form values.
- Ewald against the Madelung constant of NaCl and of a simple cubic lattice (same check kUPS
  ships), and real+reciprocal energy independent of α over a range.
- Empty box: μ_ex = 0, K_H = 1/(RT) to statistical precision.
- Single fixed host atom: Widom average against numerical integration of e^{−βU(r)} in 1D.
- CPU backend equals GPU backend bit-for-bit on the same random arrays (run when a GPU is
  present, skipped with a tag otherwise).
- Entry points accept `OffsetArray` and `view` inputs for the CPU structs, per the generic
  indexing rule.
- Reference test: RUBTAK + CO2 at 298.15 K against the kUPS numbers committed in
  `test/reference/rubtak_co2_kups.json`, within combined standard error.
- StrictMode audit script under `bench/` gating the kernel and the energy functions on
  `:typestable` and `:noalloc` (CPU backend), `mode = :fast` in the loop, `:full` as the gate.

## Benchmark protocol — `bench/`

- Case: RUBTAK 3×3×3, TraPPE CO2, 12 Å cutoff, Ewald precision 1e-6, 298.15 K, one host and
  then a batch of 64 copies, insertion counts 10^4 … 10^7.
- Both codes timed on the GPU with events around the kernel work; batch upload reported
  separately. Interleaved runs, ≥ 10 repetitions, median and spread.
- CPU governor set to `performance` for the session on neuromancer; the R9700 run on galen is
  Julia-only and is the authoritative absolute number.
- `BLAS.set_num_threads(1)` at the top of every timing script.
- Every sample saved to `bench/results/*.json` with hostname, GPU, driver, Julia and package
  versions; plots regenerated from JSON only. `bench/run_kups.sh` documents the exact kUPS
  invocation and environment so anyone with a JAX install can reproduce the reference numbers.
- Plots: violin per insertion count, insertions per second for both codes; a second plot for
  the 64-framework batch.

## Repository layout

```
PureAdsorb.jl/
  Project.toml            deps: KernelAbstractions, Adapt, YAML, SpecialFunctions (erfc), Random
  src/PureAdsorb.jl       structure.jl batch.jl energy.jl ewald.jl widom.jl io.jl
  ext/                    PureAdsorbCUDAExt.jl, PureAdsorbAMDGPUExt.jl
  test/                   Project.toml, runtests.jl, *_tests.jl, reference/
  bench/                  Project.toml, widom_bench.jl, run_kups.sh, plot_widom.jl, results/, audit.jl
  docs/                   DocumenterVitepress
  data/                   RUBTAK.cif, trappe.yaml, co2.yaml (from kUPS, Apache-2.0, attributed)
```

## Oracles

| What | Oracle |
|---|---|
| Ewald | Madelung constants; α-independence |
| LJ | closed form |
| Widom statistics | empty box; 1-D numerical integral |
| Whole pipeline | kUPS on the same inputs; PorousMaterials.jl on methane in IRMOF-1 (LJ only) |
| Milestone B | kUPS `mcmc_widom.yaml` |
| Milestone C | kUPS GCMC; RASPA published IRMOF-1 methane isotherm |
