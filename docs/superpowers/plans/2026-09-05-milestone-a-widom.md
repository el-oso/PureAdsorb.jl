# PureAdsorb.jl Milestone A (Widom insertion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batched GPU Widom test-particle insertion of a rigid charged guest into a batch of porous frameworks, reproducing kUPS's μ_ex, K_H and q_st for CO2 in RUBTAK, and benchmarked against kUPS on the same GPU.

**Architecture:** CPU structs (`Framework`, `Guest`, `ForceField`) are packed into one structure-of-arrays `FrameworkBatch` with offsets and precomputed Ewald tables, moved to a backend with `Adapt`, and consumed by a single KernelAbstractions kernel where one work-item evaluates one insertion energy. Reduction to statistics happens on the host by block averaging. All energies are analytic; there is no autodiff.

**Tech Stack:** Julia ≥ 1.10, KernelAbstractions, Adapt, StaticArrays, SpecialFunctions (erfc, CPU only), YAML, TestItems + TestItemRunner, StrictMode, Runic, DocumenterVitepress, Chairmarks for benchmarks. CUDA.jl and AMDGPU.jl only as package extensions.

**Spec:** `docs/superpowers/specs/2026-09-05-pureadsorb-design.md`

## Global Constraints

- No Python anywhere in the package, tests, or benchmark harness. The only Python execution in the whole milestone is `bench/run_kups.sh` producing reference numbers on the 4070, and it runs only after the user's explicit approval of that exception (same scope as the PyTorch-on-galen stopwatch exception of 2026-08-24).
- Julia floor `1.10` in `[compat]`.
- Units: Å, eV, K, elementary charge. `KB = 8.6173303e-5 eV/K`. `KE = 14.3996454 eV·Å/e²` (equals kUPS's `HARTREE * BOHR`).
- Every `[deps]` entry added with `ion add` (writes `[compat]`); never hand-write a UUID; never edit `Manifest.toml`.
- Test-only deps go in `test/Project.toml` only. Benchmark deps in `bench/Project.toml` only.
- All iteration through JuliaMCP (`julia_create_session`, `julia_eval_code`, `julia_run_testitems` with `max_workers` always passed). Cold `julia` only for `Pkg` operations, the final `Pkg.test()` gate, and pinned benchmarks.
- Generic indexing: iterate `eachindex`/`axes`; entry points that take arrays must work on `OffsetArray` and `view` inputs or declare `Base.require_one_based_indexing`.
- No `@inbounds` without a benchmark showing the bounds check costs something and a test covering the index range.
- `runic -i .` before every commit. Commit messages describe what the code does now, never the plan or history.
- `BLAS.set_num_threads(1)` at the top of every timing script; every benchmark sample saved to `bench/results/*.json`; plots regenerated from JSON only.
- Fail fast: invalid inputs throw at construction with a message that names the problem. Kernels never throw.
- Subagents executing a task: run every long command in the foreground with a generous timeout; never background a command and end the turn waiting for it.

## kUPS conventions this plan reproduces (extracted from the kUPS source, commit e183c9a)

| Quantity | Formula |
|---|---|
| LJ pair | `U = 4ε[(σ/r)¹² − (σ/r)⁶]` for `r < r_c`, else 0. Plain truncation, no shift. |
| Mixing | `σᵢⱼ = (σᵢ+σⱼ)/2`, `εᵢⱼ = √(εᵢεⱼ)` |
| Global tail | `E_tail = Σᵢⱼ (8π/3)(NᵢNⱼ/V) εᵢⱼ σᵢⱼ³ [(σᵢⱼ/r_c)⁹/3 − (σᵢⱼ/r_c)³]` over all species pairs with `εᵢⱼ > 0`, N = particle count per species (host + guests). The MC application uses this global form, not the per-pair switching form. |
| Ewald α | solve `erfc(z) = r_c · ε_total/2` for z, `α = z/r_c` |
| Ewald k_max | `k_max = 2α √(−ln(ε_total/2))` |
| Reciprocal basis | `B = 2π A⁻ᵀ`, A = cell matrix with lattice vectors as columns |
| k-vectors | `n₁ ∈ 0..n`, `n₂,n₃ ∈ −n..n`, `k ≠ 0`, `|k| ≤ k_max`; weight `w = 2` if `n₁ ≠ 0` else `1` |
| Real space | `E_sr = KE Σ_{pairs, r<r_c} qᵢqⱼ erfc(αr)/r` |
| Reciprocal | `E_lr = KE Σ_k w_k (2π/V) exp(−k²/4α²)/k² · |S(k)|²`, `S(k) = Σᵢ qᵢ exp(i k·rᵢ)` |
| Self | `E_self = −KE (α/√π) Σᵢ qᵢ²` |
| Exclusion | `E_excl = −KE Σ_{intramolecular pairs} qᵢqⱼ/rᵢⱼ` (unscreened, direct distance) |
| Net charge | `E_net = −KE π/(2Vα²) Q²`, `Q = Σᵢ qᵢ` |
| Total | `E = E_sr + E_lr + E_self + E_excl + E_net` |
| min multiplicity | `n_axis = ⌈2 r_c / L_axis⌉`, `L = V/|b×c|` etc. |
| Pose | position uniform in fractional `[0,1)³`; orientation by Shoemake: `u₁,u₂,u₃ ~ U(0,1)`, `q = (√(1−u₁) sin 2πu₂, √(1−u₁) cos 2πu₂, √u₁ sin 2πu₃, √u₁ cos 2πu₃)` |
| Widom | `W = exp(−ΔU/kT)`; `μ_ex = −kT ln⟨W⟩`; `K_H = V⟨W⟩/kT` (Å³/eV); `q_st = kT − ⟨ΔU W⟩/⟨W⟩` (eV) |
| Errors | block averaging: SEM = std(block means, ddof=1)/√n_blocks; q_st by the delta method on the two block-mean series |

Ghost insertion energy of a rigid guest with sites `g` into host `h`:

```
ΔU = ΔU_LJ + ΔU_tail + ΔU_coul
ΔU_LJ   = Σ_{g,h, r<r_c} 4ε_gh[(σ_gh/r)¹² − (σ_gh/r)⁶]
ΔU_tail = (8π/3V) [ 2 Σ_i Σ_j N_i δ_j c_ij + Σ_ij δ_i δ_j c_ij ],
          c_ij = ε_ij σ_ij³ [(σ_ij/r_c)⁹/3 − (σ_ij/r_c)³] (ε_ij>0), δ_j = guest site count of species j
ΔU_coul = KE Σ_{g,h, r<r_c} q_g q_h erfc(αr)/r
        + KE Σ_k w_k P(k) [ 2 Re( conj(S_h(k)) S_g(k) ) + |S_g(k)|² ]
        − KE (α/√π) Σ_g q_g²
        − KE Σ_{g<g'} q_g q_g' / r_gg'
        − KE π/(2Vα²) [ (Q_h + Q_g)² − Q_h² ]
```

`ΔU_tail`, the self term, the exclusion term and the net-charge term do not depend on the pose; they are computed once per (framework, guest) at batch construction as `constant_offset` and added in the kernel.

---

## File structure

| File | Responsibility |
|---|---|
| `Project.toml`, `LICENSE`, `README.md`, `.gitignore`, `JuliaFormat.toml` | package identity, MIT, Runic style |
| `src/PureAdsorb.jl` | module, includes, exports |
| `src/constants.jl` | `KB`, `KE` |
| `src/cell.jl` | `cell_matrix`, `volume`, `perpendicular_lengths`, `min_multiplicity`, `minimum_image`, `reciprocal_basis` |
| `src/structure.jl` | `Framework`, `read_cif`, `replicate`, `cartesian` |
| `src/forcefield.jl` | `ForceField`, `read_forcefield`, `typeindex`, `Guest`, `read_guest`, `tail_coefficient`, `tail_delta` |
| `src/ewald.jl` | `EwaldParams`, `ewald_alpha`, `ewald_kmax`, `kvectors`, `pk`, `erfc_dev`, `structure_factor`, `ewald_energy` (full, for tests) |
| `src/energy.jl` | `rotate`, `insertion_energy` (pose-dependent part) |
| `src/batch.jl` | `FrameworkBatch`, `Adapt` rule |
| `src/widom.jl` | `random_poses!`, `widom_kernel!`, `widom`, `WidomResult`, block statistics, `backend_loaded` |
| `ext/PureAdsorbCUDAExt.jl`, `ext/PureAdsorbAMDGPUExt.jl` | backend registration only |
| `test/Project.toml`, `test/runtests.jl`, `test/*_tests.jl`, `test/reference/` | TestItemRunner suite |
| `bench/Project.toml`, `bench/audit.jl`, `bench/widom_bench.jl`, `bench/run_kups.sh`, `bench/run_headtohead.sh`, `bench/plot_widom.jl`, `bench/results/` | StrictMode gate and benchmarks |
| `data/RUBTAK.cif`, `data/trappe.yaml`, `data/co2.yaml`, `data/NOTICE` | inputs copied from kUPS examples (Apache-2.0), attributed |
| `docs/` | DocumenterVitepress skeleton |

---

### Task 1: Package scaffold

**Files:**
- Create: `Project.toml`, `LICENSE`, `README.md`, `.gitignore`, `JuliaFormat.toml`, `src/PureAdsorb.jl`, `src/constants.jl`, empty `src/{cell,structure,forcefield,ewald,energy,batch,widom}.jl`, `test/Project.toml`, `test/runtests.jl`, `test/constants_tests.jl`, `data/*`

**Interfaces:**
- Produces: module `PureAdsorb`; `PureAdsorb.KB::Float64`, `PureAdsorb.KE::Float64`.

- [ ] **Step 1: Generate the package with Pkg (never hand-write the UUID)**

```bash
cd /home/el_oso/Documents/claude && julia -e 'using Pkg; Pkg.generate("PureAdsorb")' && mv PureAdsorb/Project.toml PureAdsorb/src PureAdsorb.jl/ && rmdir PureAdsorb
```
Edit `Project.toml` by hand only for these authored fields: `authors = ["Jorge Vieyra"]`, `version = "0.1.0"`, and add `[compat] julia = "1.10"`.

- [ ] **Step 2: Add dependencies with ion**

```bash
cd /home/el_oso/Documents/claude/PureAdsorb.jl && ion add KernelAbstractions Adapt StaticArrays SpecialFunctions YAML Random LinearAlgebra
cd test && ion add Test TestItemRunner OffsetArrays QuadGK && cd ..
```
`Random` and `LinearAlgebra` are stdlibs; `ion` writes their compat entry as `1`.

- [ ] **Step 3: Write `JuliaFormat.toml`, `.gitignore`, `LICENSE`, `README.md`**

`JuliaFormat.toml`:
```toml
style = "runic"
```
`.gitignore`:
```
Manifest*.toml
docs/build/
bench/results/*.png
```
`LICENSE`: MIT text, copyright 2026 Jorge Vieyra. `README.md`: one paragraph stating what the package computes (Widom insertion, batched, GPU via KernelAbstractions), the input formats (P1 CIF with charges, kUPS-style YAML for force field and guest), and a usage example identical to the spec's public API block.

- [ ] **Step 4: Copy kUPS example inputs with attribution**

```bash
K=/tmp/claude-1000/-home-el-oso-Documents-claude/e9164e19-e3fc-4684-a252-55fab14ffb17/scratchpad/kups/examples
mkdir -p data && cp $K/host/RUBTAK.cif $K/lennard_jones/trappe.yaml $K/adsorbate/co2.yaml data/
```
`data/NOTICE`:
```
RUBTAK.cif, trappe.yaml and co2.yaml are copied unchanged from the kUPS
examples (https://github.com/cusp-ai-oss/kups, commit e183c9a),
Copyright 2024-2026 Cusp AI, licensed under the Apache License 2.0.
```

- [ ] **Step 5: Write the module and constants**

`src/PureAdsorb.jl`:
```julia
module PureAdsorb

using StaticArrays
using KernelAbstractions
using Adapt
using SpecialFunctions: erfc
using Random: Xoshiro, AbstractRNG
using LinearAlgebra: det, norm, cross, dot, I, Diagonal
using YAML

include("constants.jl")
include("cell.jl")
include("structure.jl")
include("forcefield.jl")
include("ewald.jl")
include("energy.jl")
include("batch.jl")
include("widom.jl")

export Framework, read_cif, replicate, ForceField, read_forcefield, Guest, read_guest,
    EwaldParams, FrameworkBatch, widom, WidomResult

end
```
Create the seven empty source files so the module loads; each later task fills one.

`src/constants.jl`:
```julia
# Boltzmann constant in eV/K and the Coulomb prefactor 1/(4πε₀) in eV·Å/e².
# CODATA 2014 values as used by kUPS (via ASE's units table), so energies agree with it.
const KB = 1.38064852e-23 / 1.6021766208e-19
const KE = 14.399645351950548
```
Verify `KE` before committing: in a JuliaMCP session evaluate the product of the kUPS `HARTREE` and `BOHR` definitions (`src/kups/core/constants.py`, lines 53 and 55, with `_ELECTRON_MASS`, `EPSILON_0`, `HBAR` from the same file) to 17 significant digits and paste that value, not a literature constant.

- [ ] **Step 6: Test runner and first test**

`test/runtests.jl`:
```julia
using TestItemRunner
@run_package_tests filter = ti -> !(:gpu in ti.tags) && !(:slow in ti.tags)
```
`test/constants_tests.jl`:
```julia
@testitem "constants match kUPS" begin
    @test PureAdsorb.KB ≈ 8.6173303e-5 rtol = 1e-7
    @test PureAdsorb.KE ≈ 14.3996454 rtol = 1e-8
end
```

- [ ] **Step 7: Run it warm**

`julia_set_workspace_folders` on the package, `julia_list_testitems`, then `julia_run_testitems` with `max_workers = 1`. Expected: 1 pass.

- [ ] **Step 8: Commit**

```bash
runic -i . && git add -A && git commit -m "Scaffold PureAdsorb.jl with kUPS-compatible physical constants"
```

---

### Task 2: Cell geometry

**Files:**
- Create: `src/cell.jl`, `test/cell_tests.jl`

**Interfaces:**
- Produces:
  - `cell_matrix(a, b, c, α, β, γ) -> SMatrix{3,3,T}` (angles in degrees, lattice vectors as columns)
  - `volume(A::SMatrix{3,3}) -> T`
  - `perpendicular_lengths(A) -> SVector{3,T}`
  - `min_multiplicity(A, cutoff) -> NTuple{3,Int}`
  - `reciprocal_basis(A) -> SMatrix{3,3,T}` (`2π A⁻ᵀ`)
  - `minimum_image(A, invA, Δ::SVector{3}) -> SVector{3}` (Cartesian displacement wrapped to the nearest image by rounding in fractional coordinates)

- [ ] **Step 1: Failing tests**

```julia
@testitem "cell matrix from parameters" begin
    using StaticArrays, LinearAlgebra
    A = PureAdsorb.cell_matrix(14.76190, 14.80147, 14.76539, 59.84578, 60.04729, 59.81310)
    @test norm(A[:, 1]) ≈ 14.76190
    @test norm(A[:, 2]) ≈ 14.80147
    @test norm(A[:, 3]) ≈ 14.76539
    @test acosd(dot(A[:, 2], A[:, 3]) / (norm(A[:, 2]) * norm(A[:, 3]))) ≈ 59.84578 rtol = 1e-6
    @test PureAdsorb.volume(A) ≈ abs(det(A))
    L = PureAdsorb.perpendicular_lengths(A)
    @test L[1] ≈ PureAdsorb.volume(A) / norm(cross(A[:, 2], A[:, 3]))
    @test PureAdsorb.min_multiplicity(A, 12.0) == ntuple(i -> ceil(Int, 24.0 / L[i]), 3)
    B = PureAdsorb.reciprocal_basis(A)
    @test B' * A ≈ 2π * I
end

@testitem "minimum image is exact inside the cutoff" begin
    using StaticArrays, LinearAlgebra, Random
    A = 3 * PureAdsorb.cell_matrix(14.76190, 14.80147, 14.76539, 59.84578, 60.04729, 59.81310)
    invA = inv(A)
    rc = 12.0
    rng = Xoshiro(1)
    for _ in 1:2000
        Δ = A * (rand(rng, SVector{3, Float64}) .- 0.5) * 3      # spans several images
        d = PureAdsorb.minimum_image(A, invA, Δ)
        best = minimum(norm(Δ + A * SVector(i, j, k)) for i in -2:2, j in -2:2, k in -2:2)
        if best < rc
            @test norm(d) ≈ best
        end
    end
end
```

- [ ] **Step 2: Run, expect failure** (`julia_run_testitems`, items filtered to `cell_tests.jl`, `max_workers = 1`): fail with `UndefVarError: cell_matrix`.

- [ ] **Step 3: Implement**

```julia
# Lattice vectors are the columns of the cell matrix; a along x, b in the xy plane.
function cell_matrix(a, b, c, α, β, γ)
    T = float(promote_type(typeof(a), typeof(b), typeof(c), typeof(α), typeof(β), typeof(γ)))
    ca, cb, cg, sg = cosd(T(α)), cosd(T(β)), cosd(T(γ)), sind(T(γ))
    cy = (ca - cb * cg) / sg
    cz = sqrt(one(T) - cb^2 - cy^2)
    return SMatrix{3, 3, T}(a, zero(T), zero(T), b * cg, b * sg, zero(T), c * cb, c * cy, c * cz)
end

volume(A::SMatrix{3, 3}) = abs(det(A))

function perpendicular_lengths(A::SMatrix{3, 3})
    V = volume(A)
    a, b, c = A[:, 1], A[:, 2], A[:, 3]
    return SVector(V / norm(cross(b, c)), V / norm(cross(a, c)), V / norm(cross(a, b)))
end

# Smallest supercell in which every pair closer than `cutoff` has a unique nearest image:
# 2·cutoff must fit between opposite faces.
function min_multiplicity(A::SMatrix{3, 3}, cutoff)
    L = perpendicular_lengths(A)
    return ntuple(i -> ceil(Int, 2 * cutoff / L[i]), 3)
end

reciprocal_basis(A::SMatrix{3, 3}) = 2π * inv(A)'

# Rounding in fractional coordinates finds the true nearest image only for separations
# shorter than half the smallest perpendicular length; min_multiplicity guarantees that for
# every separation inside the cutoff, and longer ones are masked out by the caller.
function minimum_image(A::SMatrix{3, 3}, invA::SMatrix{3, 3}, Δ::SVector{3})
    f = invA * Δ
    return Δ - A * round.(f)
end
```

- [ ] **Step 4: Run, expect 2 passes.**

- [ ] **Step 5: Commit** — `runic -i . && git add -A && git commit -m "Add triclinic cell geometry: matrix, perpendicular lengths, minimum image"`

---

### Task 3: Framework and CIF reader

**Files:**
- Create: `src/structure.jl`, `test/structure_tests.jl`

**Interfaces:**
- Produces:
  - `struct Framework{T}`: `cell::SMatrix{3,3,T,9}`, `frac::Vector{SVector{3,T}}`, `labels::Vector{String}`, `symbols::Vector{String}`, `charges::Vector{T}`
  - `read_cif(path; T = Float64) -> Framework{T}`; throws `ArgumentError` for non-P1 files or missing `_atom_site_charge`
  - `replicate(fw::Framework, n::NTuple{3,Int}) -> Framework`
  - `cartesian(fw) -> Vector{SVector{3,T}}`
  - `natoms(fw)`, `total_charge(fw)`

- [ ] **Step 1: Failing tests**

```julia
@testitem "read RUBTAK.cif" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    @test PureAdsorb.natoms(fw) == 117
    @test fw.symbols[1] == "Zr"
    @test fw.charges[1] ≈ 2.38565
    @test fw.frac[1] ≈ SVector(0.37986, 0.37998, 0.61969)
    @test abs(PureAdsorb.total_charge(fw)) < 1e-3
    @test fw.cell ≈ PureAdsorb.cell_matrix(14.76190, 14.80147, 14.76539, 59.84578, 60.04729, 59.81310)
end

@testitem "replicate preserves counts and charge" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    sc = replicate(fw, (3, 3, 3))
    @test PureAdsorb.natoms(sc) == 27 * 117
    @test sc.cell ≈ 3 * fw.cell
    @test PureAdsorb.total_charge(sc) ≈ 27 * PureAdsorb.total_charge(fw)
    @test all(0 .<= reduce(vcat, collect.(sc.frac)) .< 1)
end

@testitem "read_cif rejects non-P1 and chargeless files" begin
    dir = mktempdir()
    p = joinpath(dir, "bad.cif")
    write(p, "data_x\n_symmetry_space_group_name_H-M 'P 21'\nloop_\n_atom_site_label\n_atom_site_fract_x\n")
    @test_throws "P1" read_cif(p)
    src = read(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"), String)
    write(p, replace(src, "_atom_site_charge\n" => ""))
    @test_throws "_atom_site_charge" read_cif(p)
end
```
The header check for `_atom_site_charge` must run before any row is parsed, otherwise the chargeless file fails on row length first.

- [ ] **Step 2: Run, expect `UndefVarError: read_cif`.**

- [ ] **Step 3: Implement**

```julia
struct Framework{T}
    cell::SMatrix{3, 3, T, 9}
    frac::Vector{SVector{3, T}}
    labels::Vector{String}
    symbols::Vector{String}
    charges::Vector{T}
end

natoms(fw::Framework) = length(fw.frac)
total_charge(fw::Framework) = sum(fw.charges)
cartesian(fw::Framework) = [fw.cell * f for f in fw.frac]

# Minimal CIF reader: one data block, P1 only, one atom_site loop with fractional
# coordinates and a charge column. Anything else is rejected rather than guessed.
function read_cif(path::AbstractString; T = Float64)
    lines = strip.(readlines(path))
    getval(key) = begin
        i = findfirst(l -> startswith(l, key * " ") || startswith(l, key * "\t"), lines)
        isnothing(i) && throw(ArgumentError("CIF is missing $key"))
        strip(lines[i][length(key)+1:end])
    end
    sg = strip(getval("_symmetry_space_group_name_H-M"), ['\'', '"'])
    replace(sg, " " => "") == "P1" || throw(ArgumentError("only P1 CIF files are supported, got space group $sg"))
    keys6 = ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
    cell = cell_matrix(ntuple(i -> parse(T, getval(keys6[i])), 6)...)
    hstart = findfirst(==("_atom_site_label"), lines)
    isnothing(hstart) && throw(ArgumentError("CIF has no _atom_site_label loop"))
    cols = String[]
    i = hstart
    while i <= length(lines) && startswith(lines[i], "_atom_site_")
        push!(cols, lines[i]); i += 1
    end
    "_atom_site_charge" in cols || throw(ArgumentError("CIF has no _atom_site_charge column; partial charges are required"))
    col(name) = findfirst(==(name), cols)
    cx, cy, cz, cq = col("_atom_site_fract_x"), col("_atom_site_fract_y"), col("_atom_site_fract_z"), col("_atom_site_charge")
    cl, cs = col("_atom_site_label"), col("_atom_site_type_symbol")
    any(isnothing, (cx, cy, cz, cl, cs)) && throw(ArgumentError("CIF atom_site loop lacks label, type_symbol or fractional coordinates"))
    frac = SVector{3, T}[]; labels = String[]; symbols = String[]; charges = T[]
    while i <= length(lines) && !isempty(lines[i]) && !startswith(lines[i], "_") && !startswith(lines[i], "loop_")
        f = split(lines[i])
        length(f) == length(cols) || throw(ArgumentError("CIF atom row has $(length(f)) fields, header has $(length(cols))"))
        push!(frac, SVector(parse(T, f[cx]), parse(T, f[cy]), parse(T, f[cz])))
        push!(labels, String(f[cl])); push!(symbols, String(f[cs])); push!(charges, parse(T, f[cq]))
        i += 1
    end
    return Framework{T}(cell, frac, labels, symbols, charges)
end

function replicate(fw::Framework{T}, n::NTuple{3, Int}) where {T}
    all(>=(1), n) || throw(ArgumentError("replication factors must be ≥ 1, got $n"))
    N = prod(n)
    frac = Vector{SVector{3, T}}(undef, N * natoms(fw))
    idx = 0
    for k in 0:n[3]-1, j in 0:n[2]-1, i in 0:n[1]-1, f in fw.frac
        idx += 1
        frac[idx] = (f + SVector(i, j, k)) ./ SVector(n)
    end
    return Framework{T}(fw.cell * Diagonal(SVector(n)), frac, repeat(fw.labels, N), repeat(fw.symbols, N), repeat(fw.charges, N))
end
```

- [ ] **Step 4: Run, expect 3 passes.**

- [ ] **Step 5: Commit** — `"Add Framework with a P1 CIF reader and supercell replication"`

---

### Task 4: Force field and guest

**Files:**
- Create: `src/forcefield.jl`, `test/forcefield_tests.jl`

**Interfaces:**
- Produces:
  - `struct ForceField{T}`: `names::Vector{String}`, `sigma::Matrix{T}`, `epsilon::Matrix{T}` (mixed tables), `cutoff::T`, `tail::Bool`
  - `ForceField(names, σ::AbstractVector, ε::AbstractVector; cutoff, tail = true)`; `read_forcefield(path; T = Float64)`
  - `typeindex(ff, name::AbstractString) -> Int` (throws `ArgumentError` naming the unknown type and the available names)
  - `struct Guest{T, N}`: `sites::SVector{N, SVector{3,T}}`, `types::SVector{N, Int}`, `charges::SVector{N, T}`, `tc::T`, `pc::T`, `omega::T`
  - `read_guest(path, ff; T = Float64)`
  - `tail_coefficient(ff, i, j) -> T`
  - `tail_delta(ff, counts::AbstractVector{<:Integer}, guest_counts::AbstractVector{<:Integer}, V) -> T`

- [ ] **Step 1: Failing tests**

```julia
@testitem "force field mixing and lookup" begin
    ff = ForceField(["A", "B"], [3.0, 4.0], [0.01, 0.04]; cutoff = 12.0)
    @test ff.sigma[1, 2] ≈ 3.5
    @test ff.epsilon[1, 2] ≈ 0.02
    @test PureAdsorb.typeindex(ff, "B") == 2
    @test_throws "unknown LJ type" PureAdsorb.typeindex(ff, "C")
    c = PureAdsorb.tail_coefficient(ff, 1, 2)
    @test c ≈ 0.02 * 3.5^3 * ((3.5 / 12)^9 / 3 - (3.5 / 12)^3)
end

@testitem "read trappe.yaml and co2.yaml" begin
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    @test ff.cutoff == 12.0 && ff.tail
    i = PureAdsorb.typeindex(ff, "Ac_")
    @test ff.sigma[i, i] ≈ 3.0985
    @test ff.epsilon[i, i] ≈ 0.0014311662224050347
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    @test length(g.sites) == 3
    @test g.charges == [0.7, -0.35, -0.35]
    @test g.sites[2] ≈ [-1.16, 0, 0]
    @test g.tc ≈ 303.75
end

@testitem "tail delta for a ghost guest" begin
    ff = ForceField(["A", "B"], [3.0, 4.0], [0.01, 0.04]; cutoff = 12.0)
    V = 1000.0
    counts = [10, 0]          # host: 10 A atoms
    guest = [0, 2]            # guest: 2 B sites
    c = (i, j) -> PureAdsorb.tail_coefficient(ff, i, j)
    expected = (8π / 3 / V) * (2 * (10 * 2 * c(1, 2)) + 4 * c(2, 2))
    @test PureAdsorb.tail_delta(ff, counts, guest, V) ≈ expected
end
```

- [ ] **Step 2: Run, expect `UndefVarError: ForceField`.**

- [ ] **Step 3: Implement**

```julia
struct ForceField{T}
    names::Vector{String}
    sigma::Matrix{T}
    epsilon::Matrix{T}
    cutoff::T
    tail::Bool
end

function ForceField(names::AbstractVector{<:AbstractString}, σ::AbstractVector, ε::AbstractVector; cutoff, tail = true)
    axes(names, 1) == axes(σ, 1) == axes(ε, 1) || throw(DimensionMismatch("names, σ and ε must share axes: $(axes(names)) $(axes(σ)) $(axes(ε))"))
    T = float(promote_type(eltype(σ), eltype(ε), typeof(cutoff)))
    n = length(names)
    S = Matrix{T}(undef, n, n); E = Matrix{T}(undef, n, n)
    for (a, i) in enumerate(eachindex(σ)), (b, j) in enumerate(eachindex(σ))
        S[a, b] = (σ[i] + σ[j]) / 2
        E[a, b] = sqrt(ε[i] * ε[j])
    end
    return ForceField{T}(collect(String, names), S, E, T(cutoff), tail)
end

function typeindex(ff::ForceField, name::AbstractString)
    i = findfirst(==(name), ff.names)
    isnothing(i) && throw(ArgumentError("unknown LJ type $name; force field defines $(ff.names)"))
    return i
end

# kUPS parameter files list [σ, ε] per type name; a missing σ defaults to 1 and a missing ε to 0.
function read_forcefield(path::AbstractString; T = Float64)
    d = YAML.load_file(path)
    params = d["parameters"]
    names = collect(String, keys(params))
    σ = [T(something(params[n][1], 1.0)) for n in names]
    ε = [T(something(params[n][2], 0.0)) for n in names]
    return ForceField(names, σ, ε; cutoff = T(d["cutoff"]), tail = get(d, "tail_correction", true))
end

struct Guest{T, N}
    sites::SVector{N, SVector{3, T}}
    types::SVector{N, Int}
    charges::SVector{N, T}
    tc::T
    pc::T
    omega::T
end

function read_guest(path::AbstractString, ff::ForceField; T = Float64)
    d = YAML.load_file(path)
    N = length(d["positions"])
    sites = SVector{N}(SVector{3, T}(p...) for p in d["positions"])
    types = SVector{N}(typeindex(ff, s) for s in d["symbols"])
    charges = SVector{N, T}(d["charges"]...)
    return Guest{T, N}(sites, types, charges, T(d["critical_temperature"]), T(d["critical_pressure"]), T(d["acentric_factor"]))
end

function tail_coefficient(ff::ForceField{T}, i, j) where {T}
    ε = ff.epsilon[i, j]
    iszero(ε) && return zero(T)
    s = ff.sigma[i, j]
    x = s / ff.cutoff
    return ε * s^3 * (x^9 / 3 - x^3)
end

# Change of the global analytic tail correction when a guest with `guest_counts` sites per
# species is added to a system holding `counts` particles per species.
function tail_delta(ff::ForceField{T}, counts::AbstractVector{<:Integer}, guest_counts::AbstractVector{<:Integer}, V) where {T}
    ff.tail || return zero(T)
    n = length(ff.names)
    length(counts) == length(guest_counts) == n || throw(DimensionMismatch("counts must have one entry per LJ type ($n)"))
    acc = zero(T)
    for (a, i) in enumerate(eachindex(counts)), (b, j) in enumerate(eachindex(counts))
        c = tail_coefficient(ff, a, b)
        acc += 2 * counts[i] * guest_counts[j] * c + guest_counts[i] * guest_counts[j] * c
    end
    return T(8π / 3) / T(V) * acc
end
```

- [ ] **Step 4: Run, expect 3 passes.**

- [ ] **Step 5: Commit** — `"Add ForceField with Lorentz-Berthelot tables, Guest, and the tail-correction delta"`

---

### Task 5: Ewald setup and full-energy oracle

**Files:**
- Create: `src/ewald.jl`, `test/ewald_tests.jl`

**Interfaces:**
- Produces:
  - `struct EwaldParams{T}`: `cutoff`, `precision`; `EwaldParams(; cutoff, precision = 1e-6)`
  - `ewald_alpha(cutoff, precision)` (bisection on `erfc(z) − cutoff·precision/2`)
  - `ewald_kmax(alpha, precision)`
  - `kvectors(A::SMatrix, kmax) -> (ks::Vector{SVector{3,T}}, weights::Vector{T})`
  - `pk(k², alpha, V)`
  - `erfc_dev(x)`: GPU-callable complementary error function (Numerical Recipes 3rd ed. §6.2.2 Chebyshev fit, 28 coefficients, relative error below `1.2e-12` for `Float64`). Used in all energy code.
  - `structure_factor(ks, positions, charges) -> Vector{Complex{T}}`
  - `ewald_energy(A, positions, charges, molecules::AbstractVector{<:Integer}, alpha, cutoff, ks, weights)`: full periodic energy (sr + lr + self + excl + net); calls `Base.require_one_based_indexing(positions, charges, molecules)`. Tests only.

- [ ] **Step 1: Failing tests**

```julia
@testitem "alpha and kmax follow kUPS" begin
    using SpecialFunctions
    α = PureAdsorb.ewald_alpha(12.0, 1e-6)
    @test erfc(α * 12.0) ≈ 12.0 * 5e-7 rtol = 1e-9
    @test PureAdsorb.ewald_kmax(α, 1e-6) ≈ 2α * sqrt(-log(5e-7))
end

@testitem "erfc_dev matches SpecialFunctions" begin
    using SpecialFunctions
    for x in 0:0.01:6
        @test PureAdsorb.erfc_dev(x) ≈ erfc(x) rtol = 1e-11
    end
    @test PureAdsorb.erfc_dev(-1.0) ≈ erfc(-1.0) rtol = 1e-11
end

@testitem "Madelung constant of NaCl" begin
    using StaticArrays
    a = 5.64
    A = SMatrix{3, 3}(a, 0, 0, 0, a, 0, 0, 0, a)
    base = (((0, 0, 0), 1.0), ((0.5, 0.5, 0), 1.0), ((0.5, 0, 0.5), 1.0), ((0, 0.5, 0.5), 1.0),
            ((0.5, 0, 0), -1.0), ((0, 0.5, 0), -1.0), ((0, 0, 0.5), -1.0), ((0.5, 0.5, 0.5), -1.0))
    for (rc, prec) in ((a * 0.49, 1e-8), (2a, 1e-8))
        m = PureAdsorb.min_multiplicity(A, rc)
        Asc = A * Diagonal(SVector(m))
        pos = SVector{3, Float64}[]; q = Float64[]
        for k in 0:m[3]-1, j in 0:m[2]-1, i in 0:m[1]-1, (f, c) in base
            push!(pos, A * (SVector(f...) + SVector(i, j, k))); push!(q, c)
        end
        α = PureAdsorb.ewald_alpha(rc, prec)
        ks, w = PureAdsorb.kvectors(Asc, PureAdsorb.ewald_kmax(α, prec))
        E = PureAdsorb.ewald_energy(Asc, pos, q, collect(eachindex(pos)), α, rc, ks, w)
        @test E / (4 * prod(m)) ≈ -1.747565 * PureAdsorb.KE / (a / 2) rtol = 1e-5   # per ion pair
    end
end

@testitem "energy independent of alpha" begin
    using StaticArrays, Random
    A = SMatrix{3, 3}(20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0)
    rng = Xoshiro(3)
    pos = [A * rand(rng, SVector{3, Float64}) for _ in 1:40]
    q = [isodd(i) ? 0.5 : -0.5 for i in 1:40]
    mol = collect(1:40)
    Es = map((6.0, 8.0, 9.5)) do rc
        α = PureAdsorb.ewald_alpha(rc, 1e-8)
        ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1e-8))
        PureAdsorb.ewald_energy(A, pos, q, mol, α, rc, ks, w)
    end
    @test all(e -> isapprox(e, Es[1]; rtol = 1e-6), Es)
end

@testitem "intramolecular exclusion removes the pair" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    pos = [SVector(15.0, 15.0, 15.0), SVector(16.16, 15.0, 15.0)]
    q = [0.7, -0.35]
    α = PureAdsorb.ewald_alpha(12.0, 1e-8)
    ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1e-8))
    Esame = PureAdsorb.ewald_energy(A, pos, q, [1, 1], α, 12.0, ks, w)
    Ediff = PureAdsorb.ewald_energy(A, pos, q, [1, 2], α, 12.0, ks, w)
    @test Ediff - Esame ≈ PureAdsorb.KE * 0.7 * -0.35 / 1.16 rtol = 1e-3   # periodic images make this approximate
end
```
`Diagonal` in the Madelung test needs `using LinearAlgebra`.

- [ ] **Step 2: Run, expect `UndefVarError: ewald_alpha`.**

- [ ] **Step 3: Implement**

```julia
struct EwaldParams{T}
    cutoff::T
    precision::T
end
EwaldParams(; cutoff, precision = 1e-6) = EwaldParams(promote(float(cutoff), float(precision))...)

# α such that erfc(α r_c) = r_c · ε/2, the kUPS selection when the real-space cutoff is fixed.
function ewald_alpha(cutoff, precision)
    target = cutoff * precision / 2
    lo, hi = zero(target), oftype(target, 20)
    f(z) = erfc(z) - target
    f(hi) < 0 || throw(ArgumentError("cannot reach precision $precision with cutoff $cutoff"))
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1e-12 && break
    end
    return (lo + hi) / 2 / cutoff
end

ewald_kmax(alpha, precision) = 2 * alpha * sqrt(-log(precision / 2))

pk(k2, alpha, V) = (2π / V) * exp(-k2 / (4 * alpha^2)) / k2

# Complementary error function callable inside GPU kernels: Chebyshev fit from Numerical
# Recipes 3rd ed. §6.2.2 (erfccheb), 28 coefficients, relative error below 1.2e-12.
const _ERFC_COF = (-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5,
    -2.0278578112534e-5, -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8,
    -8.5238095915e-8, 6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
    9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17)

function _erfccheb(z)
    T = typeof(z)
    t = 2 / (2 + z)
    ty = 4t - 2
    d = zero(T); dd = zero(T)
    for j in length(_ERFC_COF):-1:2
        tmp = d
        d = ty * d - dd + T(_ERFC_COF[j])
        dd = tmp
    end
    return t * exp(-z * z + (ty * d - dd + T(_ERFC_COF[1])) / 2)
end
erfc_dev(x) = x >= 0 ? _erfccheb(x) : 2 - _erfccheb(-x)

# Half-space enumeration with kUPS's weighting: n₁ ≥ 0, and every vector with n₁ > 0 stands
# in for its mirror image with weight 2. n_i = ⌈k_max L_i / 2π⌉ bounds the integer range
# exactly for a triclinic cell (L_i are the perpendicular lengths).
function kvectors(A::SMatrix{3, 3, T}, kmax) where {T}
    B = reciprocal_basis(A)
    L = perpendicular_lengths(A)
    n = ntuple(i -> ceil(Int, kmax * L[i] / (2π)), 3)
    ks = SVector{3, T}[]; w = T[]
    for n1 in 0:n[1], n2 in -n[2]:n[2], n3 in -n[3]:n[3]
        (n1 == 0 && n2 == 0 && n3 == 0) && continue
        k = B * SVector(n1, n2, n3)
        norm(k) <= kmax || continue
        push!(ks, k); push!(w, n1 == 0 ? one(T) : T(2))
    end
    return ks, w
end

function structure_factor(ks, positions, charges)
    T = float(eltype(charges))
    S = zeros(Complex{T}, length(ks))
    for (i, k) in enumerate(ks)
        acc = zero(Complex{T})
        for j in eachindex(positions, charges)
            acc += charges[j] * cis(dot(k, positions[j]))
        end
        S[i] = acc
    end
    return S
end

# Full Ewald energy of one periodic system; the reference for every kernel that evaluates a
# difference of such energies. Intramolecular pairs use the direct distance, as kUPS does.
function ewald_energy(A::SMatrix{3, 3, T}, positions, charges, molecules, alpha, cutoff, ks, weights) where {T}
    Base.require_one_based_indexing(positions, charges, molecules)
    invA = inv(A); V = volume(A)
    E_sr = zero(T); E_excl = zero(T)
    for i in eachindex(positions), j in eachindex(positions)
        j > i || continue
        if molecules[i] == molecules[j]
            E_excl -= charges[i] * charges[j] / norm(positions[j] - positions[i])
        else
            Δ = minimum_image(A, invA, positions[j] - positions[i])
            r = norm(Δ)
            r < cutoff && (E_sr += charges[i] * charges[j] * erfc_dev(alpha * r) / r)
        end
    end
    S = structure_factor(ks, positions, charges)
    E_lr = zero(T)
    for i in eachindex(ks, weights, S)
        E_lr += weights[i] * pk(dot(ks[i], ks[i]), alpha, V) * abs2(S[i])
    end
    E_self = -alpha / sqrt(T(π)) * sum(abs2, charges)
    Q = sum(charges)
    E_net = -T(π) / (2 * V * alpha^2) * Q^2
    return KE * (E_sr + E_lr + E_self + E_excl + E_net)
end
```
The `_ERFC_COF` values must be copied from Numerical Recipes 3rd ed. §6.2.2 and checked by the `erfc_dev` test; the list above is transcribed from memory of that table and the test is what validates it. If the test fails, transcribe again from the book, do not loosen the tolerance.

- [ ] **Step 4: Run, expect 5 passes.** If the Madelung test fails at `rtol = 1e-5` but passes at `1e-4`, tighten `prec` in the test to `1e-10`, not the tolerance.

- [ ] **Step 5: Commit** — `"Add Ewald summation with kUPS parameter selection and half-space k-vectors"`

---

### Task 6: Insertion energy (pose-dependent part)

**Files:**
- Create: `src/energy.jl`, `test/energy_tests.jl`

**Interfaces:**
- Consumes: `Guest`, `minimum_image`, `pk`, `erfc_dev`, `KE`.
- Produces:
  - `rotate(q::SVector{4}, v::SVector{3}) -> SVector{3}`; quaternion component order `(x, y, z, w)`. Before implementing, read `src/kups/core/utils/quaternion.py` and confirm which component Shoemake's `√u₁ cos 2πu₃` lands in; write the order in the docstring and make `random_poses!` in Task 8 produce the same order.
  - `insertion_energy(pos::SVector{3}, q::SVector{4}, guest, sigma, epsilon, cutoff, hpos, htype, hq, A, invA, alpha, ks, weights, Shost, V) -> T`: `ΔU_LJ` + real-space + reciprocal cross and self-square terms. All arguments are arrays, views or isbits so the same function runs inside the kernel.

- [ ] **Step 1: Failing tests**

```julia
@testitem "rotation preserves length and the unit quaternion is the identity" begin
    using StaticArrays, LinearAlgebra
    v = SVector(1.16, 0.0, 0.0)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, 0.0, 1.0), v) ≈ v
    q = normalize(SVector(0.3, -0.5, 0.7, 0.2))
    @test norm(PureAdsorb.rotate(q, v)) ≈ norm(v)
    # 90° about z maps x to y: q = (0, 0, sin45°, cos45°)
    @test PureAdsorb.rotate(SVector(0.0, 0.0, sqrt(0.5), sqrt(0.5)), SVector(1.0, 0.0, 0.0)) ≈ SVector(0.0, 1.0, 0.0) atol = 1e-12
end

@testitem "insertion energy equals the full-system energy difference" begin
    using StaticArrays, LinearAlgebra, Random
    ff = ForceField(["Zr_", "H_", "C_", "O_", "C", "O"], [2.78, 2.57, 3.43, 3.12, 2.8, 3.05],
        [0.003, 0.0019, 0.0046, 0.0026, 0.0023, 0.0068]; cutoff = 10.0, tail = false)
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    g0 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml")))
    g = PureAdsorb.Guest(g0.sites, SVector(5, 6, 6), g0.charges, g0.tc, g0.pc, g0.omega)
    sc = replicate(fw, PureAdsorb.min_multiplicity(fw.cell, 10.0))
    A = sc.cell; invA = inv(A); V = PureAdsorb.volume(A)
    hpos = PureAdsorb.cartesian(sc); hq = sc.charges
    htype = Int32[PureAdsorb.typeindex(ff, s * "_") for s in sc.symbols]
    α = PureAdsorb.ewald_alpha(10.0, 1e-7)
    ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1e-7))
    Sh = PureAdsorb.structure_factor(ks, hpos, hq)
    rng = Xoshiro(7)
    pos = A * rand(rng, SVector{3, Float64}); q = normalize(rand(rng, SVector{4, Float64}) .- 0.5)
    ΔU = PureAdsorb.insertion_energy(pos, q, g, ff.sigma, ff.epsilon, ff.cutoff, hpos, htype, hq, A, invA, α, ks, w, Sh, V)
    gpos = [pos + PureAdsorb.rotate(q, s) for s in g.sites]
    mol_h = collect(1:length(hpos)); mol_g = fill(0, 3)
    Ecoul = PureAdsorb.ewald_energy(A, vcat(hpos, gpos), vcat(hq, collect(g.charges)), vcat(mol_h, mol_g), α, 10.0, ks, w) -
        PureAdsorb.ewald_energy(A, hpos, hq, mol_h, α, 10.0, ks, w)
    Elj = 0.0
    for (s, t) in zip(gpos, g.types), (h, ht) in zip(hpos, htype)
        r = norm(PureAdsorb.minimum_image(A, invA, s - h))
        r < 10.0 || continue
        x = (ff.sigma[t, ht] / r)^6
        Elj += 4ff.epsilon[t, ht] * (x^2 - x)
    end
    Eself = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    Eexcl = -PureAdsorb.KE * sum(g.charges[a] * g.charges[b] / norm(gpos[a] - gpos[b]) for a in 1:3 for b in a+1:3)
    Enet = -PureAdsorb.KE * π / (2V * α^2) * ((sum(hq) + sum(g.charges))^2 - sum(hq)^2)
    @test ΔU ≈ Elj + Ecoul - Eself - Eexcl - Enet rtol = 1e-8
end
```
RUBTAK contains Zr, H, C and O only (check with `unique(fw.symbols)` in the session and extend the type list if not). The molecule ids give every host atom its own molecule and the three guest sites a shared one.

- [ ] **Step 2: Run, expect `UndefVarError: rotate`.**

- [ ] **Step 3: Implement**

```julia
# Quaternion stored as (x, y, z, w); rotation by the expanded Rodrigues form, no matrix.
function rotate(q::SVector{4}, v::SVector{3})
    u = SVector(q[1], q[2], q[3]); w = q[4]
    return v + 2 * cross(u, cross(u, v) + w * v)
end

function insertion_energy(pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff,
        hpos, htype, hq, A, invA, alpha, ks, weights, Shost, V) where {T, N}
    rc2 = cutoff * cutoff
    E_lj = zero(T); E_sr = zero(T)
    for s in 1:N
        gp = pos + rotate(q, guest.sites[s])
        gt = guest.types[s]; gq = guest.charges[s]
        for j in eachindex(hpos, htype, hq)
            Δ = minimum_image(A, invA, gp - hpos[j])
            r2 = dot(Δ, Δ)
            r2 < rc2 || continue
            σ = sigma[gt, htype[j]]; ε = epsilon[gt, htype[j]]
            x = (σ * σ / r2)^3
            E_lj += 4 * ε * (x * x - x)
            r = sqrt(r2)
            E_sr += gq * hq[j] * erfc_dev(alpha * r) / r
        end
    end
    E_lr = zero(T)
    for i in eachindex(ks, weights, Shost)
        k = ks[i]
        Sg = zero(Complex{T})
        for s in 1:N
            gp = pos + rotate(q, guest.sites[s])
            Sg += guest.charges[s] * cis(dot(k, gp))
        end
        E_lr += weights[i] * pk(dot(k, k), alpha, V) * (2 * real(conj(Shost[i]) * Sg) + abs2(Sg))
    end
    return E_lj + KE * (E_sr + E_lr)
end
```

- [ ] **Step 4: Run, expect 2 passes.**

- [ ] **Step 5: Commit** — `"Add pose-dependent insertion energy: LJ, real-space and reciprocal Ewald"`

---

### Task 7: FrameworkBatch

**Files:**
- Create: `src/batch.jl`, `test/batch_tests.jl`

**Interfaces:**
- Produces:
  - `struct FrameworkBatch{T, VP, VI, VT, VM, VK, VS, MT}` with fields, in this order: `positions::VP`, `types::VI`, `charges::VT`, `atom_offsets::VI`, `cells::VM`, `invcells::VM`, `volumes::VT`, `alphas::VT`, `ks::VK`, `kweights::VT`, `Shost::VS`, `k_offsets::VI`, `constant_offset::VT`, `sigma::MT`, `epsilon::MT`, `cutoff::T`, `nsys::Int`
  - `FrameworkBatch(fws::AbstractVector{<:Framework}, ff::ForceField, guest::Guest, ewald::EwaldParams)`: throws if `min_multiplicity(cell, max(ff.cutoff, ewald.cutoff)) != (1,1,1)` for any framework, naming the index and the required replication; maps each framework symbol to LJ type `symbol * "_"` (kUPS's UFF naming, e.g. `Zr_`)
  - `Adapt.@adapt_structure FrameworkBatch`
  - `constant_offset[s]` = `tail_delta + KE·(self + excl + net)`

- [ ] **Step 1: Failing tests**

```julia
@testitem "batch layout and offsets" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    @test b.nsys == 2
    @test b.atom_offsets == Int32[0, 3159, 6318]
    @test length(b.ks) == 2 * (b.k_offsets[2] - b.k_offsets[1])
    @test b.constant_offset[1] == b.constant_offset[2]
    @test b.types[1] == PureAdsorb.typeindex(ff, "Zr_")
end

@testitem "batch rejects a cell too small for the cutoff" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    @test_throws "replicate" FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
end

@testitem "constant offset matches the pose-independent terms" begin
    using StaticArrays, LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    α = b.alphas[1]; V = b.volumes[1]
    self = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    excl = -PureAdsorb.KE * sum(g.charges[a] * g.charges[c] / norm(g.sites[a] - g.sites[c]) for a in 1:3 for c in a+1:3)
    counts = [count(==(t), b.types) for t in eachindex(ff.names)]
    gcounts = [count(==(t), g.types) for t in eachindex(ff.names)]
    tail = PureAdsorb.tail_delta(ff, counts, gcounts, V)
    @test b.constant_offset[1] ≈ self + excl + tail     # CO2 is neutral: no net-charge term
end
```

- [ ] **Step 2: Run, expect `UndefVarError: FrameworkBatch`.**

- [ ] **Step 3: Implement**

```julia
struct FrameworkBatch{T, VP, VI, VT, VM, VK, VS, MT}
    positions::VP
    types::VI
    charges::VT
    atom_offsets::VI
    cells::VM
    invcells::VM
    volumes::VT
    alphas::VT
    ks::VK
    kweights::VT
    Shost::VS
    k_offsets::VI
    constant_offset::VT
    sigma::MT
    epsilon::MT
    cutoff::T
    nsys::Int
end
Adapt.@adapt_structure FrameworkBatch

function FrameworkBatch(fws::AbstractVector{<:Framework{T}}, ff::ForceField{T}, guest::Guest{T}, ewald::EwaldParams{T}) where {T}
    rc = max(ff.cutoff, ewald.cutoff)
    positions = SVector{3, T}[]; types = Int32[]; charges = T[]
    atom_offsets = Int32[0]; cells = SMatrix{3, 3, T, 9}[]; invcells = SMatrix{3, 3, T, 9}[]
    volumes = T[]; alphas = T[]; ks = SVector{3, T}[]; kweights = T[]; Shost = Complex{T}[]
    k_offsets = Int32[0]; constant_offset = T[]
    gcounts = [count(==(t), guest.types) for t in eachindex(ff.names)]
    α = ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = ewald_kmax(α, ewald.precision)
    for (n, fw) in pairs(fws)
        m = min_multiplicity(fw.cell, rc)
        m == (1, 1, 1) || throw(ArgumentError("framework $n is too small for cutoff $rc; replicate it by $m first"))
        pos = cartesian(fw)
        ty = Int32[typeindex(ff, s * "_") for s in fw.symbols]
        append!(positions, pos); append!(types, ty); append!(charges, fw.charges)
        push!(atom_offsets, Int32(length(positions)))
        A = fw.cell; V = volume(A)
        push!(cells, A); push!(invcells, inv(A)); push!(volumes, V); push!(alphas, α)
        kv, w = kvectors(A, kmax)
        append!(ks, kv); append!(kweights, w)
        append!(Shost, structure_factor(kv, pos, fw.charges))
        push!(k_offsets, Int32(length(ks)))
        counts = [count(==(t), ty) for t in eachindex(ff.names)]
        self = -α / sqrt(T(π)) * sum(abs2, guest.charges)
        excl = zero(T)
        for a in eachindex(guest.sites), c in eachindex(guest.sites)
            c > a || continue
            excl -= guest.charges[a] * guest.charges[c] / norm(guest.sites[a] - guest.sites[c])
        end
        Qh = sum(fw.charges); Qg = sum(guest.charges)
        net = -T(π) / (2 * V * α^2) * ((Qh + Qg)^2 - Qh^2)
        push!(constant_offset, tail_delta(ff, counts, gcounts, V) + KE * (self + excl + net))
    end
    return FrameworkBatch(positions, types, charges, atom_offsets, cells, invcells, volumes, alphas,
        ks, kweights, Shost, k_offsets, constant_offset, ff.sigma, ff.epsilon, ff.cutoff, length(fws))
end
```

- [ ] **Step 4: Run, expect 3 passes.**

- [ ] **Step 5: Commit** — `"Add FrameworkBatch: structure-of-arrays layout with Ewald tables and pose-independent offsets"`

---

### Task 8: Widom kernel, statistics, CPU backend

**Files:**
- Create: `src/widom.jl`, `test/widom_tests.jl`

**Interfaces:**
- Produces:
  - `random_poses!(rng, sys_of, rpos, quat, nsys)`: `sys_of[i] = mod1(i, nsys)`, `rpos` uniform fractional, `quat` by Shoemake in the component order fixed in Task 6
  - `@kernel widom_kernel!(ΔU, @Const(sys_of), @Const(rpos), @Const(quat), batch, guest)`
  - `struct WidomResult{T}`: `mu_ex, mu_ex_err, K_H, K_H_err, q_st, q_st_err, nsamples::Int, nblocks::Int`
  - `widom(batch, guest; T::Real, ninsert::Integer, backend = CPU(), seed = 0, chunk = 2^16, nblocks = 10) -> Vector{WidomResult}`; throws if `ninsert < 2·nblocks·nsys` or `!backend_loaded(backend)`
  - `backend_loaded(::KernelAbstractions.Backend) = false`, `backend_loaded(::CPU) = true`

- [ ] **Step 1: Failing tests**

```julia
@testitem "empty box gives ideal-gas statistics" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0, tail = false)
    g = PureAdsorb.Guest(SVector(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    r = widom(b, g; T = 300.0, ninsert = 10_000, seed = 1)[1]
    kT = PureAdsorb.KB * 300.0
    @test r.mu_ex ≈ 0 atol = 1e-12
    @test r.K_H ≈ 30.0^3 / kT
    @test r.q_st ≈ kT
    @test r.nsamples == 10_000
end

@testitem "single LJ atom matches the radial integral" begin
    using StaticArrays, QuadGK
    L = 40.0
    A = SMatrix{3, 3}(L, 0, 0, 0, L, 0, 0, 0, L)
    fw = Framework{Float64}(A, [SVector(0.5, 0.5, 0.5)], ["X"], ["X"], [0.0])
    σ, ε, rc = 3.4, 0.0103, 12.0
    ff = ForceField(["X_"], [σ], [ε]; cutoff = rc, tail = false)
    g = PureAdsorb.Guest(SVector(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = rc))
    Tk = 300.0; β = 1 / (PureAdsorb.KB * Tk)
    r = widom(b, g; T = Tk, ninsert = 4_000_000, seed = 2, nblocks = 20)[1]
    u(x) = 4ε * ((σ / x)^12 - (σ / x)^6)
    integral, _ = quadgk(x -> (1 - exp(-β * u(x))) * x^2, 1e-3, rc; rtol = 1e-10)
    expected = 1 - 4π * integral / L^3
    meanW = r.K_H * PureAdsorb.KB * Tk / L^3
    errW = r.K_H_err * PureAdsorb.KB * Tk / L^3
    @test abs(meanW - expected) < 4 * errW
end

@testitem "RUBTAK CO2 runs and is finite" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    r = widom(b, g; T = 298.15, ninsert = 2_000, seed = 3, nblocks = 4)[1]
    @test isfinite(r.mu_ex) && isfinite(r.K_H) && isfinite(r.q_st)
    @test r.K_H > 0
end

@testitem "widom rejects too few insertions and unloaded backends" begin
    using StaticArrays, KernelAbstractions
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    fw = Framework{Float64}(A, SVector{3, Float64}[], String[], String[], Float64[])
    ff = ForceField(["X_"], [3.0], [0.001]; cutoff = 12.0)
    g = PureAdsorb.Guest(SVector(SVector(0.0, 0.0, 0.0)), SVector(1), SVector(0.0), 1.0, 1.0, 0.0)
    b = FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
    @test_throws "ninsert" widom(b, g; T = 300.0, ninsert = 5, nblocks = 10)
    struct FakeBackend <: KernelAbstractions.Backend end
    @test_throws "not loaded" widom(b, g; T = 300.0, ninsert = 100, backend = FakeBackend())
end
```

- [ ] **Step 2: Run, expect `UndefVarError: widom`.**

- [ ] **Step 3: Implement**

```julia
struct WidomResult{T}
    mu_ex::T
    mu_ex_err::T
    K_H::T
    K_H_err::T
    q_st::T
    q_st_err::T
    nsamples::Int
    nblocks::Int
end

backend_loaded(::KernelAbstractions.Backend) = false
backend_loaded(::CPU) = true

# Insertions are dealt round-robin over systems so every chunk samples each system equally.
function random_poses!(rng::AbstractRNG, sys_of, rpos, quat, nsys)
    for i in eachindex(sys_of, rpos, quat)
        sys_of[i] = Int32(mod1(i, nsys))
        rpos[i] = rand(rng, eltype(rpos))
        u1, u2, u3 = rand(rng), rand(rng), rand(rng)
        a, b = sqrt(1 - u1), sqrt(u1)
        quat[i] = eltype(quat)(a * sinpi(2u2), a * cospi(2u2), b * sinpi(2u3), b * cospi(2u3))
    end
    return nothing
end

@kernel function widom_kernel!(ΔU, @Const(sys_of), @Const(rpos), @Const(quat), batch, guest)
    i = @index(Global)
    s = sys_of[i]
    a0 = batch.atom_offsets[s] + 1; a1 = batch.atom_offsets[s+1]
    k0 = batch.k_offsets[s] + 1; k1 = batch.k_offsets[s+1]
    A = batch.cells[s]; invA = batch.invcells[s]
    pos = A * rpos[i]
    e = insertion_energy(pos, quat[i], guest, batch.sigma, batch.epsilon, batch.cutoff,
        view(batch.positions, a0:a1), view(batch.types, a0:a1), view(batch.charges, a0:a1),
        A, invA, batch.alphas[s], view(batch.ks, k0:k1), view(batch.kweights, k0:k1), view(batch.Shost, k0:k1),
        batch.volumes[s])
    ΔU[i] = e + batch.constant_offset[s]
end

function widom(batch::FrameworkBatch{F}, guest::Guest{F}; T, ninsert::Integer, backend = CPU(), seed = 0,
        chunk::Integer = 2^16, nblocks::Integer = 10) where {F}
    backend_loaded(backend) || throw(ArgumentError("backend $(typeof(backend)) requested but its package is not loaded"))
    nsys = batch.nsys
    ninsert >= 2 * nblocks * nsys || throw(ArgumentError("ninsert=$ninsert is too small: need at least 2·nblocks·nsys = $(2 * nblocks * nsys)"))
    kT = F(KB * T)
    dbatch = adapt(backend, batch)
    rng = Xoshiro(seed)
    sys_of = Vector{Int32}(undef, chunk); rpos = Vector{SVector{3, F}}(undef, chunk); quat = Vector{SVector{4, F}}(undef, chunk)
    ΔU_h = Vector{F}(undef, chunk)
    dsys = adapt(backend, sys_of); drpos = adapt(backend, rpos); dquat = adapt(backend, quat); dΔU = adapt(backend, ΔU_h)
    sW = zeros(F, nsys, nblocks); sUW = zeros(F, nsys, nblocks); n = zeros(Int, nsys, nblocks)
    kern = widom_kernel!(backend)
    done = 0; block_len = cld(ninsert, nblocks)
    while done < ninsert
        m = min(chunk, ninsert - done)
        random_poses!(rng, view(sys_of, 1:m), view(rpos, 1:m), view(quat, 1:m), nsys)
        copyto!(dsys, sys_of); copyto!(drpos, rpos); copyto!(dquat, quat)
        kern(dΔU, dsys, drpos, dquat, dbatch, guest; ndrange = m)
        KernelAbstractions.synchronize(backend)
        copyto!(ΔU_h, dΔU)
        for i in 1:m
            s = sys_of[i]; blk = min(nblocks, (done + i - 1) ÷ block_len + 1)
            w = exp(-ΔU_h[i] / kT)
            sW[s, blk] += w; sUW[s, blk] += ΔU_h[i] * w; n[s, blk] += 1
        end
        done += m
    end
    return [_reduce(view(sW, s, :), view(sUW, s, :), view(n, s, :), kT, batch.volumes[s]) for s in 1:nsys]
end

function _reduce(sW, sUW, n, kT, V)
    T = eltype(sW)
    nb = length(sW)
    mW = sW ./ n; mUW = sUW ./ n
    W = sum(sW) / sum(n); UW = sum(sUW) / sum(n)
    varW = sum(abs2, mW .- W) / (nb - 1); varUW = sum(abs2, mUW .- UW) / (nb - 1)
    cov = sum((mW .- W) .* (mUW .- UW)) / (nb - 1)
    semW = sqrt(varW / nb)
    ratio = UW / W
    var_ratio = iszero(UW) ? zero(T) : ratio^2 * (varUW / UW^2 + varW / W^2 - 2cov / (UW * W)) / nb
    return WidomResult{T}(-kT * log(W), kT * semW / W, V * W / kT, V * semW / kT,
        kT - ratio, sqrt(max(var_ratio, zero(T))), sum(n), nb)
end
```
`copyto!` between a `Vector` and a device array is defined by every backend package; on the CPU backend `adapt` returns the same `Vector`, so the copies are self-copies and cheap.

- [ ] **Step 4: Run, expect 4 passes.** The radial-integral test uses 4 million insertions on the CPU; if it takes more than ~20 s warm, reduce to 1 million and keep the tolerance in units of the reported standard error.

- [ ] **Step 5: Commit** — `"Add Widom insertion kernel with block-averaged statistics on the CPU backend"`

---

### Task 9: GPU extensions and CPU-equals-GPU test

**Files:**
- Create: `ext/PureAdsorbCUDAExt.jl`, `ext/PureAdsorbAMDGPUExt.jl`, `test/gpu_tests.jl`
- Modify: `Project.toml` (`[weakdeps]`, `[extensions]`, `[compat]`)

- [ ] **Step 1: Register weak deps with Pkg**

```bash
julia --project=/home/el_oso/Documents/claude/PureAdsorb.jl -e 'using Pkg; Pkg.add("CUDA"); Pkg.add("AMDGPU")'
```
then move both entries from `[deps]` to `[weakdeps]` by hand (the UUIDs come from Pkg, the section move is an authored edit), add
```toml
[extensions]
PureAdsorbCUDAExt = "CUDA"
PureAdsorbAMDGPUExt = "AMDGPU"
```
and compat bounds `CUDA = "5"`, `AMDGPU = "1, 2"`. Run `julia --project -e 'using Pkg; Pkg.resolve()'`.

- [ ] **Step 2: Extensions**

```julia
module PureAdsorbCUDAExt
using CUDA, PureAdsorb
PureAdsorb.backend_loaded(::CUDA.CUDABackend) = true
end
```
```julia
module PureAdsorbAMDGPUExt
using AMDGPU, PureAdsorb
PureAdsorb.backend_loaded(::AMDGPU.ROCBackend) = true
end
```
`Adapt` and `KernelAbstractions` already move a `FrameworkBatch` to `CuArray`/`ROCArray` through the `@adapt_structure` rule; the extensions only register the backend.

- [ ] **Step 3: GPU test**

```julia
@testitem "GPU matches CPU on the same poses" tags = [:gpu] begin
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    if isnothing(backend) && !isnothing(Base.find_package("AMDGPU"))
        @eval using AMDGPU
        AMDGPU.functional() && (backend = ROCBackend())
    end
    isnothing(backend) && error("no functional GPU backend; this item must run on a GPU host")
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    rc = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4)[1]
    rg = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4, backend)[1]
    @test rg.mu_ex ≈ rc.mu_ex rtol = 1e-10
    @test rg.K_H ≈ rc.K_H rtol = 1e-10
    @test rg.q_st ≈ rc.q_st rtol = 1e-10
end
```
The tolerance is `1e-10` rather than exact equality because GPU `exp` and `sincos` may differ in the last ulp. If it fails, print both values: an ulp-level gap gets `1e-8` and a comment naming the cause; anything larger is a bug.

Add `CUDA` and `AMDGPU` to `test/Project.toml` with `ion add` from inside `test/`. Run on galen (`ROCBackend`) through a JuliaMCP session there with `max_workers = 1`; the default runner skips the item by its tag.

- [ ] **Step 4: Commit** — `"Add CUDA and AMDGPU backends as package extensions"`

---

### Task 10: Quality gates: StrictMode audit, generic indexing, docs, CI

**Files:**
- Create: `bench/Project.toml`, `bench/audit.jl`, `test/generic_tests.jl`, `docs/make.jl`, `docs/src/index.md`, `docs/src/api.md`, `.github/workflows/CI.yml`

- [ ] **Step 1: Generic indexing tests**

```julia
@testitem "CPU entry points accept OffsetArray and view inputs" begin
    using OffsetArrays, StaticArrays
    σ = OffsetArray([3.0, 4.0], 0:1); ε = OffsetArray([0.01, 0.04], 0:1)
    ff = ForceField(OffsetArray(["A", "B"], 0:1), σ, ε; cutoff = 12.0)
    @test ff.sigma[1, 2] ≈ 3.5
    counts = view([10, 0, 99], 1:2); gc = view([0, 2, 99], 1:2)
    @test PureAdsorb.tail_delta(ff, counts, gc, 1000.0) ≈ PureAdsorb.tail_delta(ff, [10, 0], [0, 2], 1000.0)
    A = SMatrix{3, 3}(10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0)
    pos = OffsetArray([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], 0:1)
    q = OffsetArray([1.0, -1.0], 0:1)
    ks, w = PureAdsorb.kvectors(A, 2.0)
    @test length(PureAdsorb.structure_factor(ks, pos, q)) == length(ks)
    @test_throws ArgumentError PureAdsorb.ewald_energy(A, pos, q, [1, 2], 0.3, 4.0, ks, w)
end
```

- [ ] **Step 2: StrictMode audit**

```bash
cd bench && ion add StrictMode Chairmarks JSON CairoMakie && julia --project=. -e 'using Pkg; Pkg.develop(path = "..")' && cd ..
```
Pin `JSON = "1.6"` in `bench/Project.toml`'s compat.

`bench/audit.jl`:
```julia
using PureAdsorb, StrictMode, StaticArrays
assert_enabled()
mode = get(ENV, "STRICT_MODE", "fast") == "full" ? :full : :fast
G = PureAdsorb.Guest{Float64, 3}
V3 = Vector{SVector{3, Float64}}
M3 = SMatrix{3, 3, Float64, 9}
findings = vcat(
    check(PureAdsorb.insertion_energy,
        (SVector{3, Float64}, SVector{4, Float64}, G, Matrix{Float64}, Matrix{Float64}, Float64,
            V3, Vector{Int32}, Vector{Float64}, M3, M3, Float64, V3, Vector{Float64}, Vector{ComplexF64}, Float64);
        guarantees = (:typestable, :noalloc), mode),
    check(PureAdsorb.minimum_image, (M3, M3, SVector{3, Float64}); guarantees = (:typestable, :noalloc), mode),
    check(PureAdsorb.rotate, (SVector{4, Float64}, SVector{3, Float64}); guarantees = (:typestable, :noalloc), mode),
    check(PureAdsorb.erfc_dev, (Float64,); guarantees = (:typestable, :noalloc), mode),
)
exit(nfailures(findings))
```
Run it from the `bench` env in a JuliaMCP session first (`mode = :fast`), fix every finding, then once with `STRICT_MODE=full` cold as the gate.

- [ ] **Step 3: Docs and CI**

`docs/make.jl` with DocumenterVitepress (`ion add Documenter DocumenterVitepress` from inside `docs/`, plus `Pkg.develop(path = "..")`), `docs/src/index.md` (README content), `docs/src/api.md` (`@autodocs`). Deploy through `DocumenterVitepress.deploydocs`. `.github/workflows/CI.yml`: Julia `1.10` and `1` matrix on `ubuntu-latest`, `julia-actions/setup-julia`, `julia-actions/julia-buildpkg`, `julia-actions/julia-runtest`, `julia-actions/julia-processcoverage`, `coverallsapp/github-action@v2` with `path-to-lcov: lcov.info`. Coveralls badge in the README.

- [ ] **Step 4: Cold gate and commit**

```bash
runic -i . && julia --project -e 'using Pkg; Pkg.test()'
git add -A && git commit -m "Add StrictMode audit, generic-indexing tests, docs skeleton and CI"
```

---

### Task 11: kUPS reference numbers and the reference test

**Blocked until the user explicitly approves the kUPS-as-stopwatch exception and the 4070 is installed in the eGPU enclosure on neuromancer.** Everything else in this plan proceeds without it.

**Files:**
- Create: `bench/run_kups.sh`, `bench/kups_widom_reference.yaml`, `test/reference/rubtak_co2_kups.json`, `test/reference_tests.jl`

- [ ] **Step 1: kUPS config matching Milestone A**

kUPS runs Widom inside its NVT driver. With `init_adsorbates: [0]` and no exchange moves the host stays empty, so the reported `K_H` is the zero-loading value (confirmed in `src/kups/application/mcmc/mcmc_widom.py`). `bench/kups_widom_reference.yaml` is the shipped `examples/mcmc_widom.yaml` with `num_cycles: 2000`, `num_warmup_cycles: 0`, `num_displacements_per_cycle: 0`, `num_widom_per_cycle: 500`, `seed: 42`: one million ghost insertions.

- [ ] **Step 2: `bench/run_kups.sh`**

A shell script that, from a kUPS checkout at `$KUPS` with its own environment already created, runs the `mcmc_widom` entry point on `kups_widom_reference.yaml`, tees the printed result dict to `bench/results/kups_widom_<host>_<date>.log`, and prints the kUPS commit hash. The three `(mean, sem, n_blocks)` triples for `excess_chemical_potential`, `henry_coefficient`, `heat_of_adsorption` are copied by hand into `test/reference/rubtak_co2_kups.json` together with the kUPS commit, JAX version, driver version and GPU name. The script is documentation of an external measurement; PureAdsorb never calls it.

- [ ] **Step 3: Reference test**

```julia
@testitem "RUBTAK CO2 matches kUPS within combined error" tags = [:slow] begin
    using JSON
    ref = JSON.parsefile(joinpath(pkgdir(PureAdsorb), "test", "reference", "rubtak_co2_kups.json"))
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    r = widom(b, g; T = 298.15, ninsert = 1_000_000, seed = 42, nblocks = 20)[1]
    for (ours, err, key) in ((r.mu_ex, r.mu_ex_err, "mu_ex"), (r.K_H, r.K_H_err, "K_H"), (r.q_st, r.q_st_err, "q_st"))
        m, s = ref[key]["mean"], ref[key]["sem"]
        @test abs(ours - m) < 3 * hypot(err, s)
    end
end
```
Add `JSON` (`1.6`) to `test/Project.toml`. The runner already excludes `:slow`; run it explicitly with `julia_run_testitems`.

- [ ] **Step 4: Commit** — `"Add kUPS reference numbers for RUBTAK/CO2 and the cross-code reference test"`

If the reference test fails, suspects in order: the LJ type mapping of framework symbols (`Zr_` etc.), the tail-correction species counts (kUPS counts every particle in the system), the exclusion-term distance convention, and the k-vector enumeration bound. Compare per-term energies for one fixed pose against a temporary print in kUPS's Ewald potential (throwaway, not committed).

---

### Task 12: Benchmark harness and plots

**Files:**
- Create: `bench/widom_bench.jl`, `bench/run_headtohead.sh`, `bench/plot_widom.jl`, `bench/results/README.md`

- [ ] **Step 1: Julia timing script**

```julia
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions
BLAS.set_num_threads(1)
backend_name = get(ENV, "PA_BACKEND", "cpu")
if backend_name == "cuda"
    using CUDA; backend = CUDABackend(); gpu = CUDA.name(CUDA.device())
elseif backend_name == "rocm"
    using AMDGPU; backend = ROCBackend(); gpu = AMDGPU.HIP.name(AMDGPU.device())
else
    backend = CPU(); gpu = ""
end
fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
sc = replicate(fw, (3, 3, 3))
samples = []
for nsys in (1, 64), ninsert in (10^4, 10^5, 10^6, 10^7)
    nsys * 20 <= ninsert || continue
    b = FrameworkBatch(fill(sc, nsys), ff, g, EwaldParams(cutoff = 12.0, precision = 1e-6))
    widom(b, g; T = 298.15, ninsert = 200 * nsys, backend)          # warm-up: compile
    bm = @be widom($b, $g; T = 298.15, ninsert = $ninsert, backend = $backend) seconds = 30 samples = 10
    push!(samples, (; nsys, ninsert, backend = backend_name, times_s = [s.time for s in bm.samples]))
    println("nsys=$nsys ninsert=$ninsert median=$(median(bm).time) s"); flush(stdout)
end
meta = (; host = gethostname(), julia = string(VERSION), date = string(now()), gpu, backend = backend_name)
open(joinpath(@__DIR__, "results", "pureadsorb_widom_$(meta.host)_$(backend_name)_$(Dates.format(now(), "yyyymmdd")).json"), "w") do io
    JSON.print(io, (; meta, samples), 2)
end
```
Add a second measurement of the kernel alone (one prepared chunk, `KernelAbstractions.synchronize` after the launch, timed with `@be`) saved under `kernel_only_s`, so the report separates kernel throughput from RNG and transfer overhead.

- [ ] **Step 2: Head-to-head driver**

`bench/run_headtohead.sh`: for each grid point, alternates one kUPS run (through `run_kups.sh` with `nsys` copies of the host in `hosts:` and `num_widom_per_cycle × num_cycles = ninsert`, wall time from `/usr/bin/time -f %e`) with one PureAdsorb CUDA run, ten repetitions, after one untimed warm-up of each. Sets the CPU governor to `performance` at the start and restores it at the end. kUPS's numbers are written by hand from the log to `bench/results/kups_widom_timing_<host>_<date>.json` with the same fields as the Julia JSON.

- [ ] **Step 3: Plot from JSON only**

`bench/plot_widom.jl` reads every `bench/results/*.json`, computes insertions per second per sample, and draws one CairoMakie violin per `(ninsert, code)` for `nsys = 1` and a second panel for `nsys = 64`, with the R9700 Julia-only series as a third color when present. Saves `bench/results/widom_throughput.png`. Never runs a benchmark.

- [ ] **Step 4: `bench/results/README.md`**

States the protocol (GPU events, interleaving, governor, `BLAS` threads), lists the machines and GPUs, and says how to regenerate the plot: `julia --project=bench bench/plot_widom.jl`.

- [ ] **Step 5: Commit** — `"Add Widom throughput benchmark, head-to-head driver and JSON-driven plots"`

---

## Self-review

**Spec coverage.** Structures (Tasks 3, 4), Batch (7), Energy (5, 6), Widom kernel and statistics (8), backends (9), error handling at construction (3, 4, 7, 8), every oracle in the spec's table (Madelung 5, LJ closed form 4/6, empty box 8, 1-D integral 8, kUPS 11, CPU = GPU 9, OffsetArray/view 10), StrictMode gate (10), benchmark protocol and JSON-only plots (12), repository layout (1). The PorousMaterials.jl methane-in-IRMOF-1 oracle from the spec is deliberately absent: the package is unmaintained since 2022 and may not install on current Julia; it is deferred to Milestone C, where the GCMC isotherm comparison needs it anyway.

**Placeholders.** None; every step carries code or an exact command. The one transcription risk, the `_ERFC_COF` table, is guarded by a test at `rtol = 1e-11`.

**Type consistency.** `Guest{T,N}` field order `(sites, types, charges, tc, pc, omega)` is identical in Tasks 4, 6, 8. `FrameworkBatch` positional constructor order in Task 7 matches its declaration. `insertion_energy` argument order in Task 6 matches the kernel call in Task 8 and the audit in Task 10. `widom` uses type parameter `F` so the keyword `T` (temperature) does not shadow it.
