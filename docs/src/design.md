# Design

## Architecture

    CIF, YAML  ──►  Framework / ForceField / Guest        (CPU structs, src/structure.jl,
                                                             src/forcefield.jl)
               ──►  FrameworkBatch                          (structure-of-arrays + offsets +
                                                             precomputed Ewald tables, CPU,
                                                             src/batch.jl)
               ──►  adapt(backend, batch)                   one upload per widom() call
               ──►  for each chunk of insertions:
                       host RNG ─► positions, quaternions ─► upload
                       widom_kernel!(ΔU, batch, guest)       one KernelAbstractions kernel
                       download ΔU ─► accumulate per-system block sums
               ──►  WidomResult per system                  (host-side reduction, src/widom.jl)

Four units, each testable alone:

**Structures** (`src/structure.jl`, `src/forcefield.jl`, `src/cell.jl`) — `Framework{T}` (cell
matrix, fractional coordinates, labels, charges), `ForceField{T}` (per-pair σ/ε tables from
Lorentz–Berthelot mixing), `Guest{T,N}` (rigid site geometry, charges, critical constants).
Plain Julia structs, CPU only, generic over the element type `T`.

**Batch** (`src/batch.jl`) — `FrameworkBatch{T,...}` concatenates every framework's atoms into
flat arrays (`positions`, `types`, `charges`) with an `atom_offsets` vector marking where each
system's atoms start, and likewise `ks`/`kprefactor`/`Shost` with `k_offsets` for the
per-system reciprocal-space tables. `kprefactor[i]` (the weight times the reciprocal-space
prefactor `pk`) and `Shost` (the host structure factor) are computed once per framework at
batch-construction time, since neither depends on the insertion pose. `constant_offset[n]`
similarly collects every pose-independent energy term for system `n` (tail-correction change,
guest self-energy, guest intramolecular exclusion, net-charge correction), so the kernel adds
one precomputed scalar per insertion instead of recomputing these every time.
`Adapt.@adapt_structure` makes the whole batch backend-portable: `adapt(backend, batch)`
produces a `FrameworkBatch` whose array fields are of `backend`'s array type, with the
structure unchanged.

**Energy** (`src/energy.jl`, `src/ewald.jl`) — `insertion_energy` evaluates the Lennard-Jones
and Ewald real/reciprocal terms of one guest pose against one system's slice of the batch
arrays. It takes only isbits scalars, `SVector`s, and plain array/view arguments, so it runs
identically whether called from Julia on the CPU or compiled into a GPU kernel.

**Widom kernel and statistics** (`src/widom.jl`) — `widom_kernel!` is the single
`@kernel function`: one work-item computes one insertion's `ΔU` by looking up its system from
a per-insertion index, slicing that system's batch arrays, and calling `insertion_energy`.
Insertion `g` of the global `1:ninsert` sequence is assigned to system
`mod1((g - 1) ÷ run + 1, nsys)` (`sys_of_index`): `run` consecutive insertions share a system
before the assignment cycles to the next one, so work-items adjacent in `g` — and so adjacent on
the device — read the same framework's tables. Random poses (fractional position, Shoemake
quaternion) are generated on the host with `Random.Xoshiro` in chunks (`random_poses!`) and
uploaded before each kernel launch; results are downloaded and accumulated into per-system,
per-block sums on the host, with each system's blocks drawn from that system's own sample order
(`system_counts` gives the exact per-system count up front). `widom` drives this loop and
reduces the accumulated sums into a `WidomResult` per system (`_reduce`).

## What runs where

Setup (CIF/YAML parsing, `FrameworkBatch` construction, the Ewald parameter derivation) and
the final block-statistics reduction run on the CPU in plain Julia — none of it needs to run
on a GPU, and all of it runs once per `widom` call rather than once per insertion. The kernel
itself calls ordinary, generic Julia functions (`insertion_energy`, `minimum_image`, `rotate`,
`erfc_dev`); `KernelAbstractions` compiles that same source per backend, so nothing in `src/`
branches on which backend is active. The package has no CUDA or AMDGPU dependency: a caller
loads `CUDA.jl` or `AMDGPU.jl` themselves and passes `backend = CUDABackend()` or
`ROCBackend()` to `widom`.

## The neutral-guest shortcut

`FrameworkBatch` construction checks `all(iszero, guest.charges)` once. When the guest carries
no charge, it has no Coulomb interaction with anything, so no reciprocal-space table
(`ks`, `kprefactor`, `Shost`) is built for that batch at all — `k_offsets` stays all zeros and
the kernel's reciprocal-space loop runs zero iterations for every insertion.

## Fail-fast checks at construction

Every check below runs once, outside the kernel, so the kernel itself never needs to validate
its inputs:

- `cell_matrix`: positive cell lengths, a non-degenerate γ angle, and a geometrically valid
  cell (positive radicand).
- `read_cif`: space group must be P1; the `_atom_site_charge` column is required.
- `replicate`: replication factors must be ≥ 1.
- `FrameworkBatch`: each framework's `min_multiplicity` against
  `max(LJ cutoff, Ewald cutoff)` must already be `(1,1,1)` — the caller must replicate a
  too-small cell first.
- `ewald_alpha`: raises if the requested precision is unreachable with the given cutoff.
- `tail_delta`: the per-type count vectors must match the force field's number of LJ types
  (`DimensionMismatch`).
- `widom`: `nblocks >= 2`, `chunk >= 1`, an explicit `run` keyword must lie in
  `1:(ninsert ÷ nsys)`, every system's exact insertion count must be at least `2·nblocks`, and
  the batch's index-matched array groups (`positions`/`types`/`charges` and
  `ks`/`kprefactor`/`Shost`) must share axes (`DimensionMismatch`).

## GPU-compile constraint

`insertion_energy`'s host-atom and k-vector loops use single-array `eachindex(hpos)` /
`eachindex(ks)`, not the multi-array form `eachindex(hpos, htype, hq)`. The multi-array form
additionally checks that all its arguments share axes, and its mismatch branch builds an
error string — a piece of code GPUCompiler cannot compile, which failed `widom_kernel!`'s
compilation for CUDA specifically (`InvalidIRError`) even though the same call happened to
survive on ROCm. `hpos`/`htype`/`hq` and `ks`/`kprefactor`/`Shost` are always index-matched by
construction (slices of the same `FrameworkBatch` arrays sharing one pair of offsets), so the
single-array form is exact for this call site, not an approximation — the axes check that
`widom` still performs (see above) runs once outside the kernel instead.

Kernels never throw for any other reason either: every check that could fail is performed
before the kernel launches.
