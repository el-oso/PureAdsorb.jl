# Design

## Architecture

    CIF, YAML  ──►  Framework / ForceField / Guest        (CPU structs, src/structure.jl,
                                                             src/forcefield.jl)
               ──►  FrameworkBatch                          (structure-of-arrays + offsets +
                                                             precomputed Ewald tables + hard-core
                                                             rejection bounds, CPU, src/batch.jl)
               ──►  adapt(backend, batch)                   one upload per widom() call
               ──►  for each chunk of insertions:
                       host RNG ─► positions, quaternions ─► upload
                       hardcore_kernel!(flags, batch, guest, ρ², reach)   phase 0: flag rejects
                       download flags ─► build survivor index ─► upload
                       widom_kernel!(ΔU, batch, guest, survivor)         phase 1: energy, survivors only
                       download ΔU ─► accumulate per-system block sums (rejected: weight 0)
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

Lennard-Jones types are remapped to a compact index covering only the types actually present
(any framework's atoms, union the guest's sites): `types`, `sigma` and `epsilon` all use this
index instead of the full force field's, and `compact_to_orig` maps back to the force field's
own index for error messages built at construction. The guest's site types are remapped the
same way (`guest_types`); `guest_types_orig`, `guest_sites_orig` and `guest_charges_orig` record
the guest exactly as passed in, so `widom` can verify a later call passes the same guest the
batch was built for and throw a clear error otherwise (every precomputed quantity below depends
on the guest's sites and charges, not only its types, so a mismatch would otherwise give silently
wrong physics rather than an error).

`ks`/`kprefactor`/`Shost` hold only the k-vectors coupled to each framework's replication (see
`docs/src/theory.md`): for CO2 in RUBTAK 3×3×3 that is 190 of the unreplicated cell's 4587
k-vectors. Since a framework's `replication` is taken on trust, `FrameworkBatch` samples up to
32 of the *uncoupled* k-vectors and checks that the framework's own host structure factor is
negligible there, throwing if it is not: a framework whose atoms are not actually translational
copies under the claimed `replication` would otherwise have k-vectors dropped and its energies
shifted silently. This sample gives high probability, not certainty, of catching a false claim.
The guest's own reciprocal-space self term, which depends on orientation but
not on the host, is evaluated at 64 fixed orientations (`Xoshiro(0x5e1f)`, the same Shoemake
construction `random_poses!` uses); its mean over those orientations is folded into
`constant_offset`, and `self_term_halfrange` records half their spread — an estimate from that
finite sample, not a bound on the true continuous-orientation range (a continuous orientation
reaches roughly 1.3 times `self_term_halfrange` away from the mean). `FrameworkBatch` throws if
`2 · self_term_halfrange` exceeds `1e-3 · KB · 300 K` for a guest/framework pair, since the
orientation-averaged approximation is then no longer accurate enough.

**Cell list.** Each system's atoms are stored sorted into a grid of `ncells[n]` cells along the
stored cell's three axes, one cell per `cellwidth` Å of perpendicular length (`grid_dims`,
`FrameworkBatch`'s `cellwidth` keyword, default 3 Å): `n_i = max(1, floor(L_i / cellwidth))`. A
stable counting sort (`cell_sort`) permutes `positions`/`types`/`charges` into cell order (linear
index `i + n1*(j + n2*k)`, 0-based, over the fractional coordinate wrapped into `[0,1)`), so a
cell's atoms are the contiguous range `cell_offsets[c+1]+1:cell_offsets[c+2]` (shifted by the
system's `atom_offsets` entry). `cell_offsets` concatenates every system's `prod(ncells)+1` local
offsets, ragged, with `cellgrid_offsets` marking where each system's block starts — the same
pattern `atom_offsets`/`k_offsets` already use. This cell list now serves only the hard-core
rejection stage's phase-0 kernel (below): the energy itself no longer walks a stencil. The
construction guard is `min_multiplicity(cell, max(ff.cutoff, ewald.cutoff) + r_guest) == (1,1,1)`
(`r_guest` the guest's largest site distance from its reference point), needed because a single
minimum image per host atom (see `insertion_energy` below) is only exact when every contributing
pair's separation stays under half the cell's perpendicular length.

**Energy** (`src/energy.jl`, `src/ewald.jl`) — `insertion_energy` evaluates the Lennard-Jones
and Ewald real/reciprocal terms of one guest pose against one system's atoms, looping linearly
over `positions[(atom_base+1):(atom_base+natoms)]`: one `minimum_image` per host atom, with each
guest site's own (already rotated) offset added directly without a further minimum image — valid
exactly when the construction guard above holds. This linear form replaced a cell-list stencil
walk once measurement showed it was already the fastest form for a framework this size (the
efficiency design's E2 cellwidth measurements). `insertion_energy` takes only isbits scalars,
`SVector`s, and plain array/view arguments, so it runs identically whether called from Julia on
the CPU or compiled into a GPU kernel.

**Hard-core rejection** (`src/reject.jl`) — before computing an insertion's energy, `widom`
checks whether a rigorous lower bound on it already exceeds the point past which its Boltzmann
weight underflows to exactly `0.0` in the working float type; if so, the weight is recorded as
zero without ever calling `insertion_energy`. `theta_F(T)` is that underflow point, found by
bisection on the float grid. `hardcore_bound` (`B_s`) sums, over every guest-site/host-atom pair
in a system plus a reciprocal-space cross-term bound, the most negative energy that pair could
possibly contribute anywhere in its domain; it is computed once per system at `FrameworkBatch`
construction, alongside `kmin_table` (`K_min(a,t)`, the most negative Coulomb prefactor over a
system's atoms of a given compact type, for a given guest site), since neither depends on
temperature. `find_rho2` combines both with the temperature-dependent underflow margin into a
squared rejection radius `ρ_at²`, on every `widom` call (`build_rejection_tables`): any guest
site within `ρ_at` of any host atom of type `t` guarantees rejection is safe.

**Widom kernel and statistics** (`src/widom.jl`) — each chunk runs two `@kernel function`s.
`hardcore_kernel!` (phase 0) flags every insertion whose guest sites all stay outside the
rejection radius of every nearby host atom: each site scans the phase-0 cell list's stencil
around ITS OWN home cell (reach sized from the call's largest `ρ_at`, typically 27 cells), not a
single stencil shared by the whole pose — a site far enough from the pose's reference point could
otherwise have a nearby host atom fall outside a stencil centered there instead. A site's
fractional coordinate is wrapped into `[0,1)` before finding its home cell, since it can lie
outside the stored cell entirely. The host downloads the flags, builds the list of surviving
insertion indices in order, and uploads it; `widom_kernel!` (phase 1) then computes `ΔU` for
survivors only, reading each one's pose through the survivor index and writing back at that
insertion's original position, so a rejected insertion's `ΔU` entry is simply never touched. A
chunk with no survivors launches phase 1 not at all. Insertion `g` of the global `1:ninsert`
sequence is assigned to system
`mod1((g - 1) ÷ run + 1, nsys)` (`sys_of_index`): `run` consecutive insertions share a system
before the assignment cycles to the next one, so work-items adjacent in `g` — and so adjacent on
the device — read the same framework's tables. Random poses (fractional position, Shoemake
quaternion) are generated on the host with `Random.Xoshiro` in chunks (`random_poses!`) and
uploaded before each kernel launch; results are downloaded and accumulated into per-system,
per-block sums on the host (`boltzmann_weight`, forcing the energy-weighted product to exactly
zero whenever the weight itself is, rejected or not — a guest site within about 1e-3 Å of a host
atom overflows the Lennard-Jones term to `Inf` in Float32, and `Inf * 0.0` is `NaN`), with each
system's blocks drawn from that system's own sample order (`system_counts` gives the exact
per-system count up front). `widom` drives this loop and reduces the accumulated sums into a
`WidomResult` per system (`_reduce`). `widom_singlephase` (non-exported, test-only) runs every
insertion through phase 1 directly, skipping phase 0 entirely, so its results can be checked for
exact equality against `widom`'s.

## What runs where

Setup (CIF/YAML parsing, `FrameworkBatch` construction, the Ewald parameter derivation) and
the final block-statistics reduction run on the CPU in plain Julia — none of it needs to run
on a GPU, and all of it runs once per `widom` call rather than once per insertion. The kernel
itself calls ordinary, generic Julia functions (`insertion_energy`, `minimum_image`, `rotate`,
`pair_erfc_dev`); `KernelAbstractions` compiles that same source per backend, so nothing in `src/`
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
  `max(LJ cutoff, Ewald cutoff) + r_guest` (`r_guest` the guest's largest site distance from its
  reference point) must already be `(1,1,1)` — the caller must replicate a
  too-small cell first. A framework's claimed `replication` must be consistent with its host
  structure factor on a sample of uncoupled k-vectors (naming the framework index, the claimed
  replication and the offending value). `2 · self_term_halfrange` must not exceed
  `1e-3 · KB · 300 K` for any guest/framework pair (naming the estimate, the factor, the guest
  and the framework index). `α · ewald.cutoff` must not exceed `PAIR_ERFC_XMAX = 4` (naming both
  values and the two ways out: a looser `ewald.precision` or a shorter `ewald.cutoff`) —
  `insertion_energy`'s screened-Coulomb series (`pair_erfc_dev`) is only fitted up to that bound.
- `ewald_alpha`: raises if the requested precision is unreachable with the given cutoff.
- `tail_delta`: the per-type count vectors must match the force field's number of LJ types
  (`DimensionMismatch`).
- `widom`: the passed `guest` must match the guest `batch` was built from in types, sites and
  charges (naming the mismatch); `nblocks >= 2`, `chunk >= 1`, an explicit `run` keyword must lie
  in `1:(ninsert ÷ nsys)`, every system's exact insertion count must be at least `2·nblocks`, the
  batch's index-matched array groups (`positions`/`types`/`charges` and
  `ks`/`kprefactor`/`Shost`) must share axes, `ncells` must have one entry per system,
  `cellgrid_offsets` must have `nsys+1` entries whose last equals `length(cell_offsets)`, and
  each system's last local `cell_offsets` entry must equal its atom count (all
  `DimensionMismatch`, naming the numbers).

## GPU-compile constraint

`insertion_energy`'s reciprocal-space loop uses single-array `eachindex(ks)`, not the
multi-array form `eachindex(ks, kprefactor, Shost)`. The multi-array form additionally checks
that all its arguments share axes, and its mismatch branch builds an error string — a piece of
code GPUCompiler cannot compile, which failed `widom_kernel!`'s compilation for CUDA
specifically (`InvalidIRError`) even though the same call happened to survive on ROCm.
`ks`/`kprefactor`/`Shost` are always index-matched by construction (slices of the same
`FrameworkBatch` arrays sharing one pair of offsets), so the single-array form is exact for this
call site, not an approximation — the axes check that `widom` still performs (see above) runs
once outside the kernel instead. The cell-list stencil's atom loop indexes `positions`/`types`/
`charges` (the whole batch's arrays) with an explicit integer range computed from `cell_offsets`,
never `eachindex` over those arrays, since a work-item only ever touches its own system's slice.

Kernels never throw for any other reason either: every check that could fail is performed
before the kernel launches.
