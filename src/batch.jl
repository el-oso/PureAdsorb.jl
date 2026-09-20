"""
    FrameworkBatch{T}

Structure-of-arrays layout for a set of independent framework systems, so a single GPU kernel
launch can evaluate an insertion in every system at once. Per-system slices are
`atom_offsets[n]+1:atom_offsets[n+1]` into positions/types/charges and
`k_offsets[n]+1:k_offsets[n+1]` into ks/kprefactor/Shost. Within a system's slice, atoms are
sorted by cell-list cell (see below), not CIF order. `cutoff` (Å) truncates the LJ sum and
`ewald_cutoff` (Å) truncates the real- and reciprocal-space Ewald sums; they may differ.

`ks`/`kprefactor`/`Shost` hold only the k-vectors coupled to each system's replication: those
whose integer reciprocal-lattice coefficients are all divisible by the corresponding
replication factor, since only those repeat identically across the copies making up the
supercell and so carry a nonzero host structure factor. A framework's `replication` is taken
on trust everywhere else, so `FrameworkBatch` checks it: on a sample of up to 32 of the
*uncoupled* k-vectors, the framework's own host structure factor must be negligible, or the
framework's atoms are not actually the translational copies `replication` claims and
`FrameworkBatch` throws rather than silently dropping k-vectors and shifting energies.
`self_term_halfrange[n]` (energy units) is half the max-min spread, over 64 fixed guest
orientations, of the guest self term `KE Σ_k pref_k |S_g(k)|²` taken over the FULL (unfiltered)
k set of system `n`; its mean over those orientations is folded into `constant_offset[n]` in
place of recomputing the self term per insertion. This half-range is an estimate from that
finite sample, not a bound on the true continuous-orientation range — a continuous orientation
can reach roughly 1.3 times `self_term_halfrange` away from the mean.

A sample of up to 32 k-vectors gives high probability, not certainty, that a false
`replication` claim is caught (see `verify_replication`).

Each system's atoms are stored sorted into a grid of `ncells[n]` cells along the stored cell's
three axes (fractional coordinates, wrapped into [0,1)), so that a cell is a contiguous range of
`positions`/`types`/`charges`. `cell_offsets` holds each system's `prod(ncells[n]) + 1` local
offsets back to back (system `n`'s block starts at `cellgrid_offsets[n]+1`; local offset `c`'s
atom range is `atom_offsets[n] + cell_offsets[cellgrid_offsets[n] + c] + 1` through
`atom_offsets[n] + cell_offsets[cellgrid_offsets[n] + c + 1]`, cell index `c` linear in
`i + ncells[n][1]*(j + ncells[n][2]*k)`, 0-based). `reach[n]` is the stencil half-width (in
cells) `insertion_energy` visits around an insertion's home cell, sized so that no atom pair
within `cutoff`/`ewald_cutoff` of any guest site is missed.
"""
struct FrameworkBatch{T, VP, VI, VT, VM, VK, VS, MT, VN}
    positions::VP
    types::VI
    charges::VT
    atom_offsets::VI
    cells::VM
    invcells::VM
    volumes::VT
    alphas::VT
    ks::VK
    kprefactor::VT
    Shost::VS
    k_offsets::VI
    constant_offset::VT
    self_term_halfrange::VT
    ncells::VN
    reach::VN
    cell_offsets::VI
    cellgrid_offsets::VI
    sigma::MT
    epsilon::MT
    cutoff::T
    ewald_cutoff::T
    nsys::Int
end
Adapt.@adapt_structure FrameworkBatch

# Relative tolerance on the guest self term's orientation dependence: `FrameworkBatch` uses the
# orientation average, so a guest/cell combination whose self term swings by more than this
# fraction of a reference thermal energy needs the per-insertion sum instead. The guard checks
# `2·self_term_halfrange` rather than `self_term_halfrange` itself: the half-range is an
# estimate from 64 sampled orientations, and a continuous orientation can reach roughly 1.3
# times that estimate, so the factor of 2 covers that undersampling with margin.
const SELF_TERM_TOLERANCE = 1.0e-3
const SELF_TERM_GUARD_FACTOR = 2
const KT_REF = KB * 300

# Verifies a framework's claimed `replication` against its own host structure factor: a
# translational copy under `replication` implies a k-vector whose integer coefficients are NOT
# all divisible by `replication` has its phase cancel across the copies, leaving a negligible
# structure factor. Checks a deterministic, evenly strided sample of up to 32 such k-vectors
# (`kv_full`/`coeffs`, `kvectors`'s enumeration for the framework's cell) rather than all of
# them, since this runs once per framework at batch-construction time and a sample that
# disagrees is already proof the claim is false; a sample of 32 gives high probability, not
# certainty, of catching a false claim. `Sh_full` is the host structure factor already computed
# for the framework's full k-vector table, reused here rather than recomputed.
function verify_replication(n::Integer, fw::Framework{T}, kv_full, coeffs, Sh_full) where {T}
    uncoupled = [i for i in eachindex(coeffs) if !all(iszero, mod.(coeffs[i], fw.replication))]
    isempty(uncoupled) && return nothing
    stride = max(1, cld(length(uncoupled), 32))
    sample = uncoupled[1:stride:length(uncoupled)]
    length(sample) > 32 && (sample = sample[1:32])
    Ssample = Sh_full[sample]
    maxS, i = findmax(abs, Ssample)
    # `structure_factor` sums natoms terms of order 1; canceling to "zero" leaves roundoff noise
    # that grows with the working type's precision, not a fixed absolute scale — `1e-8` alone is
    # far tighter than Float32's roundoff floor over a few thousand atoms. `100*eps(T)` extends
    # the threshold to cover that floor while leaving Float64 (`100*eps(Float64) < 1e-8`)
    # unchanged.
    threshold = max(T(1.0e-8), 100 * eps(T)) * sum(abs, fw.charges)
    maxS > threshold && throw(
        ArgumentError(
            "framework $n claims replication $(fw.replication), but its host structure factor at " *
                "k=$(kv_full[sample[i]]) (not coupled to that replication) is $maxS, exceeding " *
                "threshold $threshold = max(1e-8, 100·eps($T))·Σ|q_host|; this replication does not describe the atoms"
        )
    )
    return nothing
end

"""
    FrameworkBatch(fws, ff::ForceField, guest::Guest, ewald::EwaldParams; cellwidth = 6) -> FrameworkBatch

Assemble a batch from host frameworks `fws`, sharing one force field, guest and set of Ewald
parameters across all of them. Each framework must already be replicated large enough that its
minimum image exceeds `2 * (max(ff.cutoff, ewald.cutoff) + r_guest)`, `r_guest` the guest's
largest site distance from its reference point, since `insertion_energy`'s cell-list stencil
takes one minimum image per host atom and relies on every contributing pair falling inside that
bound. `cellwidth` (Å) is the target cell-list grid spacing along each axis; each system gets
`max(1, floor(L_i / cellwidth))` cells along its `i`-th perpendicular length `L_i`. The default,
6 Å, is the fastest of `(2, 3, 4, 6)` measured for RUBTAK 3×3×3 + CO2 on an RTX 3050 (see the
efficiency design spec's E2 measurements): at 4 and 6 Å the stencil already spans the whole grid
on every axis for that system's cutoff-plus-guest-reach, so the narrower widths (2, 3 Å) only
add cell-traversal overhead without visiting fewer atoms than 4 or 6 Å already do.

`constant_offset[n]` collects every pose-independent term of inserting `guest` into system `n`:
the tail-correction change, the guest self-energy, its intramolecular exclusion (using the same
erfc_dev convention as `ewald_energy`'s E_excl, since the guest is rigid this is
pose-independent), the net-charge correction from adding the guest's total charge, and the
orientation-averaged reciprocal-space guest self term (see `FrameworkBatch`'s docstring).
Throws if a framework's claimed `replication` disagrees with its own host structure factor (see
`FrameworkBatch`'s docstring), naming the framework index, the claimed replication and the
offending value. Throws if `$(SELF_TERM_GUARD_FACTOR)·self_term_halfrange` exceeds
`$(SELF_TERM_TOLERANCE)·KB·300K` for a guest/framework combination, naming the half-range
estimate and the guard factor: that combination (a strongly polar guest in a small periodic
cell) needs the per-insertion sum, which this batch does not provide. Throws if
`α·ewald.cutoff` exceeds `PAIR_ERFC_XMAX`, naming both values: `insertion_energy`'s
screened-Coulomb pair term (`pair_erfc_dev`) is only fitted up to that bound.
"""
function FrameworkBatch(
        fws::AbstractVector{<:Framework{T}}, ff::ForceField{T}, guest::Guest{T}, ewald::EwaldParams{T};
        cellwidth = 6
    ) where {T}
    rc = max(ff.cutoff, ewald.cutoff)
    r_guest = maximum(norm, guest.sites)
    rc_stencil = rc + r_guest
    w = T(cellwidth)
    positions = SVector{3, T}[]
    types = Int32[]
    charges = T[]
    atom_offsets = Int32[0]
    cells = SMatrix{3, 3, T, 9}[]
    invcells = SMatrix{3, 3, T, 9}[]
    volumes = T[]
    alphas = T[]
    ks = SVector{3, T}[]
    kprefactor = T[]
    Shost = Complex{T}[]
    k_offsets = Int32[0]
    constant_offset = T[]
    self_term_halfrange = T[]
    ncells = SVector{3, Int32}[]
    reach = SVector{3, Int32}[]
    cell_offsets = Int32[]
    cellgrid_offsets = Int32[0]
    gcounts = [count(==(t), guest.types) for t in eachindex(ff.names)]
    α = ewald_alpha(ewald.cutoff, ewald.precision)
    α * ewald.cutoff <= PAIR_ERFC_XMAX || throw(
        ArgumentError(
            "α·ewald_cutoff = $(α * ewald.cutoff) exceeds PAIR_ERFC_XMAX = $PAIR_ERFC_XMAX: " *
                "insertion_energy's screened-Coulomb series is only fitted up to that bound"
        )
    )
    kmax = ewald_kmax(α, ewald.precision)
    neutral_guest = all(iszero, guest.charges)
    # One fixed set of orientations, shared across every framework in the batch: the guest
    # self term depends only on orientation, so two frameworks built from the same cell and
    # guest must get the same self-term samples.
    rng_self = Xoshiro(0x5e1f)
    self_quats = [shoemake_quaternion(rng_self, T) for _ in 1:64]
    for (n, fw) in pairs(fws)
        m = min_multiplicity(fw.cell, rc_stencil)
        m == (1, 1, 1) || throw(
            ArgumentError(
                "framework $n is too small for cutoff $rc plus guest reach $r_guest = $rc_stencil; " *
                    "replicate it by $m first"
            )
        )
        A = fw.cell
        L = perpendicular_lengths(A)
        n_grid = grid_dims(L, w)
        reach_n = stencil_reaches(L, n_grid, rc_stencil)
        pos = cartesian(fw)
        # kUPS UFF-style LJ type names carry a trailing underscore (e.g. "Zr_"); CIF element
        # symbols don't, so the lookup appends it.
        ty = Int32[typeindex(ff, s * "_") for s in fw.symbols]
        perm, local_offsets = cell_sort(fw.frac, n_grid)
        append!(positions, pos[perm])
        append!(types, ty[perm])
        append!(charges, fw.charges[perm])
        push!(atom_offsets, Int32(length(positions)))
        append!(cell_offsets, local_offsets)
        push!(cellgrid_offsets, Int32(length(cell_offsets)))
        push!(ncells, n_grid)
        push!(reach, reach_n)
        V = volume(A)
        push!(cells, A)
        push!(invcells, inv(A))
        push!(volumes, V)
        push!(alphas, α)
        # Verifying a claimed `replication` needs the full k-vector table even for a neutral
        # guest, which otherwise builds no reciprocal-space table at all.
        if !neutral_guest || fw.replication != (1, 1, 1)
            kv_full, kpref_full, Sh_full, coeffs = full_ktables(A, pos, fw.charges, α, kmax)
            fw.replication == (1, 1, 1) || verify_replication(n, fw, kv_full, coeffs, Sh_full)
        end
        self_mean = zero(T)
        # a guest without charges has no Coulomb terms, so no reciprocal-space table is built
        # and its self term is exactly zero at every orientation
        if !neutral_guest
            coupled = [all(iszero, mod.(c, fw.replication)) for c in coeffs]
            append!(ks, kv_full[coupled])
            append!(kprefactor, kpref_full[coupled])
            append!(Shost, Sh_full[coupled])
            samples = Vector{T}(undef, 64)
            for o in eachindex(samples, self_quats)
                q = self_quats[o]
                gsites = map(s -> rotate(q, s), guest.sites)
                acc = zero(T)
                for i in eachindex(kv_full, kpref_full)
                    k = kv_full[i]
                    Sg = zero(Complex{T})
                    for s in eachindex(gsites, guest.charges)
                        Sg += guest.charges[s] * cis(dot(k, gsites[s]))
                    end
                    acc += kpref_full[i] * abs2(Sg)
                end
                samples[o] = KE * acc
            end
            self_mean = sum(samples) / length(samples)
            halfrange = (maximum(samples) - minimum(samples)) / 2
            push!(self_term_halfrange, halfrange)
            SELF_TERM_GUARD_FACTOR * halfrange / KT_REF > SELF_TERM_TOLERANCE && throw(
                ArgumentError(
                    "guest self-term half-range estimate $halfrange eV (guard factor $SELF_TERM_GUARD_FACTOR) for " *
                        "guest with charges $(guest.charges) at framework $n: " *
                        "$SELF_TERM_GUARD_FACTOR·$halfrange exceeds $(SELF_TERM_TOLERANCE)·KB·300K = " *
                        "$(SELF_TERM_TOLERANCE * KT_REF) eV; this guest/cell combination needs the per-insertion " *
                        "reciprocal sum"
                )
            )
        else
            push!(self_term_halfrange, zero(T))
        end
        push!(k_offsets, Int32(length(ks)))
        counts = [count(==(t), ty) for t in eachindex(ff.names)]
        self = -α / sqrt(T(π)) * sum(abs2, guest.charges)
        excl = zero(T)
        for a in eachindex(guest.sites), c in eachindex(guest.sites)
            c > a || continue
            r = norm(guest.sites[a] - guest.sites[c])
            excl -= guest.charges[a] * guest.charges[c] * (one(T) - erfc_dev(α * r)) / r
        end
        Qh = sum(fw.charges)
        Qg = sum(guest.charges)
        net = -T(π) / (2 * V * α^2) * ((Qh + Qg)^2 - Qh^2)
        push!(constant_offset, tail_delta(ff, counts, gcounts, V) + KE * (self + excl + net) + self_mean)
    end
    return FrameworkBatch(
        positions, types, charges, atom_offsets, cells, invcells, volumes, alphas,
        ks, kprefactor, Shost, k_offsets, constant_offset, self_term_halfrange,
        ncells, reach, cell_offsets, cellgrid_offsets,
        ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff, length(fws)
    )
end
