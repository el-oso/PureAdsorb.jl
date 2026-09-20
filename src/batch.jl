"""
    FrameworkBatch{T}

Structure-of-arrays layout for a set of independent framework systems, so a single GPU kernel
launch can evaluate an insertion in every system at once. Per-system slices are
`atom_offsets[n]+1:atom_offsets[n+1]` into positions/types/charges and
`k_offsets[n]+1:k_offsets[n+1]` into ks/kprefactor/Shost. `cutoff` (Å) truncates the LJ sum and
`ewald_cutoff` (Å) truncates the real- and reciprocal-space Ewald sums; they may differ.

`ks`/`kprefactor`/`Shost` hold only the k-vectors coupled to each system's replication: those
whose integer reciprocal-lattice coefficients are all divisible by the corresponding
replication factor, since only those repeat identically across the copies making up the
supercell and so carry a nonzero host structure factor. `self_term_halfrange[n]` (energy
units) is half the max-min spread, over 64 fixed guest orientations, of the guest self term
`KE Σ_k pref_k |S_g(k)|²` taken over the FULL (unfiltered) k set of system `n`; its mean over
those orientations is folded into `constant_offset[n]` in place of recomputing the self term
per insertion.
"""
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
    kprefactor::VT
    Shost::VS
    k_offsets::VI
    constant_offset::VT
    self_term_halfrange::VT
    sigma::MT
    epsilon::MT
    cutoff::T
    ewald_cutoff::T
    nsys::Int
end
Adapt.@adapt_structure FrameworkBatch

# Relative tolerance on the guest self term's orientation dependence: `FrameworkBatch` uses the
# orientation average, so a guest/cell combination whose self term swings by more than this
# fraction of a reference thermal energy needs the per-insertion sum instead.
const SELF_TERM_TOLERANCE = 1.0e-3
const KT_REF = KB * 300

"""
    FrameworkBatch(fws, ff::ForceField, guest::Guest, ewald::EwaldParams) -> FrameworkBatch

Assemble a batch from host frameworks `fws`, sharing one force field, guest and set of Ewald
parameters across all of them. Each framework must already be replicated large enough that its
minimum image exceeds `2 * max(ff.cutoff, ewald.cutoff)`.

`constant_offset[n]` collects every pose-independent term of inserting `guest` into system `n`:
the tail-correction change, the guest self-energy, its intramolecular exclusion (using the same
erfc_dev convention as `ewald_energy`'s E_excl, since the guest is rigid this is
pose-independent), the net-charge correction from adding the guest's total charge, and the
orientation-averaged reciprocal-space guest self term (see `FrameworkBatch`'s docstring).
Throws if a guest/framework combination's self-term half-range exceeds
`$(SELF_TERM_TOLERANCE)·KB·300K`: that combination (a strongly polar guest in a small periodic
cell) needs the per-insertion sum, which this batch does not provide.
"""
function FrameworkBatch(fws::AbstractVector{<:Framework{T}}, ff::ForceField{T}, guest::Guest{T}, ewald::EwaldParams{T}) where {T}
    rc = max(ff.cutoff, ewald.cutoff)
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
    gcounts = [count(==(t), guest.types) for t in eachindex(ff.names)]
    α = ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = ewald_kmax(α, ewald.precision)
    neutral_guest = all(iszero, guest.charges)
    # One fixed set of orientations, shared across every framework in the batch: the guest
    # self term depends only on orientation, so two frameworks built from the same cell and
    # guest must get the same self-term samples.
    rng_self = Xoshiro(0x5e1f)
    self_quats = [shoemake_quaternion(rng_self, T) for _ in 1:64]
    for (n, fw) in pairs(fws)
        m = min_multiplicity(fw.cell, rc)
        m == (1, 1, 1) || throw(ArgumentError("framework $n is too small for cutoff $rc; replicate it by $m first"))
        pos = cartesian(fw)
        # kUPS UFF-style LJ type names carry a trailing underscore (e.g. "Zr_"); CIF element
        # symbols don't, so the lookup appends it.
        ty = Int32[typeindex(ff, s * "_") for s in fw.symbols]
        append!(positions, pos)
        append!(types, ty)
        append!(charges, fw.charges)
        push!(atom_offsets, Int32(length(positions)))
        A = fw.cell
        V = volume(A)
        push!(cells, A)
        push!(invcells, inv(A))
        push!(volumes, V)
        push!(alphas, α)
        self_mean = zero(T)
        # a guest without charges has no Coulomb terms, so no reciprocal-space table is built
        # and its self term is exactly zero at every orientation
        if !neutral_guest
            kv_full, kpref_full, Sh_full, coeffs = full_ktables(A, pos, fw.charges, α, kmax)
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
            halfrange / KT_REF > SELF_TERM_TOLERANCE && throw(
                ArgumentError(
                    "guest self-term half-range $halfrange eV for guest with charges $(guest.charges) at " *
                        "framework $n exceeds $(SELF_TERM_TOLERANCE)·KB·300K = $(SELF_TERM_TOLERANCE * KT_REF) eV; " *
                        "this guest/cell combination needs the per-insertion reciprocal sum"
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
        ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff, length(fws)
    )
end
