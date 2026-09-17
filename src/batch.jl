"""
    FrameworkBatch{T}

Structure-of-arrays layout for a set of independent framework systems, so a single GPU kernel
launch can evaluate an insertion in every system at once. Per-system slices are
`atom_offsets[n]+1:atom_offsets[n+1]` into positions/types/charges and
`k_offsets[n]+1:k_offsets[n+1]` into ks/kprefactor/Shost. `cutoff` (Å) truncates the LJ sum and
`ewald_cutoff` (Å) truncates the real- and reciprocal-space Ewald sums; they may differ.
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
    sigma::MT
    epsilon::MT
    cutoff::T
    ewald_cutoff::T
    nsys::Int
end
Adapt.@adapt_structure FrameworkBatch

"""
    FrameworkBatch(fws, ff::ForceField, guest::Guest, ewald::EwaldParams) -> FrameworkBatch

Assemble a batch from host frameworks `fws`, sharing one force field, guest and set of Ewald
parameters across all of them. Each framework must already be replicated large enough that its
minimum image exceeds `2 * max(ff.cutoff, ewald.cutoff)`.

`constant_offset[n]` collects every pose-independent term of inserting `guest` into system `n`:
the tail-correction change, the guest self-energy, its intramolecular exclusion (using the same
erfc_dev convention as `ewald_energy`'s E_excl, since the guest is rigid this is
pose-independent), and the net-charge correction from adding the guest's total charge.
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
    gcounts = [count(==(t), guest.types) for t in eachindex(ff.names)]
    α = ewald_alpha(ewald.cutoff, ewald.precision)
    kmax = ewald_kmax(α, ewald.precision)
    neutral_guest = all(iszero, guest.charges)
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
        # a guest without charges has no Coulomb terms, so no reciprocal-space table is built
        if !neutral_guest
            kv, w = kvectors(A, kmax)
            append!(ks, kv)
            append!(kprefactor, w[i] * pk(dot(kv[i], kv[i]), α, V) for i in eachindex(kv, w))
            append!(Shost, structure_factor(kv, pos, fw.charges))
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
        push!(constant_offset, tail_delta(ff, counts, gcounts, V) + KE * (self + excl + net))
    end
    return FrameworkBatch(
        positions, types, charges, atom_offsets, cells, invcells, volumes, alphas,
        ks, kprefactor, Shost, k_offsets, constant_offset, ff.sigma, ff.epsilon, ff.cutoff, ewald.cutoff, length(fws)
    )
end
