# Milestone B's guest-guest and guest-host energetics, built on `FrameworkBatch`'s full k-vector
# table and `SystemState`'s running structure factor. Every reciprocal-space quantity here is
# read off `Sk = Shost + Σ_guests S_guest` rather than recomputed per pair, so the host-host self
# term (`|Shost|²`, constant for a rigid framework) is the only piece of the full Ewald sum that
# never appears below — it cancels in every energy difference and plays no role in an absolute
# energy either, matching the design's own decomposition (`U_host-host` "never needs
# evaluating").

# Guest-guest tail correction, generalized from `tail_delta`'s single-guest form to `N` guests:
# passing `N .* gcounts` as the added particle counts gives, in the SAME ordered-pair sum,
# both the host-guest cross term (linear in `N`, since it appears as `counts[i] * N*gcounts[j]`)
# and the guest-guest self term at `N²` (`(N*gcounts[i]) * (N*gcounts[j])`) — the scaling ruled
# in favor of kUPS's own convention (R2), not the exact pair count `N(N-1)/2`.
guest_tail_correction(ff::ForceField{T}, host_counts, gcounts, N::Integer, V) where {T} = tail_delta(ff, host_counts, N .* gcounts, V)

# Per-guest, pose-independent Ewald terms (Ewald units, scaled by `KE` at the call site, matching
# `FrameworkBatch`'s own `self`/`excl` convention): the Gaussian self-energy correction and the
# intramolecular exclusion. `ewald_energy`'s reciprocal sum has no notion of molecules, so it
# implicitly supplies `erf(αr)/r` for every pair including a guest's own bonded sites; keeping
# those pairs out of the real-space sum (as this package does, unlike kUPS's opposite
# convention — see `ewald_energy`'s docstring and ruling R1) means the exclusion below must
# subtract exactly that leftover `erf(αr)/r = (1 − erfc(αr))/r` term, not the bare Coulomb
# `1/r` kUPS's own convention subtracts. The two conventions give the same total only when every
# intramolecular distance lies inside the real-space cutoff, so this asserts it (R1) rather than
# assuming it.
function guest_self_terms(guest::Guest{T, N}, alpha::T, cutoff::T) where {T, N}
    self = -alpha / sqrt(T(π)) * sum(abs2, guest.charges)
    excl = zero(T)
    for a in 1:N, c in 1:N
        c > a || continue
        r = norm(guest.sites[a] - guest.sites[c])
        r < cutoff || throw(
            ArgumentError(
                "guest_self_terms: intramolecular distance $r (sites $a, $c) is not less than the real-space " *
                    "cutoff $cutoff; the exclusion scheme's equivalence to kUPS's own (R1) requires every " *
                    "intramolecular site pair to lie inside the cutoff"
            )
        )
        excl -= guest.charges[a] * guest.charges[c] * (one(T) - erfc_dev(alpha * r)) / r
    end
    return self, excl
end

# Lennard-Jones and real-space (screened-Coulomb) energy between two guests' already-rotated,
# absolute site positions `sites_a`/`sites_b`, both typed by the shared compact index `gtypes`
# and charged by `gcharges` (both are the same rigid guest species). Each site pair takes its
# own minimum image — unlike `insertion_energy`'s single-image-per-host-atom shortcut, which
# only holds because a host atom is a point particle — since with `N ≤ 200` guests of a handful
# of sites the pair count stays small (see `guest_guest_energy`'s docstring) and a per-pair image
# adds no growth in cost class.
function guest_pair_realspace_energy(
        sites_a::SVector{N, SVector{3, T}}, sites_b::SVector{N, SVector{3, T}}, gtypes::SVector{N, Int}, gcharges::SVector{N, T},
        sigma, epsilon, cutoff::T, ewald_cutoff::T, A, invA, alpha::T
    ) where {N, T}
    rc_lj2 = cutoff * cutoff
    rc_ew2 = ewald_cutoff * ewald_cutoff
    E_lj = zero(T); E_sr = zero(T)
    for a in 1:N, b in 1:N
        Δ = minimum_image(A, invA, sites_a[a] - sites_b[b])
        r2 = dot(Δ, Δ)
        (r2 < rc_lj2 || r2 < rc_ew2) || continue
        ta = gtypes[a]; tb = gtypes[b]
        if r2 < rc_lj2
            σ = sigma[ta, tb]; ε = epsilon[ta, tb]
            E_lj += lj_pair_energy(r2, σ, ε)
        end
        if r2 < rc_ew2
            r = sqrt(r2)
            E_sr += gcharges[a] * gcharges[b] * pair_erfc_dev(alpha * r) / r
        end
    end
    return E_lj, E_sr
end

# Every guest site's absolute (rotated, translated) position, as an `SVector{N}` rather than a
# `Vector`, so it stays isbits and usable inside a kernel.
guest_sites_at(guest::Guest{T, N}, pos::SVector{3, T}, q::SVector{4, T}) where {T, N} = SVector{N}(ntuple(s -> pos + rotate(q, guest.sites[s]), N))

"""
    guest_guest_energy(guest, refpoints, orientations, gr, sigma, epsilon, guest_types,
                        cutoff, ewald_cutoff, A, invA, alpha) -> (E_lj, E_sr)

Guest-guest Lennard-Jones (Lorentz-Berthelot, `cutoff`) and real-space Ewald (`pair_erfc_dev`,
`ewald_cutoff`) energy over every guest pair in `gr` (a `guest_range`), an all-pairs sum: with
`N ≤ 200` guests (kUPS's `max_num_adsorbates`) and a handful of sites each, this is the cheapest
correct structure at this scale — a cell list over guests would cost more than it saves. `E_sr`
is unscaled by `KE`, matching `guest_self_terms`.
"""
function guest_guest_energy(
        guest::Guest{T, N}, refpoints, orientations, gr::UnitRange, sigma, epsilon, guest_types::SVector{N, Int},
        cutoff::T, ewald_cutoff::T, A, invA, alpha::T
    ) where {T, N}
    E_lj = zero(T); E_sr = zero(T)
    for i in gr, j in gr
        j > i || continue
        sites_i = guest_sites_at(guest, refpoints[i], orientations[i])
        sites_j = guest_sites_at(guest, refpoints[j], orientations[j])
        lj, sr = guest_pair_realspace_energy(sites_i, sites_j, guest_types, guest.charges, sigma, epsilon, cutoff, ewald_cutoff, A, invA, alpha)
        E_lj += lj; E_sr += sr
    end
    return E_lj, E_sr
end

# Guest-guest real-space delta from moving guest `i` (global index into `refpoints`/
# `orientations`) of system `n` from `(oldpos, oldq)` to `(newpos, newq)`: every OTHER guest whose
# index appears in `gr` pairs once against the old pose and once against the new one. `gr` is
# normally system `n`'s full `guest_range`, but any range or strided range over it is exact too —
# summing this over a partition of `guest_range(state, n)` into disjoint strided ranges (as
# `moves.jl`'s workgroup fan-out does) gives the same total, since the pair sum is just a plain
# sum over `gr`. The constant terms (self energy, exclusion, tail correction, net charge) do not
# depend on any guest's pose and so never appear in a `ΔU`.
function guest_guest_move_delta(
        guest::Guest{T, N}, refpoints, orientations, gr, i::Integer,
        oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        sigma, epsilon, guest_types::SVector{N, Int}, cutoff::T, ewald_cutoff::T, A, invA, alpha::T
    ) where {T, N}
    old_sites = guest_sites_at(guest, oldpos, oldq)
    new_sites = guest_sites_at(guest, newpos, newq)
    E_lj = zero(T); E_sr = zero(T)
    for j in gr
        j == i && continue
        other_sites = guest_sites_at(guest, refpoints[j], orientations[j])
        lj_o, sr_o = guest_pair_realspace_energy(old_sites, other_sites, guest_types, guest.charges, sigma, epsilon, cutoff, ewald_cutoff, A, invA, alpha)
        lj_n, sr_n = guest_pair_realspace_energy(new_sites, other_sites, guest_types, guest.charges, sigma, epsilon, cutoff, ewald_cutoff, A, invA, alpha)
        E_lj += lj_n - lj_o
        E_sr += sr_n - sr_o
    end
    return E_lj + T(KE) * E_sr
end

"""
    guest_pair_realspace_energy_range(guest, testpos, testq, refpoints, orientations, gr, exclude,
                                       sigma, epsilon, guest_types, cutoff, ewald_cutoff, A, invA, alpha) -> (E_lj, E_sr)

Real-space (LJ + screened-Coulomb) energy between one guest at pose `(testpos, testq)` and every
OTHER live guest whose global index appears in `gr`, skipping index `exclude`. Any range or
strided range over `gr` is exact, since the sum is just a plain sum over `gr` — summing this over
a partition of a system's live guests into disjoint strided ranges gives the same total, exactly
as `guest_guest_move_delta` already relies on for the NVT moves (`moves.jl`'s μVT exchange kernels
use this the same way). An insertion attempt, whose candidate pose is not itself stored in
`refpoints`/`orientations`, passes `exclude = 0`, which never appears in a 1-based `gr`; a
deletion attempt passes its own guest's global index so it is not paired against itself.
"""
function guest_pair_realspace_energy_range(
        guest::Guest{T, N}, testpos::SVector{3, T}, testq::SVector{4, T}, refpoints, orientations, gr, exclude::Integer,
        sigma, epsilon, guest_types::SVector{N, Int}, cutoff::T, ewald_cutoff::T, A, invA, alpha::T
    ) where {T, N}
    test_sites = guest_sites_at(guest, testpos, testq)
    E_lj = zero(T); E_sr = zero(T)
    for j in gr
        j == exclude && continue
        other_sites = guest_sites_at(guest, refpoints[j], orientations[j])
        lj, sr = guest_pair_realspace_energy(test_sites, other_sites, guest_types, guest.charges, sigma, epsilon, cutoff, ewald_cutoff, A, invA, alpha)
        E_lj += lj; E_sr += sr
    end
    return E_lj, E_sr
end

"""
    host_guest_realspace_energy_range(pos, q, guest, sigma, epsilon, cutoff, ewald_cutoff,
                                       positions, types, charges, atom_range, A, invA, alpha) -> energy

`host_guest_realspace_energy`'s loop body, over the caller-supplied `atom_range` (host atom
indices into `positions`/`types`/`charges`) rather than a whole system's contiguous block: any
range is exact, since the sum is just a plain sum over `atom_range`, so summing this over a
partition of a system's atom block into disjoint strided ranges (`moves.jl`'s workgroup fan-out)
gives the same total as one call over the whole block. All arguments are isbits scalars,
SVectors, or plain/view array reads with no throwing branches, so this runs unchanged inside a
GPU kernel.
"""
function host_guest_realspace_energy_range(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        positions, types, charges, atom_range, A, invA, alpha
    ) where {T, N}
    rc_lj2 = cutoff * cutoff
    rc_ew2 = ewald_cutoff * ewald_cutoff
    gsites = map(s -> rotate(q, s), guest.sites)
    E_lj = zero(T); E_sr = zero(T)
    for j in atom_range
        Δ0 = minimum_image(A, invA, pos - positions[j])
        ht = types[j]; hqj = charges[j]
        for s in 1:N
            Δ = Δ0 + gsites[s]
            r2 = dot(Δ, Δ)
            (r2 < rc_lj2 || r2 < rc_ew2) || continue
            gt = guest.types[s]; gq = guest.charges[s]
            if r2 < rc_lj2
                σ = sigma[gt, ht]; ε = epsilon[gt, ht]
                E_lj += lj_pair_energy(r2, σ, ε)
            end
            if r2 < rc_ew2
                r = sqrt(r2)
                E_sr += gq * hqj * pair_erfc_dev(alpha * r) / r
            end
        end
    end
    return E_lj + T(KE) * E_sr
end

"""
    host_guest_realspace_energy(pos, q, guest, sigma, epsilon, cutoff, ewald_cutoff,
                                 positions, types, charges, atom_base, natoms, A, invA, alpha) -> energy

Lennard-Jones plus real-space (screened-Coulomb) energy between one guest pose and the host
atoms `positions[(atom_base+1):(atom_base+natoms)]`: `insertion_energy`'s real-space loop with
the reciprocal-space cross term omitted, since Milestone B's reciprocal energy is a single sum
over the running total structure factor instead (`total_reciprocal_energy`,
`reciprocal_move_delta!`). Isolating this from the reciprocal term is what lets the per-move
throughput measurement (`bench/guest_bench.jl`) time real-space and reciprocal work separately.
`host_guest_realspace_energy_range` over `(atom_base+1):(atom_base+natoms)`.
"""
function host_guest_realspace_energy(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        positions, types, charges, atom_base::Integer, natoms::Integer, A, invA, alpha
    ) where {T, N}
    return host_guest_realspace_energy_range(
        pos, q, guest, sigma, epsilon, cutoff, ewald_cutoff, positions, types, charges,
        (atom_base + 1):(atom_base + natoms), A, invA, alpha
    )
end

"""
    total_reciprocal_energy(kprefactor, Sk, Shost) -> energy

Reciprocal-space energy of everything except the host's own self-interaction:
`KE Σ_k pref_k (|Sk(k)|² − |Shost(k)|²)`. `Sk` is the running total structure factor (host plus
every guest, `SystemState.Sk`) and `Shost` the host's own (`FrameworkBatch.Shost`), both sliced
to one system's k-vectors (`kvec_range`); the host-host term `|Shost|²` cancels identically,
leaving the host-guest cross term (every guest) and the guest self and cross terms (every guest
pair) in one sum, with no per-guest loop needed.
"""
function total_reciprocal_energy(kprefactor, Sk, Shost)
    T = eltype(kprefactor)
    acc = zero(T)
    for i in eachindex(kprefactor, Sk, Shost)
        acc += kprefactor[i] * (abs2(Sk[i]) - abs2(Shost[i]))
    end
    return T(KE) * acc
end

"""
    cross_reciprocal_energy(kprefactor, Sk, Shost) -> energy

Host-guest cross reciprocal-space energy alone, `KE Σ_k pref_k · 2 Re(conj(Shost(k))·Sguest(k))`,
`Sguest = Sk - Shost` recovered from the running total `Sk` (a system's own slice) and the host's
own structure factor `Shost`, both sliced to the same k-vectors. Unlike `total_reciprocal_energy`,
this omits the guest self term `|Sguest|²` entirely, which is what makes it exact on a table
restricted to the k-vectors coupled to the host's replication — the cross term vanishes wherever
`Shost` does (theory.md's "Which k-vectors each term needs"), but `|Sguest|²` generally does not,
so `total_reciprocal_energy`'s combined formula would be wrong there. `FrameworkBatch`'s guest-guest
table (`ks_gg`/`kprefactor_gg`, `SystemState.Sk_gg`) supplies the guest self term separately, via
`self_reciprocal_energy`, over the k-vectors the cross term does not need.
"""
function cross_reciprocal_energy(kprefactor, Sk, Shost)
    T = eltype(kprefactor)
    acc = zero(T)
    for i in eachindex(kprefactor, Sk, Shost)
        acc += kprefactor[i] * 2 * real(conj(Shost[i]) * (Sk[i] - Shost[i]))
    end
    return T(KE) * acc
end

"""
    self_reciprocal_energy(kprefactor, Sk) -> energy

Guest-only reciprocal-space energy `KE Σ_k pref_k |Sk(k)|²`, `Sk` here a purely guest structure
factor (no host term to subtract, unlike `total_reciprocal_energy`) on a table that carries no
host contribution at all — `FrameworkBatch`'s guest-guest table (`SystemState.Sk_gg` on
`batch.ks_gg`/`batch.kprefactor_gg`).
"""
function self_reciprocal_energy(kprefactor, Sk)
    T = eltype(kprefactor)
    acc = zero(T)
    for i in eachindex(kprefactor, Sk)
        acc += kprefactor[i] * abs2(Sk[i])
    end
    return T(KE) * acc
end

# One k-vector's structure-factor change and its (unweighted, unscaled by `KE`) contribution to
# `ΔU_recip`, given the guest's already-rotated old/new site positions (`guest_sites_at`):
# shared by `reciprocal_move_delta!` and `reciprocal_move_delta_energy`, which differ only in
# whether the per-k `ΔS` is kept.
@inline function _reciprocal_move_delta_k(
        k::SVector{3, T}, charges::SVector{N, T}, old_sites::SVector{N, SVector{3, T}}, new_sites::SVector{N, SVector{3, T}}, Sk_i::Complex{T}
    ) where {N, T}
    Sold = zero(Complex{T}); Snew = zero(Complex{T})
    for s in 1:N
        Sold += charges[s] * cis(dot(k, old_sites[s]))
        Snew += charges[s] * cis(dot(k, new_sites[s]))
    end
    ds = Snew - Sold
    return ds, 2 * real(conj(Sk_i) * ds) + abs2(ds)
end

# The cross-table counterpart of `_reciprocal_move_delta_k`: `Shost_i` is constant (the host does
# not move), so `Δ[2 Re(conj(Shost)·Sguest)] = 2 Re(conj(Shost)·ds)` exactly, with no `|ds|²` term
# — unlike `_reciprocal_move_delta_k`'s target `|Sk|²`, which is quadratic in the moving quantity
# itself, `2 Re(conj(Shost)·Sguest)` is only linear in it.
@inline function _cross_move_delta_k(
        k::SVector{3, T}, charges::SVector{N, T}, old_sites::SVector{N, SVector{3, T}}, new_sites::SVector{N, SVector{3, T}}, Shost_i::Complex{T}
    ) where {N, T}
    Sold = zero(Complex{T}); Snew = zero(Complex{T})
    for s in 1:N
        Sold += charges[s] * cis(dot(k, old_sites[s]))
        Snew += charges[s] * cis(dot(k, new_sites[s]))
    end
    ds = Snew - Sold
    return ds, 2 * real(conj(Shost_i) * ds)
end

"""
    reciprocal_cross_delta!(ΔS, guest, oldpos, oldq, newpos, newq, ks, kprefactor, Shost) -> ΔU_cross

`reciprocal_move_delta!`'s cross-only counterpart, for the sparse (host-coupled) table under
`has_ewald_split`: writes `ΔS[i] = S_i^new(k_i) − S_i^old(k_i)` exactly as `reciprocal_move_delta!`
does (`ΔS` is guest-only either way, so `Sk`'s own update on acceptance, `Sk .+= ΔS`, is unchanged),
and returns `ΔU_cross = KE Σ_k pref_k · 2 Re[conj(Shost) ΔS]` — `cross_reciprocal_energy`'s own
formula, differentiated with respect to one guest's move.
"""
function reciprocal_cross_delta!(
        ΔS, guest::Guest{T, N}, oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        ks, kprefactor, Shost
    ) where {T, N}
    old_sites = guest_sites_at(guest, oldpos, oldq)
    new_sites = guest_sites_at(guest, newpos, newq)
    ΔU = zero(T)
    for i in eachindex(ks)
        ds, contribution = _cross_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Shost[i])
        ΔS[i] = ds
        ΔU += kprefactor[i] * contribution
    end
    return T(KE) * ΔU
end

"""
    reciprocal_move_delta!(ΔS, guest, oldpos, oldq, newpos, newq, ks, kprefactor, Sk) -> ΔU_recip

Reciprocal-space energy change from moving one guest from `(oldpos, oldq)` to `(newpos, newq)`,
writing its structure-factor change `ΔS[i] = S_i^new(k_i) − S_i^old(k_i)` for every k-vector in
`ks`/`kprefactor`/`Sk` (a system's own slice, `kvec_range`) and returning
`ΔU_recip = KE Σ_k pref_k (2 Re[conj(Sk) ΔS] + |ΔS|²)`, `Sk` the running total *before* the move
(moving one guest changes only its own term in `Sk = Shost + Σ_guests S_guest`, so this is exact
regardless of how many other guests or host atoms contribute to `Sk`). `ΔS` is index-matched to
`ks`; applying it (`Sk .+= ΔS`) on acceptance, or discarding it on rejection, is the caller's
responsibility. All arguments are isbits scalars or plain/view array reads with no throwing
branches, and `ΔS` may be a device array view, so this runs unchanged inside a GPU kernel.
"""
function reciprocal_move_delta!(
        ΔS, guest::Guest{T, N}, oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        ks, kprefactor, Sk
    ) where {T, N}
    old_sites = guest_sites_at(guest, oldpos, oldq)
    new_sites = guest_sites_at(guest, newpos, newq)
    ΔU = zero(T)
    for i in eachindex(ks)
        ds, contribution = _reciprocal_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
        ΔS[i] = ds
        ΔU += kprefactor[i] * contribution
    end
    return T(KE) * ΔU
end

# Same formula as `reciprocal_move_delta!`, without writing `ΔS`: for the throughput measurement
# and any caller that only needs the energy change (a rejected move never applies `ΔS` anyway).
function reciprocal_move_delta_energy(
        guest::Guest{T, N}, oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        ks, kprefactor, Sk
    ) where {T, N}
    old_sites = guest_sites_at(guest, oldpos, oldq)
    new_sites = guest_sites_at(guest, newpos, newq)
    ΔU = zero(T)
    for i in eachindex(ks)
        _, contribution = _reciprocal_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
        ΔU += kprefactor[i] * contribution
    end
    return T(KE) * ΔU
end

"""
    total_energy(batch::FrameworkBatch, state::SystemState, guest::Guest, ff::ForceField, n::Integer) -> energy

Total configuration energy of system `n`: every host-guest pair (Lennard-Jones plus real- and
reciprocal-space Ewald), every guest-guest pair (likewise), the guest tail correction (`N²`
convention, R2), each guest's constant self/exclusion terms (R1) and the net-charge correction.
`U_host-host` is never computed, matching the design's own decomposition — it is a constant that
cancels in every difference and plays no role in this absolute value either. Recomputing this
from scratch and comparing it against the running total accumulated from accepted `ΔU`s is the
energy audit (task 8); it is also what seeds `SystemState.energy` at construction.
"""
function total_energy(batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, ff::ForceField{T}, n::Integer) where {T, N}
    gr = guest_range(state, n)
    kr = kvec_range(state, n)
    krb = batch_kvec_range(batch, n)
    Ng = length(gr)
    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    guest_types = SVector{N, Int}(batch.guest_types)
    # `batch.sigma`/`batch.epsilon` are indexed by the batch's compact LJ type, so
    # `host_guest_realspace_energy` (mirroring `insertion_energy`'s own convention) needs a
    # guest whose `.types` field already carries that compact index, not `guest`'s original one.
    guest_compact = Guest{T, N}(guest.sites, guest_types, guest.charges, guest.tc, guest.pc, guest.omega)

    E = zero(T)
    for i in gr
        E += host_guest_realspace_energy(
            state.refpoints[i], state.orientations[i], guest_compact, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
            batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
        )
    end

    # With `has_ewald_split(batch)`, every guest-guest quantity (real-space screened Coulomb,
    # self energy, intramolecular exclusion, and the guest-charge piece of the net-charge
    # correction) attaches to the guest-guest splitting parameter `alpha_gg`/`ewald_cutoff_gg`
    # rather than the host's own — `alpha_gg`/`ewald_cutoff_gg` equal `alpha`/`batch.ewald_cutoff`
    # when the batch has no split, so this is a no-op change there.
    split = has_ewald_split(batch)
    alpha_gg = split ? batch.alphas_gg[fw] : alpha
    ewald_cutoff_gg = batch.ewald_cutoff_gg

    E_lj_gg, E_sr_gg = guest_guest_energy(
        guest, state.refpoints, state.orientations, gr, batch.sigma, batch.epsilon, guest_types, batch.cutoff, ewald_cutoff_gg, A, invA, alpha_gg
    )
    E += E_lj_gg + T(KE) * E_sr_gg

    if split
        E += cross_reciprocal_energy(view(batch.kprefactor, krb), view(state.Sk, kr), view(batch.Shost, krb))
        krb_gg = batch_kvec_gg_range(batch, n)
        kr_gg = kvec_gg_range(state, n)
        E += self_reciprocal_energy(view(batch.kprefactor_gg, krb_gg), view(state.Sk_gg, kr_gg))
    else
        E += total_reciprocal_energy(view(batch.kprefactor, krb), view(state.Sk, kr), view(batch.Shost, krb))
    end

    gself, gexcl = guest_self_terms(guest, alpha_gg, ewald_cutoff_gg)
    E += Ng * T(KE) * (gself + gexcl)

    ntypes_ff = length(ff.names)
    gcounts = zeros(Int, ntypes_ff)
    for t in batch.guest_types_orig
        gcounts[t] += 1
    end
    host_counts = zeros(Int, ntypes_ff)
    for j in (a0 + 1):(a0 + natoms)
        host_counts[batch.compact_to_orig[batch.types[j]]] += 1
    end
    E += guest_tail_correction(ff, host_counts, gcounts, Ng, batch.volumes[fw])

    Qh = sum(view(batch.charges, (a0 + 1):(a0 + natoms)))
    Qg = sum(guest.charges)
    if split
        # The net-charge (k=0) correction for adding `Ng` guests of total charge `Ng*Qg` splits
        # exactly as `(Qh + Ng*Qg)² - Qh² = 2·Qh·(Ng·Qg) + (Ng·Qg)²` does: the cross piece
        # (bilinear in the host's and the guests' charge) attaches to `alpha`, matching
        # `cross_reciprocal_energy`'s own splitting parameter; the guest-self piece (quadratic in
        # the guests' charge alone) attaches to `alpha_gg`, matching `self_reciprocal_energy`'s.
        E += -T(KE) * T(π) / (batch.volumes[fw] * alpha^2) * Qh * (Ng * Qg)
        E += -T(KE) * T(π) / (2 * batch.volumes[fw] * alpha_gg^2) * (Ng * Qg)^2
    else
        E += -T(KE) * T(π) / (2 * batch.volumes[fw] * alpha^2) * ((Qh + Ng * Qg)^2 - Qh^2)
    end

    return E
end

# One entry per system, matching `SystemState.energy`'s own layout.
total_energy(batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T}, ff::ForceField{T}) where {T} =
    [total_energy(batch, state, guest, ff, n) for n in 1:batch.nsys]

"""
    guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS) -> (ΔU, host_energy_new)

Total energy change from moving guest `i` (a global index into `state.refpoints`/
`state.orientations`) of system `n` to pose `(newpos, newq)`, writing the k-indexed
structure-factor change into `ΔS` (`reciprocal_move_delta!`; apply it to `state.Sk` on
acceptance) and returning guest `i`'s freshly computed host-guest real-space energy at the new
pose. `ΔU` is equal, to accumulated rounding, to the difference of two `total_energy` calls
before and after applying the move — the per-guest constant terms (tail correction, self energy,
exclusion, net charge) do not depend on any pose and so are absent here, exactly as they cancel
in that difference.

Reads guest `i`'s OLD host-guest energy from `state.host_energy[i]` (`SystemState`'s per-guest
cache) instead of recomputing it, since the host is rigid and that value is still correct as long
as guest `i` has not moved since it was last written — halving the host scan a move needs against
recomputing both poses from scratch. `host_energy_new` is the caller's to apply, on the same
acceptance-conditional basis as `ΔS`: write it into `state.host_energy[i]` on acceptance, leave
the cache untouched on rejection. `total_energy` never reads `host_energy`, so a caller that fails
to keep the cache in step with the poses still gets caught by `audit_energy!`, which recomputes
every guest's host energy from poses alone.

When `has_ewald_split(batch)`, the reciprocal-space change splits the same way `total_energy`
does: `ΔS` (sized to `batch_kvec_range(batch, n)`) still updates `state.Sk` exactly as before —
`Sk .+= ΔS` on acceptance — but via the cross-only formula (`reciprocal_cross_delta!`) rather
than `reciprocal_move_delta!`'s self-quadratic one, since `ks` is then the coupled subset alone.
`ΔS_gg`, sized to `batch_kvec_gg_range(batch, n)`, is then required and carries the guest-guest
table's own structure-factor change for the caller to apply to `state.Sk_gg` (`Sk_gg .+= ΔS_gg`)
on acceptance, via `reciprocal_move_delta!`'s ordinary formula. `ΔS_gg` is ignored (and may be
left `nothing`) when `!has_ewald_split(batch)`; guest-guest real-space terms (LJ and screened
Coulomb) attach to `alpha_gg`/`ewald_cutoff_gg` there too, matching `total_energy`.
"""
function guest_move_delta(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, n::Integer, i::Integer,
        newpos::SVector{3, T}, newq::SVector{4, T}, ΔS; ΔS_gg = nothing
    ) where {T, N}
    oldpos = state.refpoints[i]; oldq = state.orientations[i]
    fw = batch.framework_of[n]
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
    guest_types = SVector{N, Int}(batch.guest_types)
    guest_compact = Guest{T, N}(guest.sites, guest_types, guest.charges, guest.tc, guest.pc, guest.omega)

    e_old = state.host_energy[i]
    e_new = host_guest_realspace_energy(
        newpos, newq, guest_compact, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
    )
    gr = guest_range(state, n)

    split = has_ewald_split(batch)
    alpha_gg = split ? batch.alphas_gg[fw] : alpha
    ewald_cutoff_gg = batch.ewald_cutoff_gg
    ΔU_gg = guest_guest_move_delta(
        guest, state.refpoints, state.orientations, gr, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, ewald_cutoff_gg, A, invA, alpha_gg
    )

    kr = kvec_range(state, n)
    krb = batch_kvec_range(batch, n)
    if split
        isnothing(ΔS_gg) && throw(
            ArgumentError("guest_move_delta: batch has an Ewald split (has_ewald_split(batch)); ΔS_gg is required")
        )
        ΔU_cross = reciprocal_cross_delta!(
            ΔS, guest, oldpos, oldq, newpos, newq, view(batch.ks, krb), view(batch.kprefactor, krb), view(batch.Shost, krb)
        )
        krb_gg = batch_kvec_gg_range(batch, n)
        kr_gg = kvec_gg_range(state, n)
        ΔU_self = reciprocal_move_delta!(
            ΔS_gg, guest, oldpos, oldq, newpos, newq, view(batch.ks_gg, krb_gg), view(batch.kprefactor_gg, krb_gg), view(state.Sk_gg, kr_gg)
        )
        ΔU_recip = ΔU_cross + ΔU_self
    else
        ΔU_recip = reciprocal_move_delta!(ΔS, guest, oldpos, oldq, newpos, newq, view(batch.ks, krb), view(batch.kprefactor, krb), view(state.Sk, kr))
    end
    return (e_new - e_old) + ΔU_gg + ΔU_recip, e_new
end

# Real-space (LJ + Ewald) move ΔU for a batch of independent proposals, one work-item per
# proposal: `sys_of[m]`/`gidx[m]` name proposal `m`'s system and (global) moving guest index,
# `oldpos`/`oldq`/`newpos`/`newq` its poses. `guest` must carry the batch's own compact LJ type
# index (as `guest_types` does), matching `host_guest_realspace_energy`'s convention — the
# caller builds it once, the same way `widom_kernel!` builds `guest_compact`. Isolates
# real-space cost from reciprocal cost for the throughput measurement (`bench/guest_bench.jl`);
# task 5's production move kernel composes this differently (RNG-driven proposals, in-place
# `Sk`/pose updates on acceptance).
@kernel function realspace_move_kernel!(ΔU, @Const(sys_of), @Const(gidx), @Const(oldpos), @Const(oldq), @Const(newpos), @Const(newq), batch, guest, refpoints, orientations, guest_offsets, guest_types::SVector{N, Int}) where {N}
    m = @index(Global)
    n = sys_of[m]; i = gidx[m]
    fw = batch.framework_of[n]
    a0 = batch.atom_offsets[fw]
    natoms = batch.atom_offsets[fw + 1] - a0
    A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    e_old = host_guest_realspace_energy(
        oldpos[m], oldq[m], guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
    )
    e_new = host_guest_realspace_energy(
        newpos[m], newq[m], guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
    )
    gr = (guest_offsets[n] + 1):guest_offsets[n + 1]
    ΔU_gg = guest_guest_move_delta(
        guest, refpoints, orientations, gr, i, oldpos[m], oldq[m], newpos[m], newq[m],
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )
    ΔU[m] = (e_new - e_old) + ΔU_gg
end

# Reciprocal-space move ΔU for the same batch of proposals as `realspace_move_kernel!`, against
# the same system's own k-vector slice and running `Sk` (no `ΔS` output: see
# `reciprocal_move_delta_energy`).
@kernel function recip_move_kernel!(ΔU, @Const(sys_of), @Const(oldpos), @Const(oldq), @Const(newpos), @Const(newq), batch, guest, Sk, k_offsets)
    m = @index(Global)
    n = sys_of[m]
    kr = (k_offsets[n] + 1):k_offsets[n + 1]
    krb = batch_kvec_range(batch, n)
    ΔU[m] = reciprocal_move_delta_energy(
        guest, oldpos[m], oldq[m], newpos[m], newq[m], view(batch.ks, krb), view(batch.kprefactor, krb), view(Sk, kr)
    )
end
