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
# `orientations`) of system `n` from `(oldpos, oldq)` to `(newpos, newq)`: every OTHER guest in
# `gr` pairs once against the old pose and once against the new one. The constant terms
# (self energy, exclusion, tail correction, net charge) do not depend on any guest's pose and so
# never appear in a `ΔU`.
function guest_guest_move_delta(
        guest::Guest{T, N}, refpoints, orientations, gr::UnitRange, i::Integer,
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
    host_guest_realspace_energy(pos, q, guest, sigma, epsilon, cutoff, ewald_cutoff,
                                 positions, types, charges, atom_base, natoms, A, invA, alpha) -> energy

Lennard-Jones plus real-space (screened-Coulomb) energy between one guest pose and the host
atoms `positions[(atom_base+1):(atom_base+natoms)]`: `insertion_energy`'s real-space loop with
the reciprocal-space cross term omitted, since Milestone B's reciprocal energy is a single sum
over the running total structure factor instead (`total_reciprocal_energy`,
`reciprocal_move_delta!`). Isolating this from the reciprocal term is what lets the per-move
throughput measurement (`bench/guest_bench.jl`) time real-space and reciprocal work separately.
All arguments are isbits scalars, SVectors, or plain/view array reads with no throwing branches,
so this runs unchanged inside a GPU kernel.
"""
function host_guest_realspace_energy(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        positions, types, charges, atom_base::Integer, natoms::Integer, A, invA, alpha
    ) where {T, N}
    rc_lj2 = cutoff * cutoff
    rc_ew2 = ewald_cutoff * ewald_cutoff
    gsites = map(s -> rotate(q, s), guest.sites)
    E_lj = zero(T); E_sr = zero(T)
    for j in (atom_base + 1):(atom_base + natoms)
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
    ΔU = zero(T)
    for i in eachindex(ks)
        k = ks[i]
        Sold = zero(Complex{T}); Snew = zero(Complex{T})
        for s in 1:N
            Sold += guest.charges[s] * cis(dot(k, oldpos + rotate(oldq, guest.sites[s])))
            Snew += guest.charges[s] * cis(dot(k, newpos + rotate(newq, guest.sites[s])))
        end
        ds = Snew - Sold
        ΔS[i] = ds
        ΔU += kprefactor[i] * (2 * real(conj(Sk[i]) * ds) + abs2(ds))
    end
    return T(KE) * ΔU
end

# Same formula as `reciprocal_move_delta!`, without writing `ΔS`: for the throughput measurement
# and any caller that only needs the energy change (a rejected move never applies `ΔS` anyway).
function reciprocal_move_delta_energy(
        guest::Guest{T, N}, oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        ks, kprefactor, Sk
    ) where {T, N}
    ΔU = zero(T)
    for i in eachindex(ks)
        k = ks[i]
        Sold = zero(Complex{T}); Snew = zero(Complex{T})
        for s in 1:N
            Sold += guest.charges[s] * cis(dot(k, oldpos + rotate(oldq, guest.sites[s])))
            Snew += guest.charges[s] * cis(dot(k, newpos + rotate(newq, guest.sites[s])))
        end
        ds = Snew - Sold
        ΔU += kprefactor[i] * (2 * real(conj(Sk[i]) * ds) + abs2(ds))
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
    Ng = length(gr)
    A = batch.cells[n]; invA = batch.invcells[n]; alpha = batch.alphas[n]
    a0 = batch.atom_offsets[n]; natoms = batch.atom_offsets[n + 1] - a0
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

    E_lj_gg, E_sr_gg = guest_guest_energy(
        guest, state.refpoints, state.orientations, gr, batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )
    E += E_lj_gg + T(KE) * E_sr_gg

    E += total_reciprocal_energy(view(batch.kprefactor, kr), view(state.Sk, kr), view(batch.Shost, kr))

    gself, gexcl = guest_self_terms(guest, alpha, batch.ewald_cutoff)
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
    E += guest_tail_correction(ff, host_counts, gcounts, Ng, batch.volumes[n])

    Qh = sum(view(batch.charges, (a0 + 1):(a0 + natoms)))
    Qg = sum(guest.charges)
    E += -T(KE) * T(π) / (2 * batch.volumes[n] * alpha^2) * ((Qh + Ng * Qg)^2 - Qh^2)

    return E
end

# One entry per system, matching `SystemState.energy`'s own layout.
total_energy(batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T}, ff::ForceField{T}) where {T} =
    [total_energy(batch, state, guest, ff, n) for n in 1:batch.nsys]

"""
    guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS) -> ΔU

Total energy change from moving guest `i` (a global index into `state.refpoints`/
`state.orientations`) of system `n` to pose `(newpos, newq)`, writing the k-indexed
structure-factor change into `ΔS` (`reciprocal_move_delta!`; apply it to `state.Sk` on
acceptance). Equal, to accumulated rounding, to the difference of two `total_energy` calls
before and after applying the move — the per-guest constant terms (tail correction, self energy,
exclusion, net charge) do not depend on any pose and so are absent here, exactly as they cancel
in that difference.
"""
function guest_move_delta(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, n::Integer, i::Integer,
        newpos::SVector{3, T}, newq::SVector{4, T}, ΔS
    ) where {T, N}
    oldpos = state.refpoints[i]; oldq = state.orientations[i]
    A = batch.cells[n]; invA = batch.invcells[n]; alpha = batch.alphas[n]
    a0 = batch.atom_offsets[n]; natoms = batch.atom_offsets[n + 1] - a0
    guest_types = SVector{N, Int}(batch.guest_types)
    guest_compact = Guest{T, N}(guest.sites, guest_types, guest.charges, guest.tc, guest.pc, guest.omega)

    e_old = host_guest_realspace_energy(
        oldpos, oldq, guest_compact, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
    )
    e_new = host_guest_realspace_energy(
        newpos, newq, guest_compact, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
    )
    gr = guest_range(state, n)
    ΔU_gg = guest_guest_move_delta(
        guest, state.refpoints, state.orientations, gr, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )
    kr = kvec_range(state, n)
    ΔU_recip = reciprocal_move_delta!(ΔS, guest, oldpos, oldq, newpos, newq, view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr))
    return (e_new - e_old) + ΔU_gg + ΔU_recip
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
    a0 = batch.atom_offsets[n]
    natoms = batch.atom_offsets[n + 1] - a0
    A = batch.cells[n]; invA = batch.invcells[n]; alpha = batch.alphas[n]
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
    ΔU[m] = reciprocal_move_delta_energy(
        guest, oldpos[m], oldq[m], newpos[m], newq[m], view(batch.ks, kr), view(batch.kprefactor, kr), view(Sk, kr)
    )
end
