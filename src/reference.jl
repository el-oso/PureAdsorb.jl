"""
    full_ktables(A, positions, charges, alpha, kmax) -> (ks, kprefactor, Shost, coeffs)

Unsparsified Ewald reciprocal-space tables for one periodic cell `A` holding `positions` and
`charges`: every k-vector `kvectors` enumerates (not only the host-coupled subset
`FrameworkBatch` keeps), together with its prefactor, host structure factor, and integer
coefficient triplet. This is the oracle's counterpart to the per-framework table construction
inside `FrameworkBatch`.
"""
function full_ktables(A::SMatrix{3, 3, T}, positions, charges, alpha, kmax) where {T}
    ks, w, coeffs = kvectors(A, kmax)
    V = volume(A)
    kprefactor = [w[i] * pk(dot(ks[i], ks[i]), alpha, V) for i in eachindex(ks, w)]
    Shost = structure_factor(ks, positions, charges)
    return ks, kprefactor, Shost, coeffs
end

"""
    insertion_energy_reference(pos, q, guest, ff_sigma, ff_epsilon, cutoff, ewald_cutoff,
                                hpos, htype, hq, A, invA, alpha, ks_full, kprefactor_full,
                                Shost_full) -> energy

Oracle for `insertion_energy`: the same brute-force real-space LJ and Ewald sums, but the
reciprocal-space sum runs over the FULL k-vector table of the stored cell (`ks_full`,
`kprefactor_full`, `Shost_full`, as `full_ktables` builds them) rather than only the
host-coupled subset, and includes both the host/guest cross term and the guest self term
`Σ_k pref_k |S_g(k)|²` explicitly rather than folding its orientation average into a
precomputed constant. Every stage of the kernel efficiency design is checked against this
reference.
"""
function insertion_energy_reference(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, ff_sigma, ff_epsilon, cutoff, ewald_cutoff,
        hpos, htype, hq, A, invA, alpha, ks_full, kprefactor_full, Shost_full
    ) where {T, N}
    rc_lj2 = cutoff * cutoff
    rc_ew2 = ewald_cutoff * ewald_cutoff
    gpos = map(s -> pos + rotate(q, s), guest.sites)
    E_lj = zero(T); E_sr = zero(T)
    for s in 1:N
        gp = gpos[s]
        gt = guest.types[s]; gq = guest.charges[s]
        for j in eachindex(hpos, htype, hq)
            Δ = minimum_image(A, invA, gp - hpos[j])
            r2 = dot(Δ, Δ)
            (r2 < rc_lj2 || r2 < rc_ew2) || continue
            if r2 < rc_lj2
                σ = ff_sigma[gt, htype[j]]; ε = ff_epsilon[gt, htype[j]]
                x = (σ * σ / r2)^3
                E_lj += 4 * ε * (x * x - x)
            end
            if r2 < rc_ew2
                r = sqrt(r2)
                E_sr += gq * hq[j] * erfc_dev(alpha * r) / r
            end
        end
    end
    E_lr = zero(T)
    for i in eachindex(ks_full, kprefactor_full, Shost_full)
        k = ks_full[i]
        Sg = zero(Complex{T})
        for s in 1:N
            Sg += guest.charges[s] * cis(dot(k, gpos[s]))
        end
        E_lr += kprefactor_full[i] * (2 * real(conj(Shost_full[i]) * Sg) + abs2(Sg))
    end
    return E_lj + T(KE) * (E_sr + E_lr)
end
