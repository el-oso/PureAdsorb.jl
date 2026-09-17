# Quaternion stored as (x, y, z, w): the vector part first, the scalar part `w` last.
# Rotates by the expanded Rodrigues form v + 2u × (u × v + w v), equivalent to v + q v q⁻¹
# without building a rotation matrix.
#
# kUPS's `quaternion.py` stores the opposite order, (w, x, y, z): scalar first
# (`Quaternion.identity() == [1, 0, 0, 0]`, and `_quat_to_mat` unpacks `w, x, y, z = q`).
# Its `Quaternion.random` builds the Shoemake (1992) sample
# [√(1-u₁) sin 2πu₂, √(1-u₁) cos 2πu₂, √u₁ sin 2πu₃, √u₁ cos 2πu₃] and feeds it straight into
# that (w, x, y, z) layout, so √u₁ cos 2πu₃ — often assumed to be the scalar part — lands in
# kUPS's *z* slot, not w; the scalar w is √(1-u₁) sin 2πu₂. To sample the same distribution
# here, build the SVector{4} in (x, y, z, w) order as
# (√(1-u₁) cos 2πu₂, √u₁ sin 2πu₃, √u₁ cos 2πu₃, √(1-u₁) sin 2πu₂).
function rotate(q::SVector{4}, v::SVector{3})
    u = SVector(q[1], q[2], q[3]); w = q[4]
    return v + 2 * cross(u, cross(u, v) + w * v)
end

# Energy of inserting one guest molecule at pose (pos, q) into a fixed host: LJ against every
# host site within the LJ cutoff, the real-space Ewald cross term (screened by erfc_dev,
# GPU-safe) against every host site within the (generally larger) Ewald cutoff, and the
# reciprocal cross plus guest-self terms against the host's precomputed structure factor
# Shost. `kprefactor[i]` is `w_k · pk(|k|², α, V)`, precomputed once per batch since it does
# not depend on the insertion pose. All arguments are isbits scalars, SVectors or plain array
# reads, so this runs unchanged inside a GPU kernel.
function insertion_energy(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        hpos, htype, hq, A, invA, alpha, ks, kprefactor, Shost, V
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
                σ = sigma[gt, htype[j]]; ε = epsilon[gt, htype[j]]
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
    for i in eachindex(ks, kprefactor, Shost)
        k = ks[i]
        Sg = zero(Complex{T})
        for s in 1:N
            Sg += guest.charges[s] * cis(dot(k, gpos[s]))
        end
        E_lr += kprefactor[i] * (2 * real(conj(Shost[i]) * Sg) + abs2(Sg))
    end
    return E_lj + T(KE) * (E_sr + E_lr)
end
