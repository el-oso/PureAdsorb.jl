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

# Lennard-Jones energy of one pair at squared distance r2, shared by `insertion_energy` and the
# hard-core rejection bound in `reject.jl`.
lj_pair_energy(r2::T, σ::T, ε::T) where {T} = (x = (σ * σ / r2)^3; 4 * ε * (x * x - x))

# Energy of inserting one guest molecule at pose (pos, q) into a fixed host, looping linearly
# over the system's `natoms` atoms (this is the fastest form for a framework this size — see the
# efficiency design's E2 cellwidth measurements — now that the hard-core rejection stage (E3)
# has already screened out most poses before this ever runs), plus the reciprocal cross term
# against the host's precomputed structure factor Shost over only the k-vectors coupled to this
# framework's replication (`FrameworkBatch` keeps no others). The guest self term
# `Σ_k pref_k |S_g(k)|²` is orientation-dependent but pose-otherwise-fixed, so its orientation
# average over the full k set is folded into `constant_offset` instead of recomputed here (see
# `FrameworkBatch`'s docstring). `kprefactor[i]` is `w_k · pk(|k|², α, V)`, precomputed once per
# batch since it does not depend on the insertion pose.
#
# `positions`/`types`/`charges` are the WHOLE batch's arrays; `atom_base` is this system's
# `atom_offsets` entry (0-based) and `natoms` its atom count, so this system's atoms are
# `positions[(atom_base+1):(atom_base+natoms)]`. Every host atom is visited, so a single minimum
# image of `pos - positions[j]` per host atom, plus each guest site's own (already rotated)
# offset added without a further minimum image, is exact whenever the framework's minimum image
# exceeds `2*(max(cutoff, ewald_cutoff) + r_guest)` (`FrameworkBatch`'s construction guard).
#
# All arguments are isbits scalars, SVectors, or plain/view array reads with no throwing
# branches, so this runs unchanged inside a GPU kernel.
function insertion_energy(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        positions, types, charges, atom_base::Integer, natoms::Integer, A, invA, alpha, ks, kprefactor, Shost
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
    E_lr = zero(T)
    # `ks`, `kprefactor` and `Shost` are index-matched by construction; a multi-array `eachindex`
    # would additionally check that here, but its mismatch branch builds an error string, which
    # GPUCompiler cannot compile.
    for i in eachindex(ks)
        k = ks[i]
        Sg = zero(Complex{T})
        for s in 1:N
            Sg += guest.charges[s] * cis(dot(k, pos + gsites[s]))
        end
        E_lr += kprefactor[i] * 2 * real(conj(Shost[i]) * Sg)
    end
    return E_lj + T(KE) * (E_sr + E_lr)
end
