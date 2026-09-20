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

# Energy of inserting one guest molecule at pose (pos, q) into a fixed host, visiting only the
# host atoms that can be within `cutoff`/`ewald_cutoff` of some guest site via a cell-list
# stencil around `pos`'s home cell, plus the reciprocal cross term against the host's
# precomputed structure factor Shost over only the k-vectors coupled to this framework's
# replication (`FrameworkBatch` keeps no others). The guest self term `Σ_k pref_k |S_g(k)|²` is
# orientation-dependent but pose-otherwise-fixed, so its orientation average over the full k set
# is folded into `constant_offset` instead of recomputed here (see `FrameworkBatch`'s
# docstring). `kprefactor[i]` is `w_k · pk(|k|², α, V)`, precomputed once per batch since it does
# not depend on the insertion pose.
#
# `positions`/`types`/`charges` are the WHOLE batch's arrays; `atom_base` is this system's
# `atom_offsets` entry (0-based) and `cell_offsets` is this system's own
# `prod(ncells) + 1`-entry slice of the batch's `cell_offsets` (local offset `c`'s atoms are
# `cell_offsets[c+1]+1:cell_offsets[c+2]`, shifted by `atom_base`, matching `FrameworkBatch`'s
# cell-sort order). `ncells`/`reach` are this system's own grid dimensions and stencil
# half-widths. Every host atom within `cutoff + r_guest`/`ewald_cutoff + r_guest` of `pos` lies
# in a visited cell (`FrameworkBatch`'s construction guard: the framework's minimum image
# exceeds `2*(max(cutoff, ewald_cutoff) + r_guest)`), so one minimum image of `pos - positions[j]`
# per host atom, plus each guest site's own (already rotated) offset added without a further
# minimum image, is exact.
#
# All arguments are isbits scalars, SVectors, or plain/view array reads with no throwing
# branches, so this runs unchanged inside a GPU kernel: home-cell and wrap arithmetic uses
# `unsafe_trunc`/branch-free wrapping (`home_cell_dev`, `wrap_cell`) rather than `floor`/`mod`
# by a runtime value.
function insertion_energy(
        pos::SVector{3, T}, q::SVector{4, T}, guest::Guest{T, N}, sigma, epsilon, cutoff, ewald_cutoff,
        positions, types, charges, atom_base::Integer, ncells::SVector{3, Int32}, reach::SVector{3, Int32},
        cell_offsets, A, invA, alpha, ks, kprefactor, Shost
    ) where {T, N}
    rc_lj2 = cutoff * cutoff
    rc_ew2 = ewald_cutoff * ewald_cutoff
    gsites = map(s -> rotate(q, s), guest.sites)
    E_lj = zero(T); E_sr = zero(T)

    f = invA * pos
    n1 = ncells[1]; n2 = ncells[2]; n3 = ncells[3]
    m1 = reach[1]; m2 = reach[2]; m3 = reach[3]
    h1 = home_cell_dev(f[1], n1); h2 = home_cell_dev(f[2], n2); h3 = home_cell_dev(f[3], n3)
    start1, count1 = stencil_start_count(h1, m1, n1)
    start2, count2 = stencil_start_count(h2, m2, n2)
    start3, count3 = stencil_start_count(h3, m3, n3)

    for t3 in zero(Int32):(count3 - one(Int32))
        c3 = wrap_cell(start3 + t3, n3)
        for t2 in zero(Int32):(count2 - one(Int32))
            c2 = wrap_cell(start2 + t2, n2)
            for t1 in zero(Int32):(count1 - one(Int32))
                c1 = wrap_cell(start1 + t1, n1)
                c = cell_linear(c1, c2, c3, n1, n2)
                a0 = atom_base + cell_offsets[c + 1] + 1
                a1 = atom_base + cell_offsets[c + 2]
                for j in a0:a1
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
