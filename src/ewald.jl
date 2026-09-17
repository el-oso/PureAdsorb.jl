struct EwaldParams{T}
    cutoff::T
    precision::T
end
EwaldParams(; cutoff, precision = 1.0e-6) = EwaldParams(promote(float(cutoff), float(precision))...)

# α such that erfc(α r_c) = r_c · ε/2, the kUPS selection when the real-space cutoff is fixed.
function ewald_alpha(cutoff, precision)
    target = cutoff * precision / 2
    lo, hi = zero(target), oftype(target, 20)
    f(z) = erfc(z) - target
    f(hi) < 0 || throw(ArgumentError("cannot reach precision $precision with cutoff $cutoff"))
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) > 0 ? (lo = mid) : (hi = mid)
        hi - lo < 1.0e-12 && break
    end
    return (lo + hi) / 2 / cutoff
end

ewald_kmax(alpha, precision) = 2 * alpha * sqrt(-log(precision / 2))

pk(k2, alpha, V) = (2π / V) * exp(-k2 / (4 * alpha^2)) / k2

# Complementary error function callable inside GPU kernels: a Chebyshev fit valid for every
# z ≥ 0, following the construction in Numerical Recipes 3rd ed. §6.2.2. With
# t = 2/(2+z), x = 2t - 1 ∈ [-1, 1], the 28 coefficients below are the Chebyshev series of
# g(x) = log(erfc(z)/t) + z² (equivalently z = 4/(x+1) - 2), fit by the standard discrete
# cosine transform at the Chebyshev nodes x_k = cos(π(k - 1/2)/28), k = 1..28, in BigFloat
# precision. Regenerate by evaluating that transform against any high-precision erfc.
const _ERFC_COF = (
    -1.3026537197817094, 6.419697923564902e-1, 1.9476473204185836e-2,
    -9.561514786808632e-3, -9.465953444820369e-4, 3.6683949785276145e-4, 4.252332480690777e-5,
    -2.0278578112534242e-5, -1.6242900046470256e-6, 1.3036558355805232e-6, 1.5626441722066142e-8,
    -8.523809591492654e-8, 6.5290544390988515e-9, 5.0593434955514685e-9, -9.913641564930322e-10,
    -2.2736512229318417e-10, 9.646791102014962e-11, 2.394038083065756e-12, -6.886027526536277e-12,
    8.944879271730368e-13, 3.130921408127142e-13, -1.1270822525221733e-13, 3.8108713588528985e-16,
    7.106125536922576e-15, -1.5230878975373303e-15, -9.464871412184462e-17, 1.218683266906788e-16,
    -3.0494735135424696e-17,
)

# Clenshaw evaluation of the Chebyshev series above at x = ty/2, in the standard form
# f(x) = (ty/2)·d - dd + c₀/2 (Numerical Recipes' `chebev`); halving the whole bracket
# instead, as a naive reading of the recursion suggests, discards half of every `dd` term
# and is wrong by ~1% at z = O(1).
function _erfccheb(z)
    T = typeof(z)
    t = 2 / (2 + z)
    ty = 4t - 2
    d = zero(T)
    dd = zero(T)
    for j in length(_ERFC_COF):-1:2
        tmp = d
        d = ty * d - dd + T(_ERFC_COF[j])
        dd = tmp
    end
    return t * exp(-z * z + (ty / 2) * d - dd + T(_ERFC_COF[1]) / 2)
end
erfc_dev(x) = x >= 0 ? _erfccheb(x) : 2 - _erfccheb(-x)

# Half-space enumeration with kUPS's weighting: n₁ ≥ 0, and every vector with n₁ > 0 stands
# in for its mirror image with weight 2. Since aᵢ·bⱼ = 2π δᵢⱼ, nᵢ = k·aᵢ/2π, so any k with
# |k| ≤ kmax has |nᵢ| ≤ kmax·|aᵢ|/2π, where |aᵢ| = norm(A[:,i]) is the lattice vector's own
# length; the shorter perpendicular length underbounds this for a triclinic cell.
function kvectors(A::SMatrix{3, 3, T}, kmax) where {T}
    B = reciprocal_basis(A)
    n = ntuple(i -> ceil(Int, kmax * norm(A[:, i]) / (2π)), 3)
    ks = SVector{3, T}[]
    w = T[]
    for n1 in 0:n[1], n2 in (-n[2]):n[2], n3 in (-n[3]):n[3]
        (iszero(n1) && iszero(n2) && iszero(n3)) && continue
        k = B * SVector(n1, n2, n3)
        norm(k) <= kmax || continue
        push!(ks, k)
        push!(w, iszero(n1) ? one(T) : T(2))
    end
    return ks, w
end

function structure_factor(ks, positions, charges)
    T = float(eltype(charges))
    S = zeros(Complex{T}, length(ks))
    for (i, k) in enumerate(ks)
        acc = zero(Complex{T})
        for j in eachindex(positions, charges)
            acc += charges[j] * cis(dot(k, positions[j]))
        end
        S[i] = acc
    end
    return S
end

# Full Ewald energy of one periodic system; the reference for every kernel that evaluates a
# difference of such energies. The reciprocal-space sum has no notion of molecules, so it
# always includes the erf(αr)/r part of every pair's interaction, intramolecular or not;
# excluding an intramolecular pair therefore subtracts only that erf part (the direct
# distance, not a periodic image), leaving its erfc(αr)/r part out of both sums entirely.
function ewald_energy(A::SMatrix{3, 3, T}, positions, charges, molecules, alpha, cutoff, ks, weights) where {T}
    Base.require_one_based_indexing(positions, charges, molecules)
    invA = inv(A)
    V = volume(A)
    E_sr = zero(T)
    E_excl = zero(T)
    for i in eachindex(positions), j in eachindex(positions)
        j > i || continue
        if molecules[i] == molecules[j]
            r = norm(positions[j] - positions[i])
            E_excl -= charges[i] * charges[j] * (one(T) - erfc_dev(alpha * r)) / r
        else
            Δ = minimum_image(A, invA, positions[j] - positions[i])
            r = norm(Δ)
            r < cutoff && (E_sr += charges[i] * charges[j] * erfc_dev(alpha * r) / r)
        end
    end
    S = structure_factor(ks, positions, charges)
    E_lr = zero(T)
    for i in eachindex(ks, weights, S)
        E_lr += weights[i] * pk(dot(ks[i], ks[i]), alpha, V) * abs2(S[i])
    end
    E_self = -alpha / sqrt(T(π)) * sum(abs2, charges)
    Q = sum(charges)
    E_net = -T(π) / (2 * V * alpha^2) * Q^2
    return KE * (E_sr + E_lr + E_self + E_excl + E_net)
end
