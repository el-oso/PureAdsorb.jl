"""
    EwaldParams{T}

Ewald summation controls: `cutoff` (Å) for the real-space and reciprocal-space sums, and the
requested relative `precision`, from which the splitting parameter α and the k-space cutoff
are derived.
"""
struct EwaldParams{T}
    cutoff::T
    precision::T
end

"""
    EwaldParams(; cutoff, precision = 1.0e-6) -> EwaldParams

Construct with a real-space cutoff (Å) and a target relative precision (default `1e-6`).
"""
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

function pk(k2, alpha, V)
    T = typeof(V)
    return 2 * T(π) / V * exp(-k2 / (4 * alpha^2)) / k2
end

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

# Clenshaw evaluation of the Chebyshev series above at x = ty/2 (Numerical Recipes' `chebev`).
# The bracket is `(ty/2)·d − dd + c₀/2`; halving the whole bracket drops half of each `dd` and
# is wrong by ~1% at z ≈ 1.
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

# The screened-Coulomb pair term only ever calls erfc on x = α·r with r < ewald_cutoff, so a
# series valid on the single interval [0, PAIR_ERFC_XMAX] replaces `erfc_dev`'s series (valid
# for every x ≥ 0) there with far fewer terms. Construction: with t = 2/(2+x), h(x) =
# log(erfc(x)/t) + x² is smooth and O(1) even as x → ∞ (erfc(x) ~ exp(-x²)/(x√π) cancels the
# x² term), so it is the same substitution `_erfccheb` uses. Restricting to x ∈ [0, xmax] means
# the substituted value x_sub = 2t-1 only ever reaches down to xlo = 2·(2/(2+xmax))-1 rather
# than -1; re-fitting a Chebyshev series on ξ, the affine rescaling of x_sub from [xlo, 1] onto
# [-1, 1], converges far faster over this narrower range than fitting on all of [-1,1] would.
# Regenerate by evaluating the standard discrete cosine transform, c_j = (2/N) Σ_k h(x_k)
# cos(jπ(k-1/2)/N), at Chebyshev nodes ξ_k = cos(π(k-1/2)/N) mapped back through
# x_sub = xlo + (ξ_k+1)(1-xlo)/2 and x_k = 2/(x_sub+1) - 2, in BigFloat precision, against any
# high-precision erfc.
const PAIR_ERFC_XMAX = 4.0
# x_sub at x = PAIR_ERFC_XMAX: t(xmax) = 2/(2+xmax), xlo = 2t(xmax) - 1.
const PAIR_ERFC_XLO = 2 * (2 / (2 + PAIR_ERFC_XMAX)) - 1

# N = 17 gives a max relative error of 9.5e-15 against SpecialFunctions.erfc over
# [0, PAIR_ERFC_XMAX] in Float64 arithmetic (dense grid plus random points) — comfortably past
# the 1e-12 target itself, which N = 15 (6.97e-13) already met; the extra margin matters because
# `insertion_energy`'s total, a sum of terms with mixed signs, can amplify a per-call bias far
# more than a per-call relative-error bound suggests (measured up to ~500x on RUBTAK 3×3×3 CO2
# poses between N = 15, whose 1e-12-level bias left the total at 3e-10 relative to the oracle,
# and N = 17, at 2e-13). Float32 cannot reach a literal 1e-7 relative bound with this
# construction: rounding in the polynomial evaluation and the final `exp` floors the achievable
# error near 1.3e-6 regardless of term count past N = 8 (measured plateau), which is still
# better than `erfc_dev`'s own Float32 accuracy on this range (measured 1.8e-6); at Float32's
# much coarser noise floor, term count beyond that plateau has no comparable effect on the total.
_pair_erfc_coef(::Type{Float64}) = (
    -0.888894912038317, 0.4479385778277769, -0.00021608983881297707, -0.003410058708799526,
    8.555152981629436e-5, 5.532830637302869e-5, -5.2098383667087075e-6, -8.380063246741613e-7,
    1.9764068422519322e-7, 1.7838566371819855e-9, -5.26989861621132e-9, 5.378177355603348e-10,
    7.748359588558286e-11, -2.3770397709020423e-11, 8.934868696420547e-13, 5.207576546902621e-13,
    -9.42604840757663e-14,
)
_pair_erfc_coef(::Type{Float32}) = (
    -0.8888949f0, 0.4479386f0, -0.00021608984f0, -0.0034100588f0,
    8.555145f-5, 5.5327768f-5, -5.2045684f-6, -8.397902f-7,
)

# erfc(z) for the screened-Coulomb pair term, z = α·r ∈ [0, PAIR_ERFC_XMAX]. `FrameworkBatch`
# checks the batch's α·ewald_cutoff against this bound at construction, so z is in range by the
# time a kernel calls this.
function pair_erfc_dev(z::T) where {T}
    cof = _pair_erfc_coef(T)
    xlo = T(PAIR_ERFC_XLO)
    t = 2 / (2 + z)
    ξ = 2 * (2t - 1 - xlo) / (1 - xlo) - one(T)
    d = zero(T); dd = zero(T)
    for j in length(cof):-1:2
        tmp = d
        d = 2ξ * d - dd + cof[j]
        dd = tmp
    end
    return t * exp(ξ * d - dd + cof[1] / 2 - z * z)
end

# Half-space enumeration with kUPS's weighting: n₁ ≥ 0, and every vector with n₁ > 0 stands
# in for its mirror image with weight 2. Since aᵢ·bⱼ = 2π δᵢⱼ, nᵢ = k·aᵢ/2π, so any k with
# |k| ≤ kmax has |nᵢ| ≤ kmax·|aᵢ|/2π, where |aᵢ| = norm(A[:,i]) is the lattice vector's own
# length; the shorter perpendicular length underbounds this for a triclinic cell.
#
# The third return value gives each kept vector's integer coefficients (n1, n2, n3) in the
# reciprocal basis (`k == B * SVector(n...)`); a k-vector is coupled to a supercell built by
# replicating the stored cell by `replication` iff every coefficient is divisible by the
# corresponding factor, since only then does the phase `k · r` repeat identically in every
# copy of the unreplicated cell.
function kvectors(A::SMatrix{3, 3, T}, kmax) where {T}
    B = reciprocal_basis(A)
    n = ntuple(i -> ceil(Int, kmax * norm(A[:, i]) / (2π)), 3)
    ks = SVector{3, T}[]
    w = T[]
    coeffs = NTuple{3, Int}[]
    for n1 in 0:n[1], n2 in (-n[2]):n[2], n3 in (-n[3]):n[3]
        (iszero(n1) && iszero(n2) && iszero(n3)) && continue
        k = B * SVector(n1, n2, n3)
        norm(k) <= kmax || continue
        push!(ks, k)
        push!(w, iszero(n1) ? one(T) : T(2))
        push!(coeffs, (n1, n2, n3))
    end
    return ks, w, coeffs
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
