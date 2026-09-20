"""
    theta_F(::Type{T}) -> T

Smallest value ``\\theta`` of type `T` for which `exp(-θ)` underflows to zero. `exp` is
monotone non-increasing on non-negative arguments, so every value larger than `theta_F(T)`
also underflows: a Boltzmann weight `exp(-ΔU/kT)` is exactly `0.0` in `T` precisely when
`ΔU/kT > theta_F(T)`. Found by bisection on the float grid itself (about 745.13 for `Float64`,
about 103.97 for `Float32`), not read off a table, so it tracks whatever `exp` implementation
this Julia/hardware combination actually runs.
"""
function theta_F(::Type{T}) where {T}
    lo, hi = zero(T), T(2000)
    iszero(exp(-hi)) || throw(ArgumentError("theta_F($T): $hi is too small for exp(-θ) to underflow to zero"))
    while nextfloat(lo) != hi
        mid = lo + (hi - lo) / 2
        iszero(exp(-mid)) ? (hi = mid) : (lo = mid)
    end
    return hi
end

# One guest-site/host-atom pair's energy at separation r: the same Lennard-Jones plus
# screened-Coulomb split `insertion_energy` sums over every pair, evaluated for one pair alone.
pair_energy(r::T, σ::T, ε::T, K::T, alpha::T) where {T} = lj_pair_energy(r * r, σ, ε) + K * pair_erfc_dev(alpha * r) / r

"""
    find_r0(σ, ε, K, alpha, r_lj) -> r0

Zero of the pair energy `pair_energy(r, σ, ε, K, alpha)` below its own minimum, for an
attractive pair (`K` negative) with a finite Lennard-Jones well (`ε` positive): the pair energy
diverges to `+Inf` as `r → 0` (the repulsive `r⁻¹²` term dominates the attractive `r⁻¹` Coulomb
term) while it is already negative (at most `-ε`, itself negative) at the Lennard-Jones-alone
minimum radius `σ·2^(1/6)`, so a sign change is guaranteed somewhere below that radius.
Bisection needs only a lower bracket, found by halving `σ·2^(1/6)` until the sign flips.

Clamped to `r_lj` (the Lennard-Jones cutoff): `pair_energy` never truncates its Lennard-Jones
term, but the actual insertion energy does (`r < r_lj`), so the unclamped root can lie beyond
the range over which the Lennard-Jones term is ever really present. `hardcore_bound`'s envelope
`u_ah ≥ 0` on `(0, r0]` only holds where the Lennard-Jones term is included, i.e. up to `r_lj`.
"""
function find_r0(σ::T, ε::T, K::T, alpha::T, r_lj::T) where {T}
    ε > zero(T) || throw(ArgumentError("find_r0: ε=$ε must be positive (no Lennard-Jones well to bracket a root against)"))
    K < zero(T) || throw(ArgumentError("find_r0: K=$K must be negative (attractive pair)"))
    hi = σ * T(2)^(one(T) / 6)
    pair_energy(hi, σ, ε, K, alpha) < zero(T) ||
        throw(ArgumentError("find_r0: pair_energy at σ·2^(1/6)=$hi is not negative for σ=$σ, ε=$ε, K=$K"))
    lo = hi
    while pair_energy(lo, σ, ε, K, alpha) <= zero(T)
        lo /= 2
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        pair_energy(mid, σ, ε, K, alpha) > zero(T) ? (lo = mid) : (hi = mid)
    end
    return min((lo + hi) / 2, r_lj)
end

"""
    hardcore_bound(guest, sigma, epsilon, positions, types, charges, atoms, r_lj, alpha, kprefactor, Shost) -> B_s

Rigorous lower-magnitude bound on everything `insertion_energy` sums except one designated
"trigger" pair: for every other guest-site/host-atom pair in `atoms`, the most negative value
`u_ah` can take anywhere on `(0, cutoff]` (`-ε` when the Coulomb term `K = KE·q_a·q_h` is
non-negative; `-ε - |K|·pair_erfc_dev(alpha·r0)/r0` at `r0 = find_r0(σ, ε, K, alpha, r_lj)` when
`K` is negative, since the Coulomb term's magnitude is largest, for `r ≥ r0`, at `r0` itself),
plus the reciprocal-space cross-term bound `KE·Σ|q_guest|·Σ_k 2·kprefactor_k·|Shost_k|` (from
`|Re(conj(Shost)·Sg)| ≤ |Shost|·|Sg| ≤ |Shost|·Σ|q_guest|`). `r0` is clamped to `r_lj` (the
Lennard-Jones cutoff): `find_r0`'s zero of the untruncated pair energy can lie beyond `r_lj`,
past which the true `u_ah` no longer includes the Lennard-Jones term at all. Returns `Inf` when
any pair combines an attractive Coulomb term with no Lennard-Jones repulsion at all (ε zero, K
negative): the pair energy is then unbounded below as `r → 0` and no finite bound exists.
"""
function hardcore_bound(
        guest::Guest{T, N}, sigma, epsilon, positions, types, charges, atoms, r_lj::T, alpha::T, kprefactor, Shost
    ) where {T, N}
    acc = zero(T)
    for a in 1:N
        gt = guest.types[a]
        qa = guest.charges[a]
        for h in atoms
            ht = types[h]
            σ = sigma[gt, ht]
            ε = epsilon[gt, ht]
            K = T(KE) * qa * charges[h]
            if K >= zero(T)
                acc += ε
            else
                iszero(ε) && return T(Inf)
                r0 = find_r0(σ, ε, K, alpha, r_lj)
                acc += ε + abs(K) * pair_erfc_dev(alpha * r0) / r0
            end
        end
    end
    Rs = zero(T)
    for i in eachindex(kprefactor, Shost)
        Rs += kprefactor[i] * abs(Shost[i])
    end
    return acc + 2 * T(KE) * sum(abs, guest.charges) * Rs
end

"""
    kmin_table(guest, types, charges, atoms, ntypes) -> Matrix{T}

`K_min[a, t]`, the most negative `K_ah = KE·q_a·q_h` over `atoms` of compact Lennard-Jones type
`t` for guest site `a` (zero when no such atom carries a charge that makes `K_ah` negative).
Depends only on charges and the guest, not on temperature, so it is computed once at
`FrameworkBatch` construction rather than per `widom` call.
"""
function kmin_table(guest::Guest{T, N}, types, charges, atoms, ntypes::Integer) where {T, N}
    kmin = zeros(T, N, ntypes)
    for a in 1:N
        qa = guest.charges[a]
        for h in atoms
            ht = types[h]
            K = T(KE) * qa * charges[h]
            K < kmin[a, ht] && (kmin[a, ht] = K)
        end
    end
    return kmin
end

"""
    find_rho2(sigma_at, epsilon_at, kmin_at, margin, r_lj) -> ρ²

Squared rejection radius `ρ_at²` (E3's "Core radii"): `ρ_at` is the first root, scanning up from
`r → 0`, of `LJ_at(r) − |K_min(a,t)|/r = margin`. Every separation under `ρ_at` then satisfies
the rejection condition for any host atom of type `t`, since `LJ_at(r) − |K_min(a,t)|/r` lower
bounds the true pair energy of guest site `a` with any such atom (its actual `K_ah` is no more
negative than `K_min(a,t)`, so its actual Coulomb term is no more negative than
`-|K_min(a,t)|/r`). An infinite `margin` (a system whose `B_s` is infinite) disables rejection:
the returned `ρ²` is zero, so the phase-0 kernel's `r² < ρ²` check never trips.

`ρ_at` is clamped to `r_lj` (the Lennard-Jones cutoff): the root solves an equation that assumes
the Lennard-Jones term is present, but the true insertion energy only includes it for `r < r_lj`,
so an unclamped root beyond `r_lj` would flag poses whose true energy has no Lennard-Jones
repulsion at all.
"""
function find_rho2(sigma_at::T, epsilon_at::T, kmin_at::T, margin::T, r_lj::T) where {T}
    isinf(margin) && return zero(T)
    f(r) = lj_pair_energy(r * r, sigma_at, epsilon_at) - abs(kmin_at) / r - margin
    hi = sigma_at
    while f(hi) >= zero(T)
        hi *= 2
    end
    lo = hi
    while f(lo) < zero(T)
        lo /= 2
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        f(mid) > zero(T) ? (lo = mid) : (hi = mid)
    end
    return min((lo + hi) / 2, r_lj)^2
end
