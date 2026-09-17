"""
    WidomResult{T}

Per-system result of Widom test-particle insertion, with `W = exp(-ΔU/kT)` the Boltzmann
insertion weight:

- `mu_ex` (eV): excess chemical potential, `-kT·ln⟨W⟩`.
- `K_H` (Å³/eV): Henry's constant, `V⟨W⟩/kT`.
- `q_st` (eV): `kT - ⟨ΔU·W⟩/⟨W⟩`, kUPS's `heat_of_adsorption` (the negative of the isosteric
  heat of adsorption).
- `mu_ex_err`, `K_H_err`, `q_st_err`: standard errors of the above, from block statistics.
- `nsamples`: total insertions used; `nblocks`: number of blocks the standard errors are from.
"""
struct WidomResult{T}
    mu_ex::T
    mu_ex_err::T
    K_H::T
    K_H_err::T
    q_st::T
    q_st_err::T
    nsamples::Int
    nblocks::Int
end

# Insertions are dealt round-robin within each chunk; per-system counts across a chunk differ
# by at most one.
function random_poses!(rng::AbstractRNG, sys_of, rpos, quat, nsys)
    for i in eachindex(sys_of, rpos, quat)
        sys_of[i] = Int32(mod1(i, nsys))
        rpos[i] = rand(rng, eltype(rpos))
        u1, u2, u3 = rand(rng), rand(rng), rand(rng)
        a, b = sqrt(1 - u1), sqrt(u1)
        # Shoemake (1992) sample mapped into (x, y, z, w) order per the comment block above
        # `rotate`: (√(1-u₁) cos 2πu₂, √u₁ sin 2πu₃, √u₁ cos 2πu₃, √(1-u₁) sin 2πu₂).
        quat[i] = eltype(quat)(a * cospi(2u2), b * sinpi(2u3), b * cospi(2u3), a * sinpi(2u2))
    end
    return nothing
end

@kernel function widom_kernel!(ΔU, @Const(sys_of), @Const(rpos), @Const(quat), batch, guest)
    i = @index(Global)
    s = sys_of[i]
    a0 = batch.atom_offsets[s] + 1
    a1 = batch.atom_offsets[s + 1]
    k0 = batch.k_offsets[s] + 1
    k1 = batch.k_offsets[s + 1]
    A = batch.cells[s]
    invA = batch.invcells[s]
    pos = A * rpos[i]
    e = insertion_energy(
        pos, quat[i], guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        view(batch.positions, a0:a1), view(batch.types, a0:a1), view(batch.charges, a0:a1),
        A, invA, batch.alphas[s], view(batch.ks, k0:k1), view(batch.kprefactor, k0:k1), view(batch.Shost, k0:k1)
    )
    ΔU[i] = e + batch.constant_offset[s]
end

"""
    widom(batch::FrameworkBatch, guest::Guest; T, ninsert, backend = CPU(), seed = 0,
          chunk = 2^16, nblocks = 10) -> Vector{WidomResult}

Widom test-particle insertion at temperature `T` (K): `ninsert` random poses per system give
the Boltzmann-weighted insertion average `W = ⟨exp(-ΔU/kT)⟩`, split into `nblocks` blocks for a
standard-error estimate, and reduced to a `WidomResult` per system in `batch`. Insertions are
generated and evaluated in chunks of `chunk` poses per `backend` kernel launch. `seed` sets the
random-pose generator.
"""
function widom(
        batch::FrameworkBatch{F}, guest::Guest{F}; T, ninsert::Integer, backend = CPU(), seed = 0,
        chunk::Integer = 2^16, nblocks::Integer = 10
    ) where {F}
    nblocks >= 2 || throw(ArgumentError("nblocks=$nblocks: at least two blocks are needed for a standard error"))
    chunk >= 1 || throw(ArgumentError("chunk=$chunk must be ≥ 1"))
    nsys = batch.nsys
    ninsert >= 2 * nblocks * nsys ||
        throw(ArgumentError("ninsert=$ninsert is too small: need at least 2·nblocks·nsys = $(2 * nblocks * nsys)"))
    kT = F(KB * T)
    dbatch = adapt(backend, batch)
    rng = Xoshiro(seed)
    sys_of = Vector{Int32}(undef, chunk)
    rpos = Vector{SVector{3, F}}(undef, chunk)
    quat = Vector{SVector{4, F}}(undef, chunk)
    ΔU_h = Vector{F}(undef, chunk)
    dsys = adapt(backend, sys_of)
    drpos = adapt(backend, rpos)
    dquat = adapt(backend, quat)
    dΔU = adapt(backend, ΔU_h)
    sW = zeros(F, nsys, nblocks)
    sUW = zeros(F, nsys, nblocks)
    n = zeros(Int, nsys, nblocks)
    kern = widom_kernel!(backend)
    done = 0
    # Floor division so the clamp below absorbs the remainder into the last block instead of
    # leaving it with zero samples (the guard above guarantees floor(ninsert/nblocks) >= 2·nsys).
    block_len = ninsert ÷ nblocks
    while done < ninsert
        m = min(chunk, ninsert - done)
        random_poses!(rng, view(sys_of, 1:m), view(rpos, 1:m), view(quat, 1:m), nsys)
        copyto!(dsys, sys_of)
        copyto!(drpos, rpos)
        copyto!(dquat, quat)
        kern(dΔU, dsys, drpos, dquat, dbatch, guest; ndrange = m)
        KernelAbstractions.synchronize(backend)
        copyto!(ΔU_h, dΔU)
        for i in 1:m
            s = sys_of[i]
            blk = min(nblocks, (done + i - 1) ÷ block_len + 1)
            w = exp(-ΔU_h[i] / kT)
            sW[s, blk] += w
            sUW[s, blk] += ΔU_h[i] * w
            n[s, blk] += 1
        end
        done += m
    end
    return [_reduce(view(sW, s, :), view(sUW, s, :), view(n, s, :), kT, batch.volumes[s]) for s in 1:nsys]
end

# Block-averaged mean and standard error of the Widom weight W and the energy-weighted
# average UW = ⟨ΔU·exp(-ΔU/kT)⟩, propagated through μ_ex = -kT log W, K_H = V·W/kT and
# q_st = kT - UW/W via the delta method (first-order error propagation of a ratio of means).
function _reduce(sW, sUW, n, kT, V)
    T = eltype(sW)
    nb = length(sW)
    all(>(0), n) || throw(ArgumentError("block $(findfirst(iszero, n)) of $nb has zero samples"))
    mW = sW ./ n
    mUW = sUW ./ n
    W = sum(sW) / sum(n)
    UW = sum(sUW) / sum(n)
    varW = sum(abs2, mW .- W) / (nb - 1)
    varUW = sum(abs2, mUW .- UW) / (nb - 1)
    cov = sum((mW .- W) .* (mUW .- UW)) / (nb - 1)
    semW = sqrt(varW / nb)
    ratio = UW / W
    var_ratio = iszero(UW) ? zero(T) : ratio^2 * (varUW / UW^2 + varW / W^2 - 2cov / (UW * W)) / nb
    # The first-order delta method can return a negative variance when the block covariance
    # term dominates; the true variance is non-negative, so clamp at zero.
    return WidomResult{T}(
        -kT * log(W), kT * semW / W, V * W / kT, V * semW / kT,
        kT - ratio, sqrt(max(var_ratio, zero(T))), sum(n), nb
    )
end
