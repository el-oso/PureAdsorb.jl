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

# System assigned to global insertion index `g` (position in `1:ninsert`, independent of chunk
# boundaries) under the run-length assignment: `run` consecutive values of `g` share a system
# before the assignment cycles to the next one, so device work-items adjacent in `g` read the
# same framework's tables.
sys_of_index(g::Integer, run::Integer, nsys::Integer) = Int32(mod1((g - 1) ÷ run + 1, nsys))

# `run` length `widom` uses when the caller does not override it: a quarter of one system's
# average share of the insertions, clamped so a small batch still gets `run >= 1` and a huge one
# does not let a single system dominate an entire chunk.
default_run(ninsert::Integer, nsys::Integer) = clamp((ninsert ÷ nsys) ÷ 4, 1, 256)

# Exact per-system insertion count under the run-length assignment: `run`-length runs cycle
# round-robin through systems `1:nsys`, and any insertions left over after the last full run are
# credited to the system holding that final, shorter run.
function system_counts(ninsert::Integer, nsys::Integer, run::Integer)
    q, rem = divrem(ninsert, run)
    full, hi = divrem(q, nsys)
    ns = fill(full * run, nsys)
    for r in 0:(hi - 1)
        ns[r + 1] += run
    end
    rem > 0 && (ns[mod(q, nsys) + 1] += rem)
    return ns
end

# Shoemake (1992) uniform random unit quaternion, in (x, y, z, w) order per the comment block
# above `rotate`: (√(1-u₁) cos 2πu₂, √u₁ sin 2πu₃, √u₁ cos 2πu₃, √(1-u₁) sin 2πu₂).
function shoemake_quaternion(rng::AbstractRNG, ::Type{T}) where {T}
    u1, u2, u3 = rand(rng), rand(rng), rand(rng)
    a, b = sqrt(1 - u1), sqrt(u1)
    return SVector{4, T}(a * cospi(2u2), b * sinpi(2u3), b * cospi(2u3), a * sinpi(2u2))
end

# Poses for global insertion indices `first_g:(first_g + length(sys_of) - 1)`.
function random_poses!(rng::AbstractRNG, sys_of, rpos, quat, first_g::Integer, run::Integer, nsys::Integer)
    for i in eachindex(sys_of, rpos, quat)
        sys_of[i] = sys_of_index(first_g + i - 1, run, nsys)
        rpos[i] = rand(rng, eltype(rpos))
        quat[i] = shoemake_quaternion(rng, eltype(eltype(quat)))
    end
    return nothing
end

@kernel function widom_kernel!(ΔU, @Const(sys_of), @Const(rpos), @Const(quat), batch, guest)
    i = @index(Global)
    s = sys_of[i]
    a0 = batch.atom_offsets[s]
    g0 = batch.cellgrid_offsets[s] + 1
    g1 = batch.cellgrid_offsets[s + 1]
    k0 = batch.k_offsets[s] + 1
    k1 = batch.k_offsets[s + 1]
    A = batch.cells[s]
    invA = batch.invcells[s]
    pos = A * rpos[i]
    e = insertion_energy(
        pos, quat[i], guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, a0, batch.ncells[s], batch.reach[s],
        view(batch.cell_offsets, g0:g1),
        A, invA, batch.alphas[s], view(batch.ks, k0:k1), view(batch.kprefactor, k0:k1), view(batch.Shost, k0:k1)
    )
    ΔU[i] = e + batch.constant_offset[s]
end

"""
    widom(batch::FrameworkBatch, guest::Guest; T, ninsert, backend = CPU(), seed = 0,
          chunk = 2^16, nblocks = 10, run = nothing) -> Vector{WidomResult}

Widom test-particle insertion at temperature `T` (K): `ninsert` random poses per system give
the Boltzmann-weighted insertion average `W = ⟨exp(-ΔU/kT)⟩`, split into `nblocks` blocks for a
standard-error estimate, and reduced to a `WidomResult` per system in `batch`. Insertions are
generated and evaluated in chunks of `chunk` poses per `backend` kernel launch. `seed` sets the
random-pose generator.

Insertion `g` of `1:ninsert` goes to system `mod1((g - 1) ÷ run + 1, nsys)`: `run` consecutive
insertions share a system before the assignment cycles to the next one, so device work-items
that are adjacent in `g` read the same framework's tables. `run` defaults to
`clamp((ninsert ÷ nsys) ÷ 4, 1, 256)` and can be overridden, subject to
`1 <= run <= ninsert ÷ nsys`. Each system's block statistics are drawn from its own sample
order, so one system's block boundaries do not depend on how many insertions any other system
receives.
"""
function widom(
        batch::FrameworkBatch{F}, guest::Guest{F}; T, ninsert::Integer, backend = CPU(), seed = 0,
        chunk::Integer = 2^16, nblocks::Integer = 10, run::Union{Nothing, Integer} = nothing
    ) where {F}
    nblocks >= 2 || throw(ArgumentError("nblocks=$nblocks: at least two blocks are needed for a standard error"))
    chunk >= 1 || throw(ArgumentError("chunk=$chunk must be ≥ 1"))
    nsys = batch.nsys
    per = ninsert ÷ nsys
    runlen = if isnothing(run)
        default_run(ninsert, nsys)
    else
        run >= 1 || throw(ArgumentError("run=$run must be ≥ 1"))
        run <= per || throw(ArgumentError("run=$run must be ≤ per=$per (= ninsert ÷ nsys)"))
        Int(run)
    end
    ns = system_counts(ninsert, nsys, runlen)
    minimum(ns) >= 2 * nblocks || throw(
        ArgumentError(
            "ninsert=$ninsert, nsys=$nsys, run=$runlen: system $(argmin(ns)) gets only $(minimum(ns)) samples, " *
                "need at least 2·nblocks = $(2 * nblocks) per system"
        )
    )
    axes(batch.positions) == axes(batch.types) == axes(batch.charges) || throw(
        DimensionMismatch(
            "batch positions/types/charges must share axes: $(axes(batch.positions)) vs $(axes(batch.types)) vs $(axes(batch.charges))"
        )
    )
    axes(batch.ks) == axes(batch.kprefactor) == axes(batch.Shost) || throw(
        DimensionMismatch(
            "batch ks/kprefactor/Shost must share axes: $(axes(batch.ks)) vs $(axes(batch.kprefactor)) vs $(axes(batch.Shost))"
        )
    )
    length(batch.ncells) == length(batch.reach) == nsys || throw(
        DimensionMismatch(
            "batch ncells/reach must have one entry per system (nsys=$nsys): " *
                "$(length(batch.ncells)) / $(length(batch.reach))"
        )
    )
    length(batch.cellgrid_offsets) == nsys + 1 || throw(
        DimensionMismatch(
            "batch cellgrid_offsets must have nsys+1=$(nsys + 1) entries, got $(length(batch.cellgrid_offsets))"
        )
    )
    batch.cellgrid_offsets[end] == length(batch.cell_offsets) || throw(
        DimensionMismatch(
            "batch cellgrid_offsets[end]=$(batch.cellgrid_offsets[end]) must equal " *
                "length(cell_offsets)=$(length(batch.cell_offsets))"
        )
    )
    for s in 1:nsys
        g1 = batch.cellgrid_offsets[s + 1]
        natoms_s = batch.atom_offsets[s + 1] - batch.atom_offsets[s]
        batch.cell_offsets[g1] == natoms_s || throw(
            DimensionMismatch(
                "system $s: last cell_offsets entry $(batch.cell_offsets[g1]) must equal its atom count $natoms_s"
            )
        )
    end
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
    # Per-system count of insertions already accumulated: each system's blocks are drawn from
    # its own sample order, not from the interleaved global insertion index.
    seen = zeros(Int, nsys)
    # Floor division so the clamp below absorbs the remainder into the last block instead of
    # leaving it with zero samples (the guard above guarantees ns[s] >= 2·nblocks for every s).
    block_len = ns .÷ nblocks
    done = 0
    while done < ninsert
        m = min(chunk, ninsert - done)
        random_poses!(rng, view(sys_of, 1:m), view(rpos, 1:m), view(quat, 1:m), done + 1, runlen, nsys)
        copyto!(dsys, sys_of)
        copyto!(drpos, rpos)
        copyto!(dquat, quat)
        kern(dΔU, dsys, drpos, dquat, dbatch, guest; ndrange = m)
        KernelAbstractions.synchronize(backend)
        copyto!(ΔU_h, dΔU)
        for i in 1:m
            s = sys_of[i]
            seen[s] += 1
            blk = min(nblocks, (seen[s] - 1) ÷ block_len[s] + 1)
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
