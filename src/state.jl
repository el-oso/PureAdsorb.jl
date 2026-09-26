# The three NVT moves this milestone's chain applies (translation, rotation, reinsertion); NVT
# excludes exchange (`exchange_prob = 0`), so counters need no fourth slot.
const NMOVETYPES = 3

"""
    SystemState{F}

Mutable per-system NVT Monte Carlo state for the `nsys` independent chains of a
`FrameworkBatch{F}`: guest poses, the running reciprocal-space structure factor, a running
total energy, one RNG stream per chain, and per-move-type accepted/attempted counters.
`FrameworkBatch` itself stays immutable and holds only the host data, so one batch can be
reused to build several independent `SystemState`s.

Guests use the same ragged, offset-indexed layout `FrameworkBatch` uses for atoms:
`guest_offsets[n]+1:guest_offsets[n+1]` (`guest_range`) indexes system `n`'s slice of
`refpoints`/`orientations`, and NVT holds each system's slice length fixed after construction
even though nothing here assumes systems share a count — Milestone C varies it. `refpoints`
(Å, Cartesian, wrapped into the cell) and `orientations` (unit quaternions, `rotate`'s
`(x, y, z, w)` convention, the same one Milestone A uses) together give each guest's pose.

`k_offsets` (`kvec_range`) copies the layout of the `FrameworkBatch` the state was built from,
so `Sk[i] = Shost(k_i) + Σ_guests S_guest(k_i)` for every k-vector that batch carries. This is
correct only when that batch was built with `fullk = true`: a guest's own structure factor is
nonzero at every k, not only the host-coupled subset the sparse (Milestone A) path keeps, so the
constructor requires it.

`energy` is a per-system running total, set at construction to `total_energy(batch, state,
guest, ff, n)` for each system `n` (`src/guest.jl`) and meant to be updated incrementally from
accepted moves thereafter, checked periodically against a from-scratch recomputation
(`audit_energy!`).

`rng_seed`/`rng_counter` are a per-chain seed (mixed from the constructor's `seed` and the
chain's system index via `splitmix64`, so distinct chains from one `seed` get distinct,
reproducible values) and a per-chain draw counter, starting at zero: together they are the
inputs a counter-based per-chain RNG keys on (seed, chain, counter), matching this project's
plan to make reproducibility independent of batch size and chunking by construction rather than
relying on the counter-based generator itself to decorrelate a shared key. The actual keyed
generator is a later task; this only allocates and initializes its inputs.

`accepted`/`attempted` are per-system `SVector{$NMOVETYPES, Int32}` counts, ordered
(translation, rotation, reinsertion) — the three NVT moves (`exchange_prob = 0` excludes the
fourth) — both starting at zero.
"""
struct SystemState{F, VP, VQ, VI, VS, VE, VU, VM}
    guest_offsets::VI
    refpoints::VP
    orientations::VQ
    k_offsets::VI
    Sk::VS
    energy::VE
    rng_seed::VU
    rng_counter::VU
    accepted::VM
    attempted::VM
    nsys::Int
end
Adapt.@adapt_structure SystemState

function SystemState(
        guest_offsets, refpoints, orientations, k_offsets, Sk, energy, rng_seed, rng_counter, accepted, attempted,
        nsys
    )
    F = eltype(energy)
    return SystemState{F}(guest_offsets, refpoints, orientations, k_offsets, Sk, energy, rng_seed, rng_counter, accepted, attempted, nsys)
end

function SystemState{F}(
        guest_offsets, refpoints, orientations, k_offsets, Sk, energy, rng_seed, rng_counter, accepted, attempted,
        nsys
    ) where {F}
    return SystemState{
        F, typeof(refpoints), typeof(orientations), typeof(guest_offsets), typeof(Sk), typeof(energy), typeof(rng_seed), typeof(accepted),
    }(
        guest_offsets, refpoints, orientations, k_offsets, Sk, energy, rng_seed, rng_counter, accepted, attempted, Int(nsys)
    )
end

"""
    guest_range(state::SystemState, n::Integer) -> UnitRange

System `n`'s slice of `refpoints`/`orientations`.
"""
guest_range(state::SystemState, n::Integer) = (state.guest_offsets[n] + 1):state.guest_offsets[n + 1]

"""
    kvec_range(state::SystemState, n::Integer) -> UnitRange

System `n`'s slice of `Sk`, matching the `FrameworkBatch` the state was built from.
"""
kvec_range(state::SystemState, n::Integer) = (state.k_offsets[n] + 1):state.k_offsets[n + 1]

"""
    nguests(state::SystemState, n::Integer) -> Integer

Number of guests in system `n`.
"""
nguests(state::SystemState, n::Integer) = state.guest_offsets[n + 1] - state.guest_offsets[n]

# SplitMix64's finalizer (Steele, Lea & Flood 2014), one round: a fast, deterministic bijective
# mix from a (seed, index) pair to a per-chain seed, so distinct chain indices built from the same
# `seed` get distinct, reproducible values with no correlation an adversarial index could exploit.
function splitmix64(seed::UInt64, index::UInt64)
    z = seed + index * 0x9E3779B97F4A7C15
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

# Placement-loop cap: with a valid rejection radius, redrawing only the still-overlapping slots
# converges within a handful of rounds; this turns a guest count that cannot physically fit into
# a fail-fast error instead of an infinite loop.
const MAX_PLACEMENT_ROUNDS = 10_000

# Draws fractional-in-[0,1) poses and unit quaternions for `ntot` guest slots (`sys_of[i]` the
# system slot `i` belongs to) that clear `widom`'s own hard-core rejection test at temperature
# `kT` against the host: reuses `build_rejection_tables` and `hardcore_kernel!` exactly as
# `widom` does, so "does not overlap the host" means the same thing it means there — provably
# negligible Boltzmann weight at `kT`, not a literal atomic clash. Like `widom`'s own rejection
# stage, this checks each candidate against the host only: guests placed earlier are not checked
# against guests placed later, so the returned poses can still clash with each other.
function initial_poses(
        batch::FrameworkBatch{F}, guest::Guest{F, N}, sys_of::Vector{Int32}, kT::F, rng::AbstractRNG, backend
    ) where {F, N}
    ntot = length(sys_of)
    rpos = Vector{SVector{3, F}}(undef, ntot)
    quat = Vector{SVector{4, F}}(undef, ntot)
    iszero(ntot) && return rpos, quat
    for i in eachindex(rpos, quat)
        rpos[i] = rand(rng, SVector{3, F})
        quat[i] = shoemake_quaternion(rng, F)
    end
    guest_compact = Guest{F, N}(guest.sites, SVector{N, Int}(batch.guest_types), guest.charges, guest.tc, guest.pc, guest.omega)
    rho2, reach0, ntypes = build_rejection_tables(batch, guest_compact, kT)
    dbatch = adapt(backend, batch)
    drho2, dreach0 = adapt(backend, rho2), adapt(backend, reach0)
    pending = collect(Int32, 1:ntot)
    kern = hardcore_kernel!(backend)
    rounds = 0
    while !isempty(pending)
        rounds += 1
        rounds <= MAX_PLACEMENT_ROUNDS || throw(
            ArgumentError(
                "initial placement did not clear the host after $MAX_PLACEMENT_ROUNDS rounds " *
                    "($(length(pending)) of $ntot guest slots still overlap at kT=$kT)"
            )
        )
        m = length(pending)
        dpsys = adapt(backend, sys_of[pending])
        drpos = adapt(backend, rpos[pending])
        dquat = adapt(backend, quat[pending])
        dflags = adapt(backend, zeros(UInt8, m))
        kern(dflags, dpsys, drpos, dquat, dbatch, guest_compact, drho2, dreach0, Int32(ntypes); ndrange = m)
        KernelAbstractions.synchronize(backend)
        flags = Array(dflags)
        still = Int32[]
        for (j, i) in enumerate(pending)
            if !iszero(flags[j])
                rpos[i] = rand(rng, SVector{3, F})
                quat[i] = shoemake_quaternion(rng, F)
                push!(still, i)
            end
        end
        pending = still
    end
    return rpos, quat
end

# Cartesian positions and charges of every site of every guest in `gr` (a `guest_range`), for
# feeding to `structure_factor`.
function guest_site_positions_charges(guest::Guest{F, N}, refpoints, orientations, gr) where {F, N}
    pos = Vector{SVector{3, F}}(undef, length(gr) * N)
    q = Vector{F}(undef, length(gr) * N)
    idx = 0
    for i in gr, s in 1:N
        idx += 1
        pos[idx] = refpoints[i] + rotate(orientations[i], guest.sites[s])
        q[idx] = guest.charges[s]
    end
    return pos, q
end

"""
    SystemState(batch::FrameworkBatch{F}, guest::Guest{F,N}, ncounts::AbstractVector{<:Integer},
                ff::ForceField{F}; T, seed = 0, backend = CPU()) -> SystemState

Build a `SystemState` for `batch`'s `nsys` systems, `ncounts[n]` guests in system `n`. `guest`
must be the same guest (by value) `batch` was built from, checked the same way `widom` checks
it. `batch` must have been built with `fullk = true` (see `SystemState`'s docstring). `ff` is
the force field `batch` was built from, needed to seed each system's `energy` with its total
configuration energy (`total_energy`, `src/guest.jl`).

Each guest's initial pose is drawn uniformly (position in the cell, orientation on SO(3)) and
resampled until it clears `widom`'s hard-core rejection test at temperature `T` (K) against the
host — see `initial_poses`; guest–guest overlap is not checked, since an overlapping placement
only drives `total_energy` to a very large (or infinite) value here, not an error. `seed` seeds both the
placement draws and (via `splitmix64`) each chain's own RNG stream. `backend` runs the
placement's rejection kernel (`CPU()` by default).
"""
function SystemState(
        batch::FrameworkBatch{F}, guest::Guest{F, N}, ncounts::AbstractVector{<:Integer}, ff::ForceField{F};
        T, seed::Integer = 0, backend = CPU()
    ) where {F, N}
    (
        guest.types == batch.guest_types_orig && guest.sites == batch.guest_sites_orig &&
            guest.charges == batch.guest_charges_orig
    ) || throw(
        ArgumentError(
            "guest passed to SystemState does not match the guest FrameworkBatch was built with: " *
                "passed types=$(guest.types), sites=$(guest.sites), charges=$(guest.charges); " *
                "batch types=$(batch.guest_types_orig), sites=$(batch.guest_sites_orig), charges=$(batch.guest_charges_orig)"
        )
    )
    batch.fullk || throw(
        ArgumentError(
            "SystemState needs a FrameworkBatch built with fullk=true: a guest's structure factor " *
                "is nonzero at every k-vector, not only the host-coupled subset the sparse " *
                "(Milestone A) path keeps"
        )
    )
    nsys = batch.nsys
    length(ncounts) == nsys || throw(
        DimensionMismatch("ncounts must have one entry per system (nsys=$nsys), got $(length(ncounts))")
    )
    all(>=(0), ncounts) || throw(ArgumentError("ncounts must be non-negative, got $ncounts"))

    guest_offsets = Vector{Int32}(undef, nsys + 1)
    guest_offsets[1] = 0
    for n in 1:nsys
        guest_offsets[n + 1] = guest_offsets[n] + Int32(ncounts[n])
    end
    ntot = Int(guest_offsets[end])
    sys_of = Vector{Int32}(undef, ntot)
    for n in 1:nsys, i in (guest_offsets[n] + 1):guest_offsets[n + 1]
        sys_of[i] = Int32(n)
    end

    kT = F(KB * T)
    rng = Xoshiro(seed)
    rpos, quat = initial_poses(batch, guest, sys_of, kT, rng, backend)
    refpoints = Vector{SVector{3, F}}(undef, ntot)
    for i in 1:ntot
        refpoints[i] = batch.cells[sys_of[i]] * rpos[i]
    end

    k_offsets = copy(batch.k_offsets)
    Sk = copy(batch.Shost)
    for n in 1:nsys
        kr = (k_offsets[n] + 1):k_offsets[n + 1]
        gr = (guest_offsets[n] + 1):guest_offsets[n + 1]
        (isempty(kr) || isempty(gr)) && continue
        sitepos, siteq = guest_site_positions_charges(guest, refpoints, quat, gr)
        Sk[kr] .+= structure_factor(view(batch.ks, kr), sitepos, siteq)
    end

    energy = zeros(F, nsys)
    rng_seed = [splitmix64(UInt64(seed), UInt64(n)) for n in 1:nsys]
    rng_counter = zeros(UInt64, nsys)
    accepted = fill(zero(SVector{NMOVETYPES, Int32}), nsys)
    attempted = fill(zero(SVector{NMOVETYPES, Int32}), nsys)
    st = SystemState(guest_offsets, refpoints, quat, k_offsets, Sk, energy, rng_seed, rng_counter, accepted, attempted, nsys)
    for n in 1:nsys
        st.energy[n] = total_energy(batch, st, guest, ff, n)
    end
    return st
end
