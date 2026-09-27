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

Guests use the same ragged, offset-indexed layout `FrameworkBatch` uses for atoms, but with a
capacity/occupancy split NVT itself never exercises (`capacity(state, n) ==
nguests(state, n)` always, there): `guest_offsets[n]+1:guest_offsets[n+1]` is system `n`'s full
reserved SLOT range in `refpoints`/`orientations`/`host_energy` — its capacity, fixed once at
construction and never resized afterward — while `occupancy[n] <= capacity(state, n)` counts how
many of those slots, starting from the first, currently hold a live guest. `guest_range` (system
`n`'s LIVE range, `guest_offsets[n]+1:guest_offsets[n]+occupancy[n]`) and `nguests` (`occupancy[n]`)
are what every energy computation and move reads; a reserved-but-unoccupied slot holds a fixed
sentinel pose (zero reference point, identity orientation) and a zero cached host energy, and nothing
reads it until an insertion claims it. Insertion writes the new guest at slot
`guest_offsets[n]+occupancy[n]+1` and increments `occupancy[n]`; deletion of an occupied slot
copies the pose and cached host energy of the system's LAST occupied slot into the freed one and
decrements `occupancy[n]`, which keeps the live slots contiguous from the start of the block
without shifting any array (`insert_guest!`, `delete_guest!`). Both fail loudly — `insert_guest!`
throws rather than writing past `capacity(state, n)`, and `audit_energy!` throws if `occupancy[n]`
ever exceeds it — since a chain that silently saturated its capacity would sample a truncated
distribution while still looking healthy. `refpoints` (Å, Cartesian, wrapped into the cell) and
`orientations` (unit quaternions, `rotate`'s `(x, y, z, w)` convention, the same one Milestone A
uses) together give each guest's pose. `insert_guest!`/`delete_guest!` update only these arrays
and `occupancy`; a caller applying a μVT move is responsible for updating `Sk` and `energy` to
match, exactly as `mc_step!`'s kernels already do for the NVT moves. Those kernels, and
`select_and_propose`, currently index a system's guests as `guest_offsets[n]+1:guest_offsets[n+1]`
directly rather than through `guest_range` — correct only as long as `occupancy[n] ==
capacity(state, n)`, which holds for every `SystemState` this package's moves actually run
against today, since nothing yet calls `insert_guest!`/`delete_guest!` on a state a move touches.

`k_offsets` (`kvec_range`) gives each system its OWN slice of `Sk`, one system at a time, so that
`Sk[i] = Shost(k_i) + Σ_guests S_guest(k_i)` for every k-vector that system's framework carries —
`Sk` cannot reuse `batch`'s own `k_offsets`/`Shost` directly when two systems share a framework
(`FrameworkBatch`'s docstring), since a shared host still has independent guests per system, so
`batch_kvec_range` resolves a system's framework's slice of `batch`'s deduplicated tables while
`kvec_range` resolves that system's own slice of `Sk` here. This is correct only when that batch
was built with `fullk = true`: a guest's own structure factor is nonzero at every k, not only the
host-coupled subset the sparse (Milestone A) path keeps, so the constructor requires it.

`energy` is a per-system running total, set at construction to `total_energy(batch, state,
guest, ff, n)` for each system `n` (`src/guest.jl`) and meant to be updated incrementally from
accepted moves thereafter, checked periodically against a from-scratch recomputation
(`audit_energy!`).

`sk_abs_accum` and `energy_abs_accum` are the per-k-vector and per-system running sums, over every
accepted move since `Sk`/`energy` were last set exactly (construction, or the previous
`audit_energy!`), of a per-move rounding scale: `2*Σ|guest.charges|` for `sk_abs_accum` (a proven
upper bound on `abs(ΔS)` — `ΔS = Snew - Sold` with `|Snew|, |Sold| <= Σ|guest.charges|` — that
stays representative when a small move makes `Snew` and `Sold` nearly cancel, unlike `abs(ΔS)`
itself) and `abs(ΔU)` for `energy_abs_accum`. Either way, the scale Higham's recursive-summation
bound actually calls for is the sum of the magnitudes of the terms summed, not the magnitude of
the running total, which heavy cancellation across guests (for `Sk`) or across an equilibrated
chain's fluctuations (for `energy`) can make far smaller than the terms that produced it.
`apply_sk_kernel!`/`decide_move_kernel!` add to these on every acceptance; `audit_energy!` reads
them to size its tolerance and zeroes them once it has re-set `Sk`/`energy` to an exact value.

`host_energy` is a per-guest cache of guest `i`'s own host-guest real-space energy
(`host_guest_realspace_energy`), index-matched to `refpoints`/`orientations`. The host is rigid,
so that energy is a function of guest `i`'s own pose alone and is invalidated only by guest `i`'s
own move: `guest_move_delta` (`src/guest.jl`) reads it instead of recomputing the guest's old
host energy from scratch, halving a move's host scan, and returns the freshly computed new value
for the caller to write back on acceptance — left untouched on rejection, since the pose did not
change. `total_energy` never reads this cache; it recomputes every guest's host energy from
poses, so a `host_energy` entry that falls out of step with the actual poses still shows up as a
running total that no longer matches `total_energy`, caught by `audit_energy!`.

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
struct SystemState{F, VP, VQ, VI, VS, VE, VU, VM, VA, VO}
    guest_offsets::VI
    occupancy::VO
    refpoints::VP
    orientations::VQ
    k_offsets::VI
    Sk::VS
    sk_abs_accum::VA
    energy::VE
    energy_abs_accum::VE
    host_energy::VE
    rng_seed::VU
    rng_counter::VU
    accepted::VM
    attempted::VM
    nsys::Int
end
Adapt.@adapt_structure SystemState

function SystemState(
        guest_offsets, occupancy, refpoints, orientations, k_offsets, Sk, sk_abs_accum, energy, energy_abs_accum,
        host_energy, rng_seed, rng_counter, accepted, attempted, nsys
    )
    F = eltype(energy)
    return SystemState{F}(
        guest_offsets, occupancy, refpoints, orientations, k_offsets, Sk, sk_abs_accum, energy, energy_abs_accum,
        host_energy, rng_seed, rng_counter, accepted, attempted, nsys
    )
end

function SystemState{F}(
        guest_offsets, occupancy, refpoints, orientations, k_offsets, Sk, sk_abs_accum, energy, energy_abs_accum,
        host_energy, rng_seed, rng_counter, accepted, attempted, nsys
    ) where {F}
    return SystemState{
        F, typeof(refpoints), typeof(orientations), typeof(guest_offsets), typeof(Sk), typeof(energy), typeof(rng_seed),
        typeof(accepted), typeof(sk_abs_accum), typeof(occupancy),
    }(
        guest_offsets, occupancy, refpoints, orientations, k_offsets, Sk, sk_abs_accum, energy, energy_abs_accum,
        host_energy, rng_seed, rng_counter, accepted, attempted, Int(nsys)
    )
end

"""
    capacity(state::SystemState, n::Integer) -> Integer

System `n`'s reserved slot count in `refpoints`/`orientations`/`host_energy` — fixed at
construction and never resized, whether or not every slot currently holds a live guest.
"""
capacity(state::SystemState, n::Integer) = state.guest_offsets[n + 1] - state.guest_offsets[n]

"""
    guest_range(state::SystemState, n::Integer) -> UnitRange

System `n`'s LIVE slice of `refpoints`/`orientations`/`host_energy`: its first `nguests(state, n)`
reserved slots, out of `capacity(state, n)` total.
"""
guest_range(state::SystemState, n::Integer) = (state.guest_offsets[n] + 1):(state.guest_offsets[n] + state.occupancy[n])

"""
    kvec_range(state::SystemState, n::Integer) -> UnitRange

System `n`'s slice of `Sk`, matching the `FrameworkBatch` the state was built from.
"""
kvec_range(state::SystemState, n::Integer) = (state.k_offsets[n] + 1):state.k_offsets[n + 1]

"""
    nguests(state::SystemState, n::Integer) -> Integer

Number of LIVE guests in system `n` (its occupancy), out of `capacity(state, n)` reserved slots.
"""
nguests(state::SystemState, n::Integer) = state.occupancy[n]

"""
    insert_guest!(state::SystemState{F}, n::Integer, pos::SVector{3,F}, orient::SVector{4,F},
                  host_energy_new::F = zero(F)) -> Integer

Writes a new guest's pose (and, if given, its host-guest real-space energy) into system `n`'s next
free slot, `guest_offsets[n] + occupancy[n] + 1`, and increments `occupancy[n]` — the slot
immediately after the system's current last occupant, so occupied slots stay contiguous from the
start of the system's reserved block. Returns the global slot index written. Updates only
`refpoints`/`orientations`/`host_energy`/`occupancy`; a caller applying a μVT move is responsible
for updating `Sk` and `energy` to match.

Throws `ArgumentError` if system `n` is already at capacity (`nguests(state, n) ==
capacity(state, n)`).
"""
function insert_guest!(
        state::SystemState{F}, n::Integer, pos::SVector{3, F}, orient::SVector{4, F}, host_energy_new::F = zero(F)
    ) where {F}
    occ = state.occupancy[n]
    cap = capacity(state, n)
    occ < cap || throw(
        ArgumentError("insert_guest!: system $n is already at capacity ($cap); cannot insert another guest")
    )
    slot = state.guest_offsets[n] + occ + 1
    state.refpoints[slot] = pos
    state.orientations[slot] = orient
    state.host_energy[slot] = host_energy_new
    state.occupancy[n] = occ + 1
    return slot
end

"""
    delete_guest!(state::SystemState, n::Integer, slot::Integer) -> Nothing

Removes the guest at global slot `slot`, one of system `n`'s currently occupied slots
(`guest_range(state, n)`): copies the pose and cached host-guest energy of the system's LAST
occupied slot into `slot` (a no-op when `slot` already is that last slot), then decrements
`occupancy[n]`. Occupied slots stay contiguous from the start of the system's reserved block, the
array itself is never shifted, and the multiset of poses among every OTHER occupied guest of
system `n` is unchanged. Updates only `refpoints`/`orientations`/`host_energy`/`occupancy`; a
caller applying a μVT move is responsible for updating `Sk` and `energy` to match.

Throws `ArgumentError` if system `n` has no guests, or if `slot` is not one of its occupied slots.
"""
function delete_guest!(state::SystemState, n::Integer, slot::Integer)
    occ = state.occupancy[n]
    occ > 0 || throw(ArgumentError("delete_guest!: system $n has no guests to delete"))
    lo = state.guest_offsets[n] + 1
    hi = state.guest_offsets[n] + occ
    (lo <= slot <= hi) || throw(
        ArgumentError("delete_guest!: slot $slot is not one of system $n's occupied slots ($lo:$hi)")
    )
    if slot != hi
        state.refpoints[slot] = state.refpoints[hi]
        state.orientations[slot] = state.orientations[hi]
        state.host_energy[slot] = state.host_energy[hi]
    end
    state.occupancy[n] = occ - 1
    return nothing
end

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
                ff::ForceField{F}; T, seed = 0, backend = CPU(),
                capacities = ncounts) -> SystemState

Build a `SystemState` for `batch`'s `nsys` systems, `ncounts[n]` guests in system `n` and
`capacities[n] >= ncounts[n]` reserved guest slots in system `n` (`capacity(state, n)`;
`capacities` defaults to `ncounts` itself, reserving no slack). `guest` must be the same guest (by
value) `batch` was built from, checked the same way `widom` checks it. `batch` must have been
built with `fullk = true` (see `SystemState`'s docstring). `ff` is the force field `batch` was
built from, needed to seed each system's `energy` with its total configuration energy
(`total_energy`, `src/guest.jl`).

Each of the `ncounts[n]` initial guests' pose is drawn uniformly (position in the cell,
orientation on SO(3)) and resampled until it clears `widom`'s hard-core rejection test at
temperature `T` (K) against the host — see `initial_poses`; guest–guest overlap is not checked,
since an overlapping placement only drives `total_energy` to a very large (or infinite) value
here, not an error. `seed` seeds both the placement draws and (via `splitmix64`) each chain's own
RNG stream. `backend` runs the placement's rejection kernel (`CPU()` by default). Any of
`capacities[n] - ncounts[n]` slots reserved beyond that hold a fixed sentinel pose (zero reference
point, identity orientation) and a zero cached host energy until an `insert_guest!` claims one.
"""
function SystemState(
        batch::FrameworkBatch{F}, guest::Guest{F, N}, ncounts::AbstractVector{<:Integer}, ff::ForceField{F};
        T, seed::Integer = 0, backend = CPU(), capacities::AbstractVector{<:Integer} = ncounts
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
    length(capacities) == nsys || throw(
        DimensionMismatch("capacities must have one entry per system (nsys=$nsys), got $(length(capacities))")
    )
    all(>=(0), ncounts) || throw(ArgumentError("ncounts must be non-negative, got $ncounts"))
    all(ncounts .<= capacities) ||
        throw(ArgumentError("capacities must be >= ncounts in every system: ncounts=$ncounts, capacities=$capacities"))

    # `guest_offsets` brackets each system's full reserved CAPACITY block; `occ_offsets` (below,
    # used only to build the initial placement) brackets its initially OCCUPIED guests within that
    # same block, always a prefix of it.
    guest_offsets = Vector{Int32}(undef, nsys + 1)
    guest_offsets[1] = 0
    for n in 1:nsys
        guest_offsets[n + 1] = guest_offsets[n] + Int32(capacities[n])
    end
    ntot = Int(guest_offsets[end])

    occ_offsets = Vector{Int32}(undef, nsys + 1)
    occ_offsets[1] = 0
    for n in 1:nsys
        occ_offsets[n + 1] = occ_offsets[n] + Int32(ncounts[n])
    end
    ntot_occ = Int(occ_offsets[end])
    sys_of = Vector{Int32}(undef, ntot_occ)
    for n in 1:nsys, j in (occ_offsets[n] + 1):occ_offsets[n + 1]
        sys_of[j] = Int32(n)
    end

    kT = F(KB * T)
    rng = Xoshiro(seed)
    rpos, quat_occ = initial_poses(batch, guest, sys_of, kT, rng, backend)

    refpoints = fill(zero(SVector{3, F}), ntot)
    orientations = fill(SVector{4, F}(0, 0, 0, 1), ntot)
    for n in 1:nsys, j in (occ_offsets[n] + 1):occ_offsets[n + 1]
        slot = guest_offsets[n] + (j - occ_offsets[n])
        refpoints[slot] = batch.cells[batch.framework_of[n]] * rpos[j]
        orientations[slot] = quat_occ[j]
    end

    # `state.Sk` cannot reuse `batch`'s own (per-FRAMEWORK, deduplicated) `Shost`/`k_offsets`
    # directly: `Sk` also carries each system's own guests, so it needs one slice per SYSTEM even
    # when two systems share a framework. Each system's slice starts as a fresh copy of its own
    # framework's `Shost` (`batch_kvec_range`), sized by that framework's own k-vector count —
    # the same total layout `copy(batch.k_offsets)`/`copy(batch.Shost)` gave before frameworks
    # were deduplicated, just no longer literally aliasing `batch`'s storage.
    k_offsets = Vector{Int32}(undef, nsys + 1)
    k_offsets[1] = 0
    for n in 1:nsys
        k_offsets[n + 1] = k_offsets[n] + Int32(length(batch_kvec_range(batch, n)))
    end
    occupancy = Vector{Int32}(Int32.(ncounts))
    Sk = Vector{Complex{F}}(undef, k_offsets[end])
    for n in 1:nsys
        kr = (k_offsets[n] + 1):k_offsets[n + 1]
        krb = batch_kvec_range(batch, n)
        Sk[kr] .= view(batch.Shost, krb)
        gr = (guest_offsets[n] + 1):(guest_offsets[n] + occupancy[n])
        (isempty(kr) || isempty(gr)) && continue
        sitepos, siteq = guest_site_positions_charges(guest, refpoints, orientations, gr)
        Sk[kr] .+= structure_factor(view(batch.ks, krb), sitepos, siteq)
    end

    guest_types_c = SVector{N, Int}(batch.guest_types)
    guest_compact = Guest{F, N}(guest.sites, guest_types_c, guest.charges, guest.tc, guest.pc, guest.omega)
    host_energy = zeros(F, ntot)
    for n in 1:nsys
        fw = batch.framework_of[n]
        a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
        A = batch.cells[fw]; invA = batch.invcells[fw]; alpha = batch.alphas[fw]
        for i in (guest_offsets[n] + 1):(guest_offsets[n] + occupancy[n])
            host_energy[i] = host_guest_realspace_energy(
                refpoints[i], orientations[i], guest_compact, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
                batch.positions, batch.types, batch.charges, a0, natoms, A, invA, alpha
            )
        end
    end

    sk_abs_accum = zeros(F, length(Sk))
    energy = zeros(F, nsys)
    energy_abs_accum = zeros(F, nsys)
    rng_seed = [splitmix64(UInt64(seed), UInt64(n)) for n in 1:nsys]
    rng_counter = zeros(UInt64, nsys)
    accepted = fill(zero(SVector{NMOVETYPES, Int32}), nsys)
    attempted = fill(zero(SVector{NMOVETYPES, Int32}), nsys)
    st = SystemState(
        guest_offsets, occupancy, refpoints, orientations, k_offsets, Sk, sk_abs_accum, energy, energy_abs_accum,
        host_energy, rng_seed, rng_counter, accepted, attempted, nsys
    )
    for n in 1:nsys
        st.energy[n] = total_energy(batch, st, guest, ff, n)
    end
    return st
end
