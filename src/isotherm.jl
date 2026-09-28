# An isotherm is a batch indexed by (framework, pressure, replica): §3 of the design notes that
# framework deduplication (`FrameworkBatch`'s own construction, `batch.jl`) means a multi-pressure
# sweep on one host stores that host once, so building the batch costs one framework's own setup
# regardless of how many pressure points or replicas it carries. `run_isotherm!` is that
# convenience constructor plus the run plus the reduction, all in one call: build the batch, run
# `run_gcmc!` ONCE over the whole thing, then fold each pressure's `nreplicas` chains into one
# `IsothermResult` point with `combine_replicas`.

"""
    combine_replicas(values, errors) -> (mean, combined_err)

Combines `nreplicas` independent chains' own estimate of the same quantity (`values[r]`, each with
its own block-averaged standard error `errors[r]`) into one mean and one combined error:

    within²  = (Σ errors[r]²) / R²
    between² = var(values) / R          -- zero when R == 1
    combined = sqrt(within² + between²)

`within²` is the SEM the R chains' OWN reported errors would give by ordinary independent-error
propagation of their mean — the sampling error a perfectly-equilibrated chain reports, which
understates any spread from finite warmup or seed-dependent slow relaxation. `between²` is the
ordinary SEM of the mean from the replicas' own point estimates, which is itself a noisy estimate
of the true spread when `R` is small and can UNDERSTATE it if the replicas happen to land close by
chance. Adding rather than choosing between them is deliberate: each is a real, independent
source of uncertainty in the pressure-point estimate, and neither alone bounds the other away. At
`R == 1` the `between` term is exactly zero and this reduces to the single chain's own error.
"""
function combine_replicas(values::AbstractVector{F}, errors::AbstractVector{F}) where {F}
    R = length(values)
    R == length(errors) || throw(DimensionMismatch("combine_replicas: values and errors must have the same length"))
    R >= 1 || throw(ArgumentError("combine_replicas: need at least one replica"))
    m = sum(values) / F(R)
    within = sum(abs2, errors) / F(R)^2
    between = R > 1 ? sum(abs2, v - m for v in values) / (F(R - 1) * F(R)) : zero(F)
    return m, sqrt(within + between)
end

"""
    IsothermResult{F}

Loading (and energy) against pressure for one framework/guest/temperature, one entry per pressure
point, built by `run_isotherm!`.

- `pressure` (Pa).
- `loading`/`loading_err` (guests): the mean, over `nreplicas` chains, of each chain's own
  production-average loading, combined with `combine_replicas`.
- `energy`/`energy_err` (eV): the same combination applied to each chain's mean total
  configuration energy.
- `max_occupancy`/`capacity`: the highest occupancy any of this pressure's replicas reached over
  its whole run (warmup and production), against the batch-wide `capacity` every system shares
  (`run_isotherm!`'s own docstring on why capacity is not sized per pressure).
- `nreplicas`: chains averaged into every point.
"""
struct IsothermResult{F}
    pressure::Vector{F}
    loading::Vector{F}
    loading_err::Vector{F}
    energy::Vector{F}
    energy_err::Vector{F}
    max_occupancy::Vector{Int}
    capacity::Vector{Int}
    nreplicas::Int
end

"""
    run_isotherm!(fw::Framework{F}, ff::ForceField{F}, guest::Guest{F,N}, ewald::EwaldParams{F};
                 T, pressures, nreplicas, capacity, n_warmup, n_production, n_audit, step_trans,
                 step_rot, exchange_prob = 0.5, min_cycle_length = 1, seed = 0, nblocks = 10, backend = CPU(),
                 groupsize = DEFAULT_GROUPSIZE, ewald_gg = nothing,
                 nblocks_per_chain = default_nblocks_per_chain(F, length(pressures)*nreplicas)) -> IsothermResult{F}

The convenience constructor task 6 asks for: "this framework, these pressures, this many
replicas". `T` and every entry of `pressures` (Pa) may also be any
`Unitful.Temperature`/`Unitful.Pressure` (`src/units.jl`'s `ustrip_maybe`). Builds ONE batch of
`length(pressures)*nreplicas` systems, system `(p-1)*nreplicas + r` at pressure `pressures[p]`'s
Peng-Robinson fugacity (`peng_robinson_fugacity(pressures[p], T, guest)`), replica `r` — `fw`
repeated `nsys` times
collapses to ONE stored framework (`FrameworkBatch`'s own dedup), so build cost does not grow
with the number of pressure points or replicas, only with the number of DISTINCT frameworks
(here, one); `bench/results/README.md`'s isotherm section has the measured build time. Every
system starts with ZERO guests (`ncounts = 0` throughout, `SystemState`'s own convention for an
empty chain): its per-guest placement work (`initial_poses`) is then a no-op for every system,
which is what keeps the whole batch's build time close to `FrameworkBatch`'s own
single-framework cost instead of growing with `nsys * ncounts` — a nonzero starting count would
need `initial_poses`' hard-core rejection kernel to run once per system, undoing the point.

Runs `run_gcmc!` ONCE over the whole batch — one call, not one per pressure point — and reduces
its per-system `GCMCResult`s into one `IsothermResult` point per pressure, averaging over that
pressure's `nreplicas` chains with `combine_replicas`. `step_trans`/`step_rot` are scalars here
(one step size for the whole isotherm), broadcast to every system; `run_gcmc!` itself still takes
them per-chain.

`capacity` is a SINGLE value shared by every system in the batch, not one per pressure: sizing it
per pressure would need an a priori loading estimate at each point, which this package has no way
to produce before running the chain that would need it. A single capacity sized for the
HIGHEST-pressure point wastes reserved-but-unused slots at the low-pressure end (`SystemState`'s
own per-system `guest_offsets` block) — pure memory, since every energy loop is bounded by LIVE
occupancy, not capacity, so an oversized capacity costs nothing per move. Every point's worst-case
`max_occupancy` (over its replicas) is reported against `capacity` in the result, so a run that
comes close to saturating can be seen rather than silently trusted; `mc_insert!` itself still
throws immediately, independent of this function, if any system actually needed a slot beyond
capacity (`run_gcmc!`'s own docstring).

`ewald_gg`, when given, is forwarded to `FrameworkBatch` (`fullk = false` in that case, since the
two are mutually exclusive there): the guest-guest reciprocal term then gets its own splitting
parameter and smaller k-set instead of riding along in the single `fullk = true` table.
Omitting it (the default) leaves this function's own behavior exactly as it was before this
capability existed.
"""
function run_isotherm!(
        fw::Framework{F}, ff::ForceField{F}, guest::Guest{F, N}, ewald::EwaldParams{F};
        T, pressures::AbstractVector, nreplicas::Integer, capacity::Integer,
        n_warmup::Integer, n_production::Integer, n_audit::Integer, step_trans::Real, step_rot::Real,
        exchange_prob::Real = 0.5, min_cycle_length::Integer = 1, seed::Integer = 0, nblocks::Integer = 10,
        backend = CPU(), groupsize::Integer = DEFAULT_GROUPSIZE, ewald_gg::Union{Nothing, EwaldParams{F}} = nothing,
        nblocks_per_chain::Integer = default_nblocks_per_chain(F, length(pressures) * nreplicas)
    ) where {F, N}
    nreplicas >= 1 || throw(ArgumentError("run_isotherm!: nreplicas=$nreplicas must be >= 1"))
    npress = length(pressures)
    npress >= 1 || throw(ArgumentError("run_isotherm!: pressures must be non-empty"))
    nsys = npress * nreplicas

    Tk = F(ustrip_maybe(u"K", T))
    batch = FrameworkBatch(fill(fw, nsys), ff, guest, ewald; fullk = isnothing(ewald_gg), ewald_gg)
    fugacity = Vector{F}(undef, nsys)
    for p in 1:npress
        f = peng_robinson_fugacity(F(ustrip_maybe(u"Pa", pressures[p])), Tk, guest).f
        for r in 1:nreplicas
            fugacity[(p - 1) * nreplicas + r] = f
        end
    end
    state = SystemState(batch, guest, zeros(Int, nsys), ff; T = Tk, seed, capacities = fill(capacity, nsys))

    results = run_gcmc!(
        batch, state, guest, ff; T = Tk, n_warmup, n_production, n_audit, step_trans = fill(F(step_trans), nsys),
        step_rot = fill(F(step_rot), nsys), fugacity, exchange_prob, min_cycle_length, seed, nblocks, backend,
        groupsize, nblocks_per_chain
    )

    pressure_out = Vector{F}(undef, npress)
    loading = Vector{F}(undef, npress)
    loading_err = Vector{F}(undef, npress)
    energy = Vector{F}(undef, npress)
    energy_err = Vector{F}(undef, npress)
    max_occ = Vector{Int}(undef, npress)
    cap_out = Vector{Int}(undef, npress)
    for p in 1:npress
        rr = ((p - 1) * nreplicas + 1):(p * nreplicas)
        loading[p], loading_err[p] = combine_replicas([results[n].loading for n in rr], [results[n].loading_err for n in rr])
        energy[p], energy_err[p] = combine_replicas([results[n].energy for n in rr], [results[n].energy_err for n in rr])
        pressure_out[p] = F(ustrip_maybe(u"Pa", pressures[p]))
        max_occ[p] = maximum(results[n].max_occupancy for n in rr)
        cap_out[p] = results[rr[1]].capacity
    end
    return IsothermResult{F}(pressure_out, loading, loading_err, energy, energy_err, max_occ, cap_out, Int(nreplicas))
end
