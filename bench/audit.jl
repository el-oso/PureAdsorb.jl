# StrictMode gate for PureAdsorb's hot kernels.
#
# STRICT_MODE=fast (default): StrictMode's value-free scan (no analysis backend) — cheap,
# reports structural allocation/type-stability guesses.
# STRICT_MODE=full: StrictModeTest's proof tier, backed by AllocCheck and JET — the rigorous
# gate. TrimCheck is loaded alongside so a `:trimsafe` guarantee would also be provable, though
# it is not requested below.
#
# Run:
#   julia --project=bench bench/audit.jl                    # fast, from any checkout
#   STRICT_MODE=full julia --project=bench bench/audit.jl    # full, the pre-merge gate
using PureAdsorb, StrictMode, StaticArrays
using StrictModeTest, JET, AllocCheck, TrimCheck   # declared bench deps; assert_enabled requires them loaded

StrictMode.assert_enabled()

const MODE = get(ENV, "STRICT_MODE", "fast") == "full" ? :full : :fast

signature_findings(f, types; guarantees) = MODE === :full ?
    StrictModeTest.proof_findings(f, types; guarantees) : StrictMode.findings(f, types; guarantees)

insertion_energy_types(::Type{T}) where {T} = (
    SVector{3, T}, SVector{4, T}, PureAdsorb.Guest{T, 3}, Matrix{T}, Matrix{T}, T, T,
    Vector{SVector{3, T}}, Vector{Int32}, Vector{T}, Int, Int,
    SMatrix{3, 3, T, 9}, SMatrix{3, 3, T, 9}, T, Vector{SVector{3, T}}, Vector{T}, Vector{Complex{T}},
)
const M3 = SMatrix{3, 3, Float64, 9}

# `host_guest_realspace_energy` is `insertion_energy`'s own real-space loop with the reciprocal
# cross term omitted, so its argument types are `insertion_energy_types`'s first 15 (through
# `alpha`; the trailing `ks`/`kprefactor`/`Shost` are absent).
host_guest_realspace_energy_types(::Type{T}) where {T} = insertion_energy_types(T)[1:15]

sm3(::Type{T}) where {T} = SMatrix{3, 3, T, 9}
guest_pair_realspace_energy_types(::Type{T}) where {T} = (
    SVector{3, SVector{3, T}}, SVector{3, SVector{3, T}}, SVector{3, Int}, SVector{3, T},
    Matrix{T}, Matrix{T}, T, T, sm3(T), sm3(T), T,
)
guest_guest_energy_types(::Type{T}) where {T} = (
    PureAdsorb.Guest{T, 3}, Vector{SVector{3, T}}, Vector{SVector{4, T}}, UnitRange{Int}, Matrix{T}, Matrix{T},
    SVector{3, Int}, T, T, sm3(T), sm3(T), T,
)
guest_guest_move_delta_types(::Type{T}) where {T} = (
    PureAdsorb.Guest{T, 3}, Vector{SVector{3, T}}, Vector{SVector{4, T}}, UnitRange{Int}, Int,
    SVector{3, T}, SVector{4, T}, SVector{3, T}, SVector{4, T}, Matrix{T}, Matrix{T}, SVector{3, Int}, T, T, sm3(T), sm3(T), T,
)
guest_pair_realspace_energy_range_types(::Type{T}) where {T} = (
    PureAdsorb.Guest{T, 3}, SVector{3, T}, SVector{4, T}, Vector{SVector{3, T}}, Vector{SVector{4, T}}, UnitRange{Int}, Int32,
    Matrix{T}, Matrix{T}, SVector{3, Int}, T, T, sm3(T), sm3(T), T,
)
reciprocal_move_delta_energy_types(::Type{T}) where {T} = (
    PureAdsorb.Guest{T, 3}, SVector{3, T}, SVector{4, T}, SVector{3, T}, SVector{4, T},
    Vector{SVector{3, T}}, Vector{T}, Vector{Complex{T}},
)
reciprocal_move_delta_bang_types(::Type{T}) where {T} = (Vector{Complex{T}}, reciprocal_move_delta_energy_types(T)...)
guest_sites_at_types(::Type{T}) where {T} = (PureAdsorb.Guest{T, 3}, SVector{3, T}, SVector{4, T})
total_reciprocal_energy_types(::Type{T}) where {T} = (Vector{T}, Vector{Complex{T}}, Vector{Complex{T}})

# Task 5's moves (`src/moves.jl`): the proposal/acceptance primitives `move_kernel!` calls, both
# precisions. The kernels themselves are not audited here (a `@kernel function`'s calling
# convention is not a plain function signature `StrictMode.findings` can probe).
propose_move_types(::Type{T}) where {T} = (
    PureAdsorb.ChainRNG, Int32, SVector{3, T}, SVector{4, T}, T, T, SMatrix{3, 3, T, 9},
)
select_and_propose_types(::Type{T}) where {T} = (
    Int, Vector{Int32}, Vector{Int32}, Vector{UInt64}, Vector{UInt64}, Int32, Vector{T}, Vector{T}, SMatrix{3, 3, T, 9},
    Vector{SVector{3, T}}, Vector{SVector{4, T}}, Int,
)

# `SystemState`'s field types for a concrete precision, matching exactly what `SystemState`'s
# own constructor produces (`src/state.jl`), for auditing its accessors without building a real
# batch and state just to call `typeof` on them.
systemstate_type(::Type{T}) where {T} = PureAdsorb.SystemState{
    T, Vector{SVector{3, T}}, Vector{SVector{4, T}}, Vector{Int32}, Vector{Complex{T}}, Vector{T}, Vector{UInt64}, Vector{SVector{3, Int32}},
    Vector{T}, Vector{Int32},
}

# `SystemState`'s accessors, which index into device arrays inside future kernels, and its
# per-chain seed mix.
results = vcat(
    signature_findings(PureAdsorb.guest_range, (systemstate_type(Float64), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_range, (systemstate_type(Float32), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.kvec_range, (systemstate_type(Float64), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.kvec_range, (systemstate_type(Float32), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.nguests, (systemstate_type(Float64), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.nguests, (systemstate_type(Float32), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.capacity, (systemstate_type(Float64), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.capacity, (systemstate_type(Float32), Int); guarantees = (:typestable, :noalloc)),
    signature_findings(
        PureAdsorb.insert_guest!, (systemstate_type(Float64), Int, SVector{3, Float64}, SVector{4, Float64}, Float64);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.insert_guest!, (systemstate_type(Float32), Int, SVector{3, Float32}, SVector{4, Float32}, Float32);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(PureAdsorb.delete_guest!, (systemstate_type(Float64), Int, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.delete_guest!, (systemstate_type(Float32), Int, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.splitmix64, (UInt64, UInt64); guarantees = (:typestable, :noalloc)),
    # Phase-0 kernel (`hardcore_kernel!`) callees not already covered by `insertion_energy`'s own
    # (`rotate`, `minimum_image`): the cell-list stencil and per-site home-cell primitives that
    # only phase 0 uses, since `insertion_energy` (phase 1) loops linearly over the system's atoms.
    signature_findings(PureAdsorb.insertion_energy, insertion_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.insertion_energy, insertion_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.minimum_image, (M3, M3, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rotate, (SVector{4, Float64}, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.erfc_dev, (Float64,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.pair_erfc_dev, (Float64,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.pair_erfc_dev, (Float32,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.home_cell_dev, (Float64, Int32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.home_cell_dev, (Float32, Int32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.wrap_frac, (Float64,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.wrap_frac, (Float32,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.stencil_start_count, (Int32, Int32, Int32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.wrap_cell, (Int32, Int32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.cell_linear, (Int32, Int32, Int32, Int32, Int32); guarantees = (:typestable, :noalloc)),
    # Milestone B's per-move guest-guest/guest-host kernels (`src/guest.jl`): every function a
    # move kernel calls, both precisions.
    signature_findings(PureAdsorb.host_guest_realspace_energy, host_guest_realspace_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.host_guest_realspace_energy, host_guest_realspace_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_pair_realspace_energy, guest_pair_realspace_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_pair_realspace_energy, guest_pair_realspace_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_guest_energy, guest_guest_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_guest_energy, guest_guest_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_guest_move_delta, guest_guest_move_delta_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_guest_move_delta, guest_guest_move_delta_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_move_delta_energy, reciprocal_move_delta_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_move_delta_energy, reciprocal_move_delta_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(
        PureAdsorb._reciprocal_move_delta_k, (SVector{3, Float64}, SVector{3, Float64}, SVector{3, SVector{3, Float64}}, SVector{3, SVector{3, Float64}}, ComplexF64);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb._reciprocal_move_delta_k, (SVector{3, Float32}, SVector{3, Float32}, SVector{3, SVector{3, Float32}}, SVector{3, SVector{3, Float32}}, ComplexF32);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(PureAdsorb.reciprocal_move_delta!, reciprocal_move_delta_bang_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_move_delta!, reciprocal_move_delta_bang_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_sites_at, guest_sites_at_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_sites_at, guest_sites_at_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.total_reciprocal_energy, total_reciprocal_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.total_reciprocal_energy, total_reciprocal_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    # The guest-guest Ewald split (`has_ewald_split`, `src/guest.jl`): the cross-only and
    # self-only reciprocal-energy formulas each of `ks`/`ks_gg` uses.
    signature_findings(PureAdsorb.cross_reciprocal_energy, total_reciprocal_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.cross_reciprocal_energy, total_reciprocal_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.self_reciprocal_energy, (Vector{Float64}, Vector{ComplexF64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.self_reciprocal_energy, (Vector{Float32}, Vector{ComplexF32}); guarantees = (:typestable, :noalloc)),
    signature_findings(
        PureAdsorb._cross_move_delta_k, (SVector{3, Float64}, SVector{3, Float64}, SVector{3, SVector{3, Float64}}, SVector{3, SVector{3, Float64}}, ComplexF64);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb._cross_move_delta_k, (SVector{3, Float32}, SVector{3, Float32}, SVector{3, SVector{3, Float32}}, SVector{3, SVector{3, Float32}}, ComplexF32);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(PureAdsorb.reciprocal_cross_delta!, reciprocal_move_delta_bang_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_cross_delta!, reciprocal_move_delta_bang_types(Float32); guarantees = (:typestable, :noalloc)),
    # `reciprocal_cross_delta_energy`/`reciprocal_exchange_cross_energy` (`moves.jl`'s split-aware
    # production kernels): `reciprocal_cross_delta!` and `reciprocal_exchange_energy`'s own
    # cross-only counterparts, without writing `ΔS`.
    signature_findings(PureAdsorb.reciprocal_cross_delta_energy, reciprocal_move_delta_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_cross_delta_energy, reciprocal_move_delta_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(
        PureAdsorb.reciprocal_exchange_cross_energy,
        (PureAdsorb.Guest{Float64, 3}, SVector{3, SVector{3, Float64}}, SVector{3, SVector{3, Float64}}, Vector{SVector{3, Float64}}, Vector{Float64}, Vector{ComplexF64});
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.reciprocal_exchange_cross_energy,
        (PureAdsorb.Guest{Float32, 3}, SVector{3, SVector{3, Float32}}, SVector{3, SVector{3, Float32}}, Vector{SVector{3, Float32}}, Vector{Float32}, Vector{ComplexF32});
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(PureAdsorb.energy_audit_tolerance, (Float64, Float64, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.energy_audit_tolerance, (Float32, Float32, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.sk_audit_tolerance, (Float64, Float64, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.sk_audit_tolerance, (Float32, Float32, Int); guarantees = (:typestable, :noalloc)),
    # Device-side RNG (`src/rng.jl`): the counter-based generator every future move kernel draws
    # from, both precisions and both integer widths its callers use.
    signature_findings(PureAdsorb.mulhi64, (UInt64, UInt64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.ChainRNG, (UInt64, Int, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.next_bits, (PureAdsorb.ChainRNG,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.unit_from_bits, (UInt64, Type{Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.unit_from_bits, (UInt64, Type{Float32}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_uniform, (PureAdsorb.ChainRNG, Type{Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_uniform, (PureAdsorb.ChainRNG, Type{Float32}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_range, (PureAdsorb.ChainRNG, Int64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_range, (PureAdsorb.ChainRNG, Int32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_normal, (PureAdsorb.ChainRNG, Type{Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_normal, (PureAdsorb.ChainRNG, Type{Float32}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.shoemake_quaternion, (Float64, Float64, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.shoemake_quaternion, (Float32, Float32, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_quaternion, (PureAdsorb.ChainRNG, Type{Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rand_quaternion, (PureAdsorb.ChainRNG, Type{Float32}); guarantees = (:typestable, :noalloc)),
    # Task 5's moves (`src/moves.jl`).
    signature_findings(PureAdsorb.qmul, (SVector{4, Float64}, SVector{4, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.qmul, (SVector{4, Float32}, SVector{4, Float32}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.quaternion_pow, (SVector{4, Float64}, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.quaternion_pow, (SVector{4, Float32}, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_translation, (PureAdsorb.ChainRNG, SVector{3, Float64}, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_translation, (PureAdsorb.ChainRNG, SVector{3, Float32}, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_rotation, (PureAdsorb.ChainRNG, SVector{4, Float64}, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_rotation, (PureAdsorb.ChainRNG, SVector{4, Float32}, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_reinsertion, (PureAdsorb.ChainRNG, SMatrix{3, 3, Float64, 9}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_reinsertion, (PureAdsorb.ChainRNG, SMatrix{3, 3, Float32, 9}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_move, propose_move_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.propose_move, propose_move_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.metropolis_accept, (Float64, Float64, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.metropolis_accept, (Float32, Float32, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.onehot_movetype, (Int32,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.select_and_propose, select_and_propose_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.select_and_propose, select_and_propose_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.default_nblocks_per_chain, (Type{Float64}, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.default_nblocks_per_chain, (Type{Float32}, Int); guarantees = (:typestable, :noalloc)),
    # Milestone C task 3's μVT exchange moves (`src/moves.jl`): the device-kernel-safe primitives
    # `evaluate_insert_kernel!`/`decide_insert_kernel!`/`evaluate_delete_kernel!`/
    # `decide_delete_kernel!`/`apply_exchange_sk_kernel!` call, both precisions. The kernels
    # themselves are not audited here for the same reason `evaluate_move_kernel!`/
    # `decide_move_kernel!`/`apply_sk_kernel!` are not; `exchange_constant_term`/
    # `exchange_constant_coeffs` are host-only and allocating by design (their own docstrings) and
    # are excluded for the same reason `insertion_constant_term` (`nvt.jl`) always has been.
    signature_findings(PureAdsorb.log_insertion_prefactor, (Float64, Float64, Float64, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.log_insertion_prefactor, (Float32, Float32, Float32, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.log_deletion_prefactor, (Float64, Float64, Float64, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.log_deletion_prefactor, (Float32, Float32, Float32, Int); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.metropolis_accept_muvt, (Float64, Float64, Float64, Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.metropolis_accept_muvt, (Float32, Float32, Float32, Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.no_guest_sites, (Type{Float64}, Val{3}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.no_guest_sites, (Type{Float32}, Val{3}); guarantees = (:typestable, :noalloc)),
    signature_findings(
        PureAdsorb.reciprocal_exchange_energy,
        (PureAdsorb.Guest{Float64, 3}, SVector{3, SVector{3, Float64}}, SVector{3, SVector{3, Float64}}, Vector{SVector{3, Float64}}, Vector{Float64}, Vector{ComplexF64});
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.reciprocal_exchange_energy,
        (PureAdsorb.Guest{Float32, 3}, SVector{3, SVector{3, Float32}}, SVector{3, SVector{3, Float32}}, Vector{SVector{3, Float32}}, Vector{Float32}, Vector{ComplexF32});
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.guest_pair_realspace_energy_range, guest_pair_realspace_energy_range_types(Float64);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.guest_pair_realspace_energy_range, guest_pair_realspace_energy_range_types(Float32);
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(
        PureAdsorb.select_delete_index, (Int, Vector{Int32}, Vector{Int32}, Vector{UInt64}, Vector{UInt64}, Int);
        guarantees = (:typestable, :noalloc)
    ),
)
StrictMode.format_findings(stdout, results; format = :text)
exit(StrictMode.nfailures(results))
