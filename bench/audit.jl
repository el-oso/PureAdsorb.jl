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
reciprocal_move_delta_energy_types(::Type{T}) where {T} = (
    PureAdsorb.Guest{T, 3}, SVector{3, T}, SVector{4, T}, SVector{3, T}, SVector{4, T},
    Vector{SVector{3, T}}, Vector{T}, Vector{Complex{T}},
)
reciprocal_move_delta_bang_types(::Type{T}) where {T} = (Vector{Complex{T}}, reciprocal_move_delta_energy_types(T)...)
guest_sites_at_types(::Type{T}) where {T} = (PureAdsorb.Guest{T, 3}, SVector{3, T}, SVector{4, T})
total_reciprocal_energy_types(::Type{T}) where {T} = (Vector{T}, Vector{Complex{T}}, Vector{Complex{T}})

# `SystemState`'s field types for a concrete precision, matching exactly what `SystemState`'s
# own constructor produces (`src/state.jl`), for auditing its accessors without building a real
# batch and state just to call `typeof` on them.
systemstate_type(::Type{T}) where {T} = PureAdsorb.SystemState{
    T, Vector{SVector{3, T}}, Vector{SVector{4, T}}, Vector{Int32}, Vector{Complex{T}}, Vector{T}, Vector{UInt64}, Vector{SVector{3, Int32}},
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
    signature_findings(PureAdsorb.reciprocal_move_delta!, reciprocal_move_delta_bang_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.reciprocal_move_delta!, reciprocal_move_delta_bang_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_sites_at, guest_sites_at_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.guest_sites_at, guest_sites_at_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.total_reciprocal_energy, total_reciprocal_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.total_reciprocal_energy, total_reciprocal_energy_types(Float32); guarantees = (:typestable, :noalloc)),
)
StrictMode.format_findings(stdout, results; format = :text)
exit(StrictMode.nfailures(results))
