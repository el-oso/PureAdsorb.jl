# Unitful at the boundary, bare floats in the kernels: this file wraps exactly the path E1 broke
# (a caller-supplied pressure crossing into `peng_robinson_fugacity` and the μVT exchange moves'
# `fugacity` argument, `log_insertion_prefactor`/`log_deletion_prefactor`'s `PASCAL` conversion).
# Every method below strips a `Unitful.Quantity` to a bare pascal/kelvin float and forwards to the
# existing bare-Float method that every kernel, `FrameworkBatch` and `SystemState` field already
# runs on; none of those keep a `Quantity` anywhere, and this file adds no method any of them call
# into. A caller with no `Unitful` values sees no change at all: the bare-Float methods this file
# forwards to are unmodified originals, selected by the same dispatch that already existed.

"""
    peng_robinson_fugacity(P::Unitful.Pressure, Tgas::Unitful.Temperature, tc, pc, omega) -> (; f, phi, Z)
    peng_robinson_fugacity(P::Unitful.Pressure, Tgas::Unitful.Temperature, guest::Guest) -> (; f, phi, Z)

Unitful boundary for `peng_robinson_fugacity`: strips `P`/`Tgas` to bare pascals/kelvin, calls the
bare-Float method every other call site uses, and reattaches `u"Pa"` to the returned fugacity `f`
(`phi`/`Z` are already dimensionless, so they pass through unchanged). `tc`/`pc`/`omega` (or
`guest`) stay bare Floats, matching how `Guest` stores them.
"""
function peng_robinson_fugacity(P::Unitful.Pressure, Tgas::Unitful.Temperature, tc, pc, omega)
    res = peng_robinson_fugacity(ustrip(u"Pa", P), ustrip(u"K", Tgas), tc, pc, omega)
    return (f = res.f * u"Pa", phi = res.phi, Z = res.Z)
end

function peng_robinson_fugacity(P::Unitful.Pressure, Tgas::Unitful.Temperature, guest::Guest)
    res = peng_robinson_fugacity(ustrip(u"Pa", P), ustrip(u"K", Tgas), guest)
    return (f = res.f * u"Pa", phi = res.phi, Z = res.Z)
end

"""
    mc_insert!(batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)
    mc_delete!(batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)
    mc_exchange!(rng, batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)

Unitful boundary for the μVT exchange moves' fugacity argument: strips each system's fugacity to a
bare pascal `T` and forwards to the bare-Float method, which does the SAME `PASCAL`-scaled
conversion to eV·Å⁻³ (`src/constants.jl`) it always did. Accepting a per-system `Unitful.Pressure`
vector here, at the call boundary, rather than in `mc_insert_kernel!`/`mc_delete_kernel!`
themselves, keeps every kernel argument and device array a plain `T`.
"""
function mc_insert!(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_insert!(batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end

function mc_delete!(
        batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_delete!(batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end

function mc_exchange!(
        rng::AbstractRNG, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_exchange!(rng, batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end
