# Unitful at the boundary, bare floats in the kernels: this file wraps every remaining path where a
# pressure or a temperature crosses from a caller's units into this package's internal eV/Å/K
# convention — the same class of bug E1 was (a factor of 1.6e11 between pascals and eV/Å³, invisible
# to every energy check). A caller with no `Unitful` values sees no change at all.
#
# Two different mechanisms are needed, because Julia dispatches on POSITIONAL argument types but
# NOT on keyword argument types: two methods that differ only in a keyword's annotated type do not
# overload each other, the second DEFINITION REPLACES the first (measured directly: defining
# `f(a; T::AbstractString, kwargs...)` after `f(a; T, n::Integer)` leaves exactly one method, with
# `n` silently absorbed into `kwargs` and `T`'s type frozen to `AbstractString` for every caller).
#
# Where the physical quantity is a POSITIONAL argument (`peng_robinson_fugacity`'s `P`/`Tgas`,
# `mc_insert!`/`mc_delete!`/`mc_exchange!`'s `fugacity`), a genuinely separate, additive method
# below strips it and forwards to the existing bare-Float method, which is never touched.
#
# Where it is a KEYWORD-only argument (`T` on `run_nvt!`, `run_gcmc!`, `run_isotherm!`, `widom` and
# `SystemState`; `pressures` on `run_isotherm!`; `fugacity` on `run_gcmc!`), no such second method is
# possible, so `ustrip_maybe` — an ordinary two-argument function, which DOES dispatch on its
# argument's type — is called at the one line in each of those functions' own bodies that first
# turns the keyword into a bare float. It is the identity for anything already a `Real`, so a
# bare-Float caller's behavior, including the exact bits its dispatch and arithmetic produce, is
# unchanged; only `nvt.jl`/`gcmc.jl`/`widom.jl`/`state.jl`/`isotherm.jl`'s own use of that one
# keyword is touched, never their signatures or any other line.
#
# `ustrip_maybe`'s own two methods ARE a genuine additive dispatch pair (its argument, unlike a
# keyword, is positional), so it needs no forwarding trick of its own.
ustrip_maybe(unit, x::Real) = x
ustrip_maybe(unit, x::Unitful.Quantity) = ustrip(unit, x)

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
    mc_insert!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)
    mc_delete!(ws, batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)
    mc_exchange!(rng, ws, batch, state, guest, guest_types, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT; kwargs...)

Unitful boundary for the μVT exchange moves' fugacity argument: strips each system's fugacity to a
bare pascal `T` and forwards to the bare-Float method, which does the SAME `PASCAL`-scaled
conversion to eV·Å⁻³ (`src/constants.jl`) it always did. Accepting a per-system `Unitful.Pressure`
vector here, at the call boundary, rather than in the exchange kernels themselves, keeps every
kernel argument and device array a plain `T`.
"""
function mc_insert!(
        ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_insert!(ws, batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end

function mc_delete!(
        ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N}, guest_types::SVector{N, Int},
        const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_delete!(ws, batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end

function mc_exchange!(
        rng::AbstractRNG, ws::MoveWorkspace, batch::FrameworkBatch{T}, state::SystemState{T}, guest::Guest{T, N},
        guest_types::SVector{N, Int}, const_p, const_q, fugacity::AbstractVector{<:Unitful.Pressure}, kT::T; kwargs...
    ) where {T, N}
    return mc_exchange!(rng, ws, batch, state, guest, guest_types, const_p, const_q, T.(ustrip.(u"Pa", fugacity)), kT; kwargs...)
end
