# A wrong incremental `ΔU` still produces a plausible-looking Markov chain that silently samples
# the wrong distribution; recomputing the total energy from scratch periodically and comparing
# it against the value accumulated from accepted `ΔU`s is what catches that.

"""
    energy_audit_tolerance(accumulated::F, recomputed::F, nmoves::Integer) -> F

Rounding-error tolerance for `audit_energy!`: sequentially adding `nmoves` accepted `ΔU`s to a
running total accumulates a worst-case forward error of about `nmoves * u * magnitude`
(`u = eps(F)`, one unit-in-the-last-place per addition — the standard bound for recursive
floating-point summation, Higham, *Accuracy and Stability of Numerical Algorithms*, §4.2), taking
`magnitude` as the larger of the accumulated and recomputed values: both are proxies for the
scale of the terms actually summed, since a Metropolis chain essentially never accepts a `ΔU`
much larger than the equilibrium energy itself. A floor of `one(F)` keeps the tolerance from
vanishing when the energy itself is near zero (e.g. a lightly loaded system).
"""
energy_audit_tolerance(accumulated::F, recomputed::F, nmoves::Integer) where {F} = nmoves * eps(F) * max(abs(accumulated), abs(recomputed), one(F))

"""
    audit_energy!(batch, state, guest, ff, n, nmoves; tol = nothing) -> nothing

Recompute system `n`'s total energy from scratch (`total_energy`) and compare it against
`state.energy[n]`, the running value accumulated from `nmoves` accepted `ΔU`s since it was last
set exactly (construction, or the previous call to this function). Throws an `ArgumentError`
naming the system, the accumulated value, the recomputed value and the discrepancy when they
disagree by more than `tol` (default `energy_audit_tolerance`) — a fail-fast check, never a
warning, since a discrepancy here means the running total no longer tracks the true energy of
the configuration at all. On success, resets `state.energy[n]` to the freshly recomputed value,
so rounding drift never compounds past what a single audit interval can introduce.
"""
function audit_energy!(
        batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F}, ff::ForceField{F}, n::Integer, nmoves::Integer;
        tol::Union{Nothing, F} = nothing
    ) where {F}
    recomputed = total_energy(batch, state, guest, ff, n)
    accumulated = state.energy[n]
    τ = something(tol, energy_audit_tolerance(accumulated, recomputed, nmoves))
    discrepancy = abs(recomputed - accumulated)
    discrepancy <= τ || throw(
        ArgumentError(
            "energy audit failed for system $n ($F): accumulated=$accumulated, recomputed=$recomputed, " *
                "discrepancy=$discrepancy exceeds tolerance $τ over $nmoves accepted move(s) since the last audit"
        )
    )
    state.energy[n] = recomputed
    return nothing
end
