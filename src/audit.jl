# A wrong incremental `ΔU` still produces a plausible-looking Markov chain that silently samples
# the wrong distribution; recomputing the total energy from scratch periodically and comparing
# it against the value accumulated from accepted `ΔU`s is what catches that. But `total_energy`
# itself reads the reciprocal term off the running `state.Sk`, and `ΔU_recip` is identically the
# change in `Σ_k pref·|Sk|²` under `Sk ← Sk + ΔS` for ANY `ΔS` — so a `ΔS` with the wrong sign,
# phase, or guest leaves the running `Sk` and the energy accumulated from it self-consistently
# wrong, and a check against `total_energy` alone cannot see it. The fix is to audit `Sk` first,
# against a quantity `total_energy` never touches: `Sk` rebuilt from the poses themselves
# (`guest_site_positions_charges`, `structure_factor`, the same construction `SystemState` uses).
# Only once that element-wise check passes — and `state.Sk` has been overwritten with the
# rebuilt, pose-consistent value — is it sound to recompute the energy from it and check that.

# Two independent rounding sources bound each of `audit_energy!`'s two comparisons, and both
# scale with a magnitude rather than a plain count. Higham's bound for a running sum of `nmoves`
# accepted terms computed by repeated addition (Higham, *Accuracy and Stability of Numerical
# Algorithms*, chapter 4) is `nmoves * u * Σ|term|` — `u = eps(F)`, scaled by the sum of the
# MAGNITUDES OF THE TERMS ACTUALLY SUMMED, not by the magnitude of the running total: a Metropolis
# chain's accepted terms fluctuate in sign around an equilibrium (a structure-factor contribution
# can also cancel across guests), so the running total can sit far below the sum of magnitudes
# that produced it, and using the total as the scale makes the bound too tight exactly when that
# cancellation is heaviest. `abs_accum` is that running Σ|term|
# (`SystemState.sk_abs_accum`/`energy_abs_accum`). Separately, the FROM-SCRATCH recomputation each
# comparison checks against (`total_energy`, or `structure_factor`'s rebuild of one k-vector) is
# itself a sum of several terms, so it carries its own one-off rounding of order `eps(F) *
# magnitude` regardless of `nmoves` — the running total's accumulated error and the rebuild's own
# rounding are separate sources and neither term above stands in for the other. Both terms are
# floored at `one(F)` so a near-zero accumulation or magnitude still gets a nonzero tolerance.
_accumulation_tolerance(abs_accum::F, magnitude::F, nmoves::Integer) where {F} =
    nmoves * eps(F) * max(abs_accum, one(F)) + eps(F) * max(magnitude, one(F))

"""
    energy_audit_tolerance(abs_accum::F, magnitude::F, nmoves::Integer) -> F

Rounding-error tolerance for `audit_energy!`'s comparison of `state.energy[n]` (accumulated from
`nmoves` accepted `ΔU`s) against a from-scratch recomputation: `abs_accum` is the running `Σ|ΔU|`
over those moves (`SystemState.energy_abs_accum`), `magnitude` the larger of the accumulated and
freshly recomputed energy — see this file's `_accumulation_tolerance` for the two-term
derivation.
"""
energy_audit_tolerance(abs_accum::F, magnitude::F, nmoves::Integer) where {F} =
    _accumulation_tolerance(abs_accum, magnitude, nmoves)

"""
    sk_audit_tolerance(abs_accum::F, magnitude::F, nmoves::Integer) -> F

Rounding-error tolerance for `audit_energy!`'s element-wise check on `state.Sk`: the same
argument as `energy_audit_tolerance`, applied per k-vector, with `abs_accum` the running
per-move rounding scale accumulated at that k-vector over `nmoves` accepted moves
(`SystemState.sk_abs_accum`) and `magnitude` the rebuild-side scale for that k-vector — the sum of
the magnitudes of the terms `structure_factor`'s rebuild actually sums there, scaled by how many
terms it sums (`audit_energy!`'s own computation), since that recomputation's rounding grows with
both.
"""
sk_audit_tolerance(abs_accum::F, magnitude::F, nmoves::Integer) where {F} =
    _accumulation_tolerance(abs_accum, magnitude, nmoves)

"""
    audit_energy!(batch, state, guest, ff, n, nmoves; tol = nothing, sk_tol = nothing) -> nothing

Audit system `n` in three stages, in this order.

First, check `nguests(state, n) <= capacity(state, n)`. Throws an `ArgumentError` naming the
system, its occupancy and its capacity if it does not: `insert_guest!` already refuses to write
past capacity, so this only fires if some other code path corrupted `occupancy` directly, but a
chain that silently saturated its capacity would sample a truncated distribution while still
looking healthy, so this is checked here too rather than trusted to have held.

Then, rebuild `state.Sk`'s slice for system `n` from the current poses alone
(`guest_site_positions_charges`, `structure_factor` — the same construction `SystemState`'s
constructor uses) and compare it element-wise against the running value. Throws an
`ArgumentError` naming the system, the k-vector index, the running and rebuilt values and the
discrepancy when any element disagrees by more than `sk_tol` (default `sk_audit_tolerance`).
This is the check that actually exercises the structure-factor update: `total_energy`'s
reciprocal term is a function of `Sk` alone, so a `ΔS` with the wrong sign, phase or guest leaves
`state.energy` and a `total_energy` recomputed from the same corrupted `Sk` self-consistently
equal, and a comparison that skipped this stage would pass regardless. On success, overwrites
`state.Sk`'s slice with the rebuilt value (clearing any Float32 rounding drift instead of merely
detecting it) and zeroes `state.sk_abs_accum`'s slice, since the accumulated error the tolerance
was sized against is cleared at the same moment.

Only then recompute system `n`'s total energy from scratch (`total_energy`, now reading the
just-rebuilt `Sk`) and compare it against `state.energy[n]`, the running value accumulated from
`nmoves` accepted `ΔU`s since it was last set exactly (construction, or the previous call to this
function). Throws an `ArgumentError` naming the system, the accumulated value, the recomputed
value and the discrepancy when they disagree by more than `tol` (default
`energy_audit_tolerance`). On success, resets `state.energy[n]` to the freshly recomputed value
and zeroes `state.energy_abs_accum[n]`.

All three stages are fail-fast, never a warning: a discrepancy at any one of them means the
running state no longer tracks the true configuration.
"""
function audit_energy!(
        batch::FrameworkBatch{F}, state::SystemState{F}, guest::Guest{F}, ff::ForceField{F}, n::Integer, nmoves::Integer;
        tol::Union{Nothing, F} = nothing, sk_tol::Union{Nothing, F} = nothing
    ) where {F}
    occ = nguests(state, n)
    cap = capacity(state, n)
    occ <= cap ||
        throw(ArgumentError("energy audit failed for system $n ($F): occupancy $occ exceeds capacity $cap"))

    gr = guest_range(state, n)
    kr = kvec_range(state, n)
    krb = batch_kvec_range(batch, n)
    sitepos, siteq = guest_site_positions_charges(guest, state.refpoints, state.orientations, gr)
    rebuilt_Sk = view(batch.Shost, krb) .+ structure_factor(view(batch.ks, krb), sitepos, siteq)
    # `structure_factor`'s rebuild at one k-vector sums `nterms_rebuild` complex terms (every
    # guest site's `charge * cis(k·r)`, `|cis| = 1`) plus `batch.Shost`; Higham's bound for that
    # summation scales with the NUMBER of terms as well as their magnitude, so the recompute-side
    # tolerance below is `nterms_rebuild` times the sum of those terms' magnitudes, not just once.
    nterms_rebuild = length(gr) * length(guest.sites)
    total_abs_charge = sum(abs, siteq)
    # `kr` indexes `state.Sk` (one slice per SYSTEM); `krb` indexes `batch.Shost` (one slice per
    # FRAMEWORK, `FrameworkBatch`'s docstring) — same length, paired by the shared relative
    # `offset` rather than by a single absolute index.
    for offset in eachindex(kr, krb)
        kidx = kr[offset]
        kidx_b = krb[offset]
        running = state.Sk[kidx]
        rebuilt = rebuilt_Sk[offset]
        rebuild_magnitude = nterms_rebuild * (abs(batch.Shost[kidx_b]) + total_abs_charge)
        τk = something(sk_tol, sk_audit_tolerance(state.sk_abs_accum[kidx], rebuild_magnitude, nmoves))
        discrepancy = abs(rebuilt - running)
        discrepancy <= τk || throw(
            ArgumentError(
                "energy audit failed for system $n ($F): structure factor at k-vector $offset disagrees with the " *
                    "poses: running=$running, rebuilt=$rebuilt, discrepancy=$discrepancy exceeds tolerance $τk " *
                    "over $nmoves accepted move(s) since the last audit"
            )
        )
    end
    state.Sk[kr] .= rebuilt_Sk
    state.sk_abs_accum[kr] .= zero(F)

    recomputed = total_energy(batch, state, guest, ff, n)
    accumulated = state.energy[n]
    τ = something(tol, energy_audit_tolerance(state.energy_abs_accum[n], max(abs(accumulated), abs(recomputed)), nmoves))
    discrepancy = abs(recomputed - accumulated)
    discrepancy <= τ || throw(
        ArgumentError(
            "energy audit failed for system $n ($F): accumulated=$accumulated, recomputed=$recomputed, " *
                "discrepancy=$discrepancy exceeds tolerance $τ over $nmoves accepted move(s) since the last audit"
        )
    )
    state.energy[n] = recomputed
    state.energy_abs_accum[n] = zero(F)
    return nothing
end
