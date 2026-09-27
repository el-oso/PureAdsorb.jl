@testsnippet AuditChain begin
    using StaticArrays, LinearAlgebra, Random

    # A minimal Metropolis chain (translation moves only, always against system 1) used only to
    # produce a realistic sequence of ACCEPTED `ΔU`s for the audit to check — not a stand-in for
    # task 5/7's real move set or cycle driver. `corrupt_after_accept` (a 1-based accepted-move
    # count) adds `corrupt_amount` to that one accepted `ΔU` before it is folded into
    # `state.energy`, the deliberate error the audit must catch; `nothing` runs cleanly.
    function run_chain!(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N}, n::Integer,
            nsteps::Integer, rng::AbstractRNG, kT::F; corrupt_after_accept::Union{Nothing, Integer} = nothing, corrupt_amount::F = zero(F)
        ) where {F, N}
        naccept = 0
        for _ in 1:nsteps
            gr = PureAdsorb.guest_range(state, n)
            i = rand(rng, gr)
            kr = PureAdsorb.kvec_range(state, n)
            ΔS = zeros(Complex{F}, length(kr))
            oldpos = state.refpoints[i]
            newpos = oldpos + SVector{3, F}(randn(rng, F, 3))
            newq = normalize(SVector{4, F}(rand(rng, F, 4) .- F(0.5)))
            ΔU, host_energy_new = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS)
            accept = ΔU <= zero(F) || rand(rng) < exp(-Float64(ΔU) / Float64(kT))
            if accept
                naccept += 1
                naccept == corrupt_after_accept && (ΔU += corrupt_amount)
                state.refpoints[i] = newpos
                state.orientations[i] = newq
                state.Sk[kr] .+= ΔS
                state.sk_abs_accum[kr] .+= 2 * sum(abs, guest.charges)
                state.host_energy[i] = host_energy_new
                state.energy[n] += ΔU
                state.energy_abs_accum[n] += abs(ΔU)
            end
        end
        return naccept
    end
end

@testitem "audit_energy! passes on a clean chain" setup = [AuditChain] begin
    using Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4], ff; T = 298.15, seed = 9)
    kT = PureAdsorb.KB * 298.15
    naccept = run_chain!(b, st, g, 1, 2000, Xoshiro(999), kT)
    naccept > 0 || error("test setup produced no accepted moves; widen the proposal or lower kT")
    before = st.energy[1]
    PureAdsorb.audit_energy!(b, st, g, ff, 1, naccept)
    # A passing audit resets the running total to the fresh recomputation.
    recomputed = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test st.energy[1] == recomputed
    @test st.energy[1] ≈ before rtol = 1.0e-9
end

@testitem "audit_energy! catches a corrupted ΔU (Float64)" setup = [AuditChain] begin
    using Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4], ff; T = 298.15, seed = 9)
    kT = PureAdsorb.KB * 298.15
    # 1e-6 eV is many orders of magnitude above the ~1e-13 eV tolerance a few thousand accepted
    # Float64 moves derive (`energy_audit_tolerance`), so this is caught regardless of exactly
    # which accepted move it lands on.
    naccept = run_chain!(b, st, g, 1, 2000, Xoshiro(999), kT; corrupt_after_accept = 100, corrupt_amount = 1.0e-6)
    naccept >= 100 || error("test setup produced fewer than 100 accepted moves; the corruption never landed")
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, 1, naccept)
end

@testitem "audit_energy! catches a corrupted ΔU (Float32)" setup = [AuditChain] begin
    using Random
    # Float32 accumulation over many moves is exactly the case the audit exists to catch: this
    # runs long enough (thousands of accepted moves) for genuine Float32 rounding to accumulate,
    # and checks that a clean chain at that length still passes while a corrupted one still fails.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st_clean = PureAdsorb.SystemState(b, g, [4], ff; T = 298.15f0, seed = 9)
    kT = Float32(PureAdsorb.KB * 298.15)
    naccept_clean = run_chain!(b, st_clean, g, 1, 5000, Xoshiro(999), kT)
    naccept_clean > 0 || error("test setup produced no accepted moves")
    PureAdsorb.audit_energy!(b, st_clean, g, ff, 1, naccept_clean)   # must not throw

    st_bad = PureAdsorb.SystemState(b, g, [4], ff; T = 298.15f0, seed = 9)
    # 1e-2 eV is far above Float32's own rounding floor over a few thousand accumulated moves
    # (`energy_audit_tolerance` scales as `nmoves * eps(Float32) * energy_scale`, of order
    # 1e-4-1e-3 eV here), so this is a deliberate corruption, not marginal rounding.
    naccept_bad = run_chain!(b, st_bad, g, 1, 5000, Xoshiro(999), kT; corrupt_after_accept = 200, corrupt_amount = 1.0f-2)
    naccept_bad >= 200 || error("test setup produced fewer than 200 accepted moves; the corruption never landed")
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st_bad, g, ff, 1, naccept_bad)
end

@testsnippet WrongSignRecip begin
    using StaticArrays, LinearAlgebra, Random

    # A self-consistent, wrong-signed structure-factor update: `reciprocal_move_delta!` called
    # with the old and new poses swapped computes the exact reciprocal-space energy change and
    # `ΔS` for the REVERSE transition, which is `-ΔS_true` — since the underlying formula
    # (`|Sk + ΔS|² = |Sk|² + 2 Re[conj(Sk) ΔS] + |ΔS|²`) is an algebraic identity for ANY `ΔS`,
    # applying this pair to `state.Sk` and `state.energy` together leaves them mutually
    # consistent even though the sign is wrong for the pose transition that actually happened.
    # This is the class of bug P1 exists to catch: a plain recompute-and-compare energy audit
    # cannot see it, because `total_energy` reads its reciprocal term off the same (self-
    # consistently wrong) `Sk`.
    function apply_wrong_sign_move!(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N}, n::Integer,
            i::Integer, newpos::SVector{3, F}, newq::SVector{4, F}
        ) where {F, N}
        oldpos = state.refpoints[i]; oldq = state.orientations[i]
        kr = PureAdsorb.kvec_range(state, n)
        ΔS_true = zeros(Complex{F}, length(kr))
        ΔU_true, host_energy_new = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS_true)
        ΔU_recip_true = PureAdsorb.reciprocal_move_delta!(
            zeros(Complex{F}, length(kr)), guest, oldpos, oldq, newpos, newq,
            view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr)
        )
        ΔS_wrong = zeros(Complex{F}, length(kr))
        ΔU_recip_wrong = PureAdsorb.reciprocal_move_delta!(
            ΔS_wrong, guest, newpos, newq, oldpos, oldq,
            view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr)
        )
        # The real pose transition, paired with the wrong-signed reciprocal contribution: poses
        # and `Sk` now disagree, but `Sk` and `state.energy` stay self-consistent with each other.
        ΔU_wrong = (ΔU_true - ΔU_recip_true) + ΔU_recip_wrong
        state.refpoints[i] = newpos
        state.orientations[i] = newq
        state.host_energy[i] = host_energy_new
        state.Sk[kr] .+= ΔS_wrong
        state.sk_abs_accum[kr] .+= 2 * sum(abs, guest.charges)
        state.energy[n] += ΔU_wrong
        state.energy_abs_accum[n] += abs(ΔU_wrong)
        return nothing
    end
end

@testitem "a wrong-signed structure-factor update passes an energy-only audit but fails the real one (Float64)" setup = [WrongSignRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15, seed = 21)
    n = 1
    i = first(PureAdsorb.guest_range(st, n))
    rng = Xoshiro(5)
    oldpos = st.refpoints[i]
    newpos = oldpos + SVector{3, Float64}(randn(rng, 3))
    newq = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
    apply_wrong_sign_move!(b, st, g, n, i, newpos, newq)

    # The energy-only comparison `audit_energy!` used before P1 (recompute from `Sk`, compare
    # against the running total): it passes, proving the corruption is invisible to it.
    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)

    # The real audit rebuilds `Sk` from the poses first and catches the mismatch.
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "a wrong-signed structure-factor update passes an energy-only audit but fails the real one (Float32)" setup = [WrongSignRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15f0, seed = 21)
    n = 1
    i = first(PureAdsorb.guest_range(st, n))
    rng = Xoshiro(5)
    oldpos = st.refpoints[i]
    newpos = oldpos + SVector{3, Float32}(randn(rng, Float32, 3))
    newq = normalize(SVector{4, Float32}(rand(rng, Float32, 4) .- 0.5f0))
    apply_wrong_sign_move!(b, st, g, n, i, newpos, newq)

    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testsnippet PhaseAndWrongGuestRecip begin
    using StaticArrays, LinearAlgebra, Random

    # A self-consistent structure-factor update with the right magnitude but the wrong phase:
    # rotating every element of the true `ΔS` by a fixed nonzero angle leaves `|ΔS|` unchanged
    # while changing how it projects onto the running `Sk`. Applying the reciprocal-energy
    # formula to this rotated `ΔS` (rather than to the true one) gives an `(Sk, energy)` pair
    # that is mutually consistent, exactly like the sign-flip case above.
    function apply_phase_error_move!(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N}, n::Integer,
            i::Integer, newpos::SVector{3, F}, newq::SVector{4, F}, θ::F
        ) where {F, N}
        oldpos = state.refpoints[i]; oldq = state.orientations[i]
        kr = PureAdsorb.kvec_range(state, n)
        ΔS_true = zeros(Complex{F}, length(kr))
        ΔU_true, host_energy_new = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS_true)
        ΔU_recip_true = PureAdsorb.reciprocal_move_delta!(
            zeros(Complex{F}, length(kr)), guest, oldpos, oldq, newpos, newq,
            view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr)
        )
        ΔS_phase = ΔS_true .* cis(θ)
        kpref = view(batch.kprefactor, kr)
        Sk_before = view(state.Sk, kr)
        ΔU_recip_phase = F(PureAdsorb.KE) * sum(
            kpref[idx] * (2 * real(conj(Sk_before[idx]) * ΔS_phase[idx]) + abs2(ΔS_phase[idx])) for idx in eachindex(kpref)
        )
        ΔU_phase = (ΔU_true - ΔU_recip_true) + ΔU_recip_phase
        state.refpoints[i] = newpos
        state.orientations[i] = newq
        state.host_energy[i] = host_energy_new
        state.Sk[kr] .+= ΔS_phase
        state.sk_abs_accum[kr] .+= 2 * sum(abs, guest.charges)
        state.energy[n] += ΔU_phase
        state.energy_abs_accum[n] += abs(ΔU_phase)
        return nothing
    end

    # A self-consistent structure-factor update attributed to the wrong guest: the reciprocal
    # change is computed for a hypothetical transition of guest `i` (whose pose never actually
    # changes), while guest `j`'s pose is the one that really moves. `Sk`/`energy` absorb guest
    # `i`'s reciprocal contribution and guest `j`'s real-space contribution together, which is
    # self-consistent as a pair even though neither guest's own transition matches what was
    # recorded.
    function apply_wrong_guest_move!(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N}, n::Integer,
            i::Integer, j::Integer, newpos_i::SVector{3, F}, newq_i::SVector{4, F}, newpos_j::SVector{3, F}, newq_j::SVector{4, F}
        ) where {F, N}
        kr = PureAdsorb.kvec_range(state, n)
        oldpos_i = state.refpoints[i]; oldq_i = state.orientations[i]
        ΔS_i = zeros(Complex{F}, length(kr))
        ΔU_recip_i = PureAdsorb.reciprocal_move_delta!(
            ΔS_i, guest, oldpos_i, oldq_i, newpos_i, newq_i, view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr)
        )
        oldpos_j = state.refpoints[j]; oldq_j = state.orientations[j]
        ΔS_j_true = zeros(Complex{F}, length(kr))
        ΔU_true_j, host_energy_new_j = PureAdsorb.guest_move_delta(batch, state, guest, n, j, newpos_j, newq_j, ΔS_j_true)
        ΔU_recip_true_j = PureAdsorb.reciprocal_move_delta!(
            zeros(Complex{F}, length(kr)), guest, oldpos_j, oldq_j, newpos_j, newq_j,
            view(batch.ks, kr), view(batch.kprefactor, kr), view(state.Sk, kr)
        )
        ΔU_realspace_j = ΔU_true_j - ΔU_recip_true_j
        # Guest i's pose is left untouched; only guest j's actually moves.
        ΔU_wrong_guest = ΔU_realspace_j + ΔU_recip_i
        state.refpoints[j] = newpos_j
        state.orientations[j] = newq_j
        state.host_energy[j] = host_energy_new_j
        state.Sk[kr] .+= ΔS_i
        state.sk_abs_accum[kr] .+= 2 * sum(abs, guest.charges)
        state.energy[n] += ΔU_wrong_guest
        state.energy_abs_accum[n] += abs(ΔU_wrong_guest)
        return nothing
    end
end

@testitem "a phase-perturbed structure-factor update passes an energy-only audit but fails the real one (Float64)" setup = [PhaseAndWrongGuestRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15, seed = 21)
    n = 1
    i = first(PureAdsorb.guest_range(st, n))
    rng = Xoshiro(5)
    oldpos = st.refpoints[i]
    newpos = oldpos + SVector{3, Float64}(randn(rng, 3))
    newq = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
    apply_phase_error_move!(b, st, g, n, i, newpos, newq, 1.0)

    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "a phase-perturbed structure-factor update passes an energy-only audit but fails the real one (Float32)" setup = [PhaseAndWrongGuestRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15f0, seed = 21)
    n = 1
    i = first(PureAdsorb.guest_range(st, n))
    rng = Xoshiro(5)
    oldpos = st.refpoints[i]
    newpos = oldpos + SVector{3, Float32}(randn(rng, Float32, 3))
    newq = normalize(SVector{4, Float32}(rand(rng, Float32, 4) .- 0.5f0))
    apply_phase_error_move!(b, st, g, n, i, newpos, newq, 1.0f0)

    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "a structure-factor update attributed to the wrong guest passes an energy-only audit but fails the real one (Float64)" setup = [PhaseAndWrongGuestRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15, seed = 21)
    n = 1
    gr = PureAdsorb.guest_range(st, n)
    i, j = gr[1], gr[2]
    rng = Xoshiro(13)
    newpos_i = st.refpoints[i] + SVector{3, Float64}(randn(rng, 3))
    newq_i = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
    newpos_j = st.refpoints[j] + SVector{3, Float64}(randn(rng, 3))
    newq_j = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
    apply_wrong_guest_move!(b, st, g, n, i, j, newpos_i, newq_i, newpos_j, newq_j)

    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "a structure-factor update attributed to the wrong guest passes an energy-only audit but fails the real one (Float32)" setup = [PhaseAndWrongGuestRecip] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15f0, seed = 21)
    n = 1
    gr = PureAdsorb.guest_range(st, n)
    i, j = gr[1], gr[2]
    rng = Xoshiro(13)
    newpos_i = st.refpoints[i] + SVector{3, Float32}(randn(rng, Float32, 3))
    newq_i = normalize(SVector{4, Float32}(rand(rng, Float32, 4) .- 0.5f0))
    newpos_j = st.refpoints[j] + SVector{3, Float32}(randn(rng, Float32, 3))
    newq_j = normalize(SVector{4, Float32}(rand(rng, Float32, 4) .- 0.5f0))
    apply_wrong_guest_move!(b, st, g, n, i, j, newpos_i, newq_i, newpos_j, newq_j)

    legacy_recomputed = PureAdsorb.total_energy(b, st, g, ff, n)
    @test abs(legacy_recomputed - st.energy[n]) <=
        PureAdsorb.energy_audit_tolerance(st.energy_abs_accum[n], max(abs(st.energy[n]), abs(legacy_recomputed)), 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "audit_energy! catches a stale host-energy cache entry" begin
    using Random, StaticArrays, LinearAlgebra
    # `guest_move_delta` never mutates `state.host_energy` itself; a caller that reads a move's
    # returned new value but forgets to write it back (or writes a wrong one) leaves the cache
    # out of step with the guest's actual pose. `total_energy` never reads the cache, so the next
    # move computed from that guest folds a wrong `ΔU` into the running total, caught here.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [6], ff; T = 298.15, seed = 21)
    n = 1
    i = first(PureAdsorb.guest_range(st, n))
    st.host_energy[i] += 1.0   # stale by 1 eV, far above any rounding tolerance

    rng = Xoshiro(7)
    kr = PureAdsorb.kvec_range(st, n)
    ΔS = zeros(ComplexF64, length(kr))
    newpos = st.refpoints[i] + SVector{3, Float64}(randn(rng, 3))
    newq = normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
    ΔU, host_energy_new = PureAdsorb.guest_move_delta(b, st, g, n, i, newpos, newq, ΔS)
    st.refpoints[i] = newpos
    st.orientations[i] = newq
    st.Sk[kr] .+= ΔS
    st.sk_abs_accum[kr] .+= 2 * sum(abs, g.charges)
    st.host_energy[i] = host_energy_new   # the cache is fixed going forward...
    st.energy[n] += ΔU                    # ...but the running energy already absorbed the stale ΔU
    st.energy_abs_accum[n] += abs(ΔU)

    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "energy_audit_tolerance scales with nmoves, abs_accum, magnitude and precision" begin
    @test PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200) ≈ 200 * eps(Float64) * 10.0 + eps(Float64) * 10.0
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) ≈ 200 * eps(Float32) * 10.0f0 + eps(Float32) * 10.0f0
    # Float32's tolerance is looser than Float64's at the same (nmoves, abs_accum, magnitude):
    # eps(F32) ≫ eps(F64), which is the whole reason Float32 accumulation is the case worth
    # testing directly.
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) > PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200)
    # A near-zero accumulation and a near-zero energy still get a nonzero tolerance (the `one(F)`
    # floor on each term: `50 * eps` from the accumulation term's own floor, `1 * eps` from the
    # magnitude term's).
    @test PureAdsorb.energy_audit_tolerance(0.0, 0.0, 50) ≈ 51 * eps(Float64)
    # A larger sum of |ΔU| terms (heavier accepted-move traffic since the last audit) widens the
    # tolerance even at the same (nmoves, magnitude), which is the whole point of tracking it
    # separately from the running total's own magnitude.
    @test PureAdsorb.energy_audit_tolerance(100.0, 10.0, 200) > PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200)
    # A larger recompute magnitude (at the same nmoves and abs_accum) also widens the tolerance:
    # `total_energy`'s own from-scratch summation carries rounding proportional to its own scale,
    # independent of how many moves preceded the recompute.
    @test PureAdsorb.energy_audit_tolerance(10.0, 1000.0, 200) > PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200)
end

@testitem "sk_audit_tolerance scales with nmoves, abs_accum, magnitude and precision" begin
    @test PureAdsorb.sk_audit_tolerance(5.0, 5.0, 200) ≈ 200 * eps(Float64) * 5.0 + eps(Float64) * 5.0
    @test PureAdsorb.sk_audit_tolerance(5.0f0, 5.0f0, 200) ≈ 200 * eps(Float32) * 5.0f0 + eps(Float32) * 5.0f0
    @test PureAdsorb.sk_audit_tolerance(5.0f0, 5.0f0, 200) > PureAdsorb.sk_audit_tolerance(5.0, 5.0, 200)
    # A near-zero accumulation and a near-zero magnitude still get a nonzero tolerance (the
    # `one(F)` floor on each term).
    @test PureAdsorb.sk_audit_tolerance(0.0, 0.0, 50) ≈ 51 * eps(Float64)
    @test PureAdsorb.sk_audit_tolerance(50.0, 5.0, 200) > PureAdsorb.sk_audit_tolerance(5.0, 5.0, 200)
    @test PureAdsorb.sk_audit_tolerance(5.0, 500.0, 200) > PureAdsorb.sk_audit_tolerance(5.0, 5.0, 200)
end
