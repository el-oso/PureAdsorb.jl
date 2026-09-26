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
            ΔU = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS)
            accept = ΔU <= zero(F) || rand(rng) < exp(-Float64(ΔU) / Float64(kT))
            if accept
                naccept += 1
                naccept == corrupt_after_accept && (ΔU += corrupt_amount)
                state.refpoints[i] = newpos
                state.orientations[i] = newq
                state.Sk[kr] .+= ΔS
                state.energy[n] += ΔU
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
        ΔU_true = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS_true)
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
        state.refpoints[i] = newpos
        state.orientations[i] = newq
        state.Sk[kr] .+= ΔS_wrong
        state.energy[n] += (ΔU_true - ΔU_recip_true) + ΔU_recip_wrong
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
    @test abs(legacy_recomputed - st.energy[n]) <= PureAdsorb.energy_audit_tolerance(st.energy[n], legacy_recomputed, 1)

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
    @test abs(legacy_recomputed - st.energy[n]) <= PureAdsorb.energy_audit_tolerance(st.energy[n], legacy_recomputed, 1)
    @test_throws "structure factor" PureAdsorb.audit_energy!(b, st, g, ff, n, 1)
end

@testitem "energy_audit_tolerance scales with nmoves and precision" begin
    @test PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200) ≈ 200 * eps(Float64) * 10.0
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) ≈ 200 * eps(Float32) * 10.0f0
    # Float32's tolerance is looser than Float64's at the same (nmoves, energy scale): eps(F32) ≫
    # eps(F64), which is the whole reason Float32 accumulation is the case worth testing directly.
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) > PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200)
    # A near-zero energy still gets a nonzero tolerance (the `one(F)` floor).
    @test PureAdsorb.energy_audit_tolerance(0.0, 0.0, 50) ≈ 50 * eps(Float64)
end

@testitem "sk_audit_tolerance scales with nmoves and precision" begin
    @test PureAdsorb.sk_audit_tolerance(3.0 + 4.0im, 3.0 + 4.0im, 200) ≈ 200 * eps(Float64) * 5.0
    @test PureAdsorb.sk_audit_tolerance(3.0f0 + 4.0f0im, 3.0f0 + 4.0f0im, 200) ≈ 200 * eps(Float32) * 5.0f0
    @test PureAdsorb.sk_audit_tolerance(3.0f0 + 4.0f0im, 3.0f0 + 4.0f0im, 200) > PureAdsorb.sk_audit_tolerance(3.0 + 4.0im, 3.0 + 4.0im, 200)
    # A near-zero structure factor still gets a nonzero tolerance (the `one(F)` floor).
    @test PureAdsorb.sk_audit_tolerance(0.0 + 0.0im, 0.0 + 0.0im, 50) ≈ 50 * eps(Float64)
end
