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

@testitem "energy_audit_tolerance scales with nmoves and precision" begin
    @test PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200) ≈ 200 * eps(Float64) * 10.0
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) ≈ 200 * eps(Float32) * 10.0f0
    # Float32's tolerance is looser than Float64's at the same (nmoves, energy scale): eps(F32) ≫
    # eps(F64), which is the whole reason Float32 accumulation is the case worth testing directly.
    @test PureAdsorb.energy_audit_tolerance(10.0f0, 10.0f0, 200) > PureAdsorb.energy_audit_tolerance(10.0, 10.0, 200)
    # A near-zero energy still gets a nonzero tolerance (the `one(F)` floor).
    @test PureAdsorb.energy_audit_tolerance(0.0, 0.0, 50) ≈ 50 * eps(Float64)
end
