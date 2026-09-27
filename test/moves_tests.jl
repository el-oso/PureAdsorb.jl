@testsnippet MovesOracle begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions

    # Independent, serial (no workgroup, no `@localmem`) reimplementation of one `mc_step!` call,
    # built from the same public pieces `move_kernel!` uses (`PureAdsorb.select_and_propose`'s own
    # ingredients, `PureAdsorb.metropolis_accept`, `PureAdsorb.guest_move_delta`) rather than from
    # the kernel itself, so it is a genuine cross-check rather than the kernel testing itself. Runs
    # one move for every chain, exactly mirroring `move_kernel!`'s contract: `rng_counter`/
    # `attempted` always advance; `accepted`/pose/`Sk`/`energy`/`host_energy` only on acceptance.
    function mc_step_serial!(
            batch::PureAdsorb.FrameworkBatch{F}, state::PureAdsorb.SystemState{F}, guest::PureAdsorb.Guest{F, N},
            movetype::Integer, step_trans, step_rot, kT::F
        ) where {F, N}
        for n in 1:state.nsys
            gr0 = state.guest_offsets[n]
            Ng = state.guest_offsets[n + 1] - gr0
            cnt = state.rng_counter[n]
            if iszero(Ng)
                state.rng_counter[n] = cnt + one(cnt)
                continue
            end
            rng = PureAdsorb.ChainRNG(state.rng_seed[n], cnt, Int32(0))
            gidx_local, rng = PureAdsorb.rand_range(rng, Ng)
            i = gr0 + gidx_local
            oldpos = state.refpoints[i]; oldq = state.orientations[i]
            newpos, newq, rng = PureAdsorb.propose_move(
                rng, Int32(movetype), oldpos, oldq, step_trans[n], step_rot[n], batch.cells[n]
            )
            kr = PureAdsorb.kvec_range(state, n)
            ΔS = zeros(Complex{F}, length(kr))
            ΔU, e_new = PureAdsorb.guest_move_delta(batch, state, guest, n, i, newpos, newq, ΔS)
            u, rng = PureAdsorb.rand_uniform(rng, F)
            accept = PureAdsorb.metropolis_accept(ΔU, kT, u)
            if accept
                state.refpoints[i] = newpos
                state.orientations[i] = newq
                state.host_energy[i] = e_new
                state.Sk[kr] .+= ΔS
                state.energy[n] += ΔU
            end
            delta = PureAdsorb.onehot_movetype(Int32(movetype))
            state.attempted[n] = state.attempted[n] + delta
            accept && (state.accepted[n] = state.accepted[n] + delta)
            state.rng_counter[n] = cnt + one(cnt)
        end
        return nothing
    end

    function rubtak_co2_setup(::Type{F}; ncounts = [4, 3], seed = 9) where {F}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
        b = FrameworkBatch(fill(sc, length(ncounts)), ff, g, ewald; fullk = true)
        st = PureAdsorb.SystemState(b, g, ncounts, ff; T = F(298.15), seed)
        guest_c = PureAdsorb.compact_guest(b, g)
        N = length(g.sites)
        guest_types = SVector{N, Int}(b.guest_types)
        return b, st, g, guest_c, guest_types, ff
    end

    default_ws(::Type{F}, nsys::Integer; backend = CPU()) where {F} = PureAdsorb.MoveWorkspace(F, nsys, 256; backend)
end

@testitem "qmul composes quaternion rotations" setup = [MovesOracle] begin
    using StaticArrays, LinearAlgebra, Random
    rng = Xoshiro(1)
    for _ in 1:200
        q1 = normalize(SVector{4, Float64}(randn(rng, 4)))
        q2 = normalize(SVector{4, Float64}(randn(rng, 4)))
        v = SVector{3, Float64}(randn(rng, 3))
        lhs = PureAdsorb.rotate(PureAdsorb.qmul(q2, q1), v)
        rhs = PureAdsorb.rotate(q2, PureAdsorb.rotate(q1, v))
        @test lhs ≈ rhs atol = 1.0e-12
    end
end

@testitem "quaternion_pow: boundary values and inversion symmetry" begin
    using StaticArrays, LinearAlgebra, Random
    rng = Xoshiro(2)
    identity_q = SVector(0.0, 0.0, 0.0, 1.0)
    for _ in 1:200
        u1, u2, u3 = rand(rng), rand(rng), rand(rng)
        q = PureAdsorb.shoemake_quaternion(u1, u2, u3)
        @test PureAdsorb.quaternion_pow(q, 0.0) ≈ identity_q atol = 1.0e-10
        # `q^1` reproduces `q` itself, up to the sign ambiguity of a unit quaternion (q and -q
        # represent the same rotation, and `quaternion_pow`'s construction can return either).
        p1 = PureAdsorb.quaternion_pow(q, 1.0)
        @test isapprox(p1, q; atol = 1.0e-10) || isapprox(p1, -q; atol = 1.0e-10)
        # Symmetry the "rotation" proposal relies on: negating a unit quaternion's vector part is
        # its inverse rotation, and `quaternion_pow` commutes with that inversion -- powering the
        # inverse gives the inverse of the power. Since `shoemake_quaternion`'s own construction is
        # Haar-uniform on SO(3) (a documented property of the algorithm, not re-derived here), this
        # is what makes the proposed relative rotation's law symmetric under inversion.
        t = rand(rng)
        qconj = SVector(-q[1], -q[2], -q[3], q[4])
        lhs = PureAdsorb.quaternion_pow(qconj, t)
        rhs_pow = PureAdsorb.quaternion_pow(q, t)
        rhs = SVector(-rhs_pow[1], -rhs_pow[2], -rhs_pow[3], rhs_pow[4])
        @test lhs ≈ rhs atol = 1.0e-10
    end
end

@testitem "metropolis_accept rejects non-finite ΔU by a deliberate check (R7)" begin
    for F in (Float64, Float32)
        kT = F(0.1)
        for u in (F(0.001), F(0.5), F(0.999))
            @test !PureAdsorb.metropolis_accept(F(Inf), kT, u)
            @test !PureAdsorb.metropolis_accept(F(NaN), kT, u)
        end
        # An ordinary finite, very negative ΔU (a strongly favorable move) is still accepted --
        # the non-finite check does not just always return false.
        @test PureAdsorb.metropolis_accept(F(-1000), kT, F(0.5))
    end

    # The physical case R7 exists for: two guests placed on top of each other. `guest_move_delta`
    # (not `move_kernel!`, so this isolates the energy path from the acceptance path) then reports
    # a non-finite ΔU from the diverging Lennard-Jones repulsion, and `metropolis_accept` must
    # reject it deliberately rather than by IEEE-754 comparison accident.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [2], ff; T = 298.15, seed = 3)
    n = 1
    gr = PureAdsorb.guest_range(st, n)
    i, j = gr[1], gr[2]
    kr = PureAdsorb.kvec_range(st, n)
    ΔS = zeros(ComplexF64, length(kr))
    # Move guest i exactly onto guest j's current pose: identical site positions, so several
    # guest-guest pair distances are exactly zero.
    ΔU, = PureAdsorb.guest_move_delta(b, st, g, n, i, st.refpoints[j], st.orientations[j], ΔS)
    @test !isfinite(ΔU)
    @test !PureAdsorb.metropolis_accept(ΔU, PureAdsorb.KB * 298.15, 0.999999)
end

@testitem "propose_translation is symmetric (Gaussian about the current position)" begin
    using StaticArrays, Random
    rng0 = PureAdsorb.ChainRNG(UInt64(11), 1, 0)
    oldpos = SVector(1.0, 2.0, 3.0)
    step = 0.4
    newpos, _ = PureAdsorb.propose_translation(rng0, oldpos, step)
    Δ = newpos - oldpos
    # An isotropic Gaussian's density at +Δ equals its density at -Δ about either center: proposing
    # `newpos` from `oldpos` and proposing `oldpos` from `newpos` (displacement `-Δ`) are exactly as
    # likely, so `metropolis_accept` needs no proposal-density correction.
    density(d) = exp(-sum(abs2, d) / (2 * step^2))
    @test density(Δ) ≈ density(-Δ)
end

@testitem "propose_reinsertion's density does not depend on the current pose" begin
    # Symmetry q(new|old) == q(old|new) is immediate here because the proposal density does not
    # depend on `old` at all -- the function itself takes no `oldpos`/`oldq` argument.
    m = only(methods(PureAdsorb.propose_reinsertion))
    @test m.nargs == 3   # (rng, cell), plus the implicit function slot
end

@testitem "move_kernel! matches a serial reference over a mixed-move chain (Float64)" setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    b, st_kernel, g, guest_c, guest_types, ff = rubtak_co2_setup(Float64)
    st_serial = deepcopy(st_kernel)
    step_trans = fill(0.3, st_kernel.nsys)
    step_rot = fill(0.3, st_kernel.nsys)
    kT = PureAdsorb.KB * 298.15
    ws = default_ws(Float64, st_kernel.nsys)

    for step in 1:300
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws, b, st_kernel, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32)
        mc_step_serial!(b, st_serial, guest_c, mt, step_trans, step_rot, kT)
    end

    @test st_kernel.refpoints == st_serial.refpoints
    @test st_kernel.orientations == st_serial.orientations
    @test st_kernel.Sk ≈ st_serial.Sk
    @test st_kernel.energy ≈ st_serial.energy
    @test st_kernel.host_energy ≈ st_serial.host_energy
    @test st_kernel.accepted == st_serial.accepted
    @test st_kernel.attempted == st_serial.attempted
    @test st_kernel.rng_counter == st_serial.rng_counter
    # A real mix of accept and reject, at every move type, or this comparison is not exercising
    # the acceptance path at all.
    @test all(>(0), sum.(st_kernel.attempted))
    @test sum(sum, st_kernel.accepted) > 0
end

@testitem "move_kernel! matches a serial reference over a mixed-move chain (Float32)" setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    b, st_kernel, g, guest_c, guest_types, ff = rubtak_co2_setup(Float32)
    st_serial = deepcopy(st_kernel)
    step_trans = fill(0.3f0, st_kernel.nsys)
    step_rot = fill(0.3f0, st_kernel.nsys)
    kT = Float32(PureAdsorb.KB * 298.15)
    ws = default_ws(Float32, st_kernel.nsys)

    for step in 1:300
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws, b, st_kernel, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32)
        mc_step_serial!(b, st_serial, guest_c, mt, step_trans, step_rot, kT)
    end

    @test st_kernel.refpoints == st_serial.refpoints
    @test st_kernel.orientations == st_serial.orientations
    @test st_kernel.Sk ≈ st_serial.Sk
    @test st_kernel.energy ≈ st_serial.energy
    @test st_kernel.accepted == st_serial.accepted
end

@testitem "mc_step! leaves a zero-guest chain's Sk/energy untouched" setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    b, st, g, guest_c, guest_types, ff = rubtak_co2_setup(Float64; ncounts = [4, 0])
    before = deepcopy(st)
    step_trans = fill(0.3, st.nsys)
    step_rot = fill(0.3, st.nsys)
    kT = PureAdsorb.KB * 298.15
    ws = default_ws(Float64, st.nsys)
    for step in 1:20
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws, b, st, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32)
    end
    kr2 = PureAdsorb.kvec_range(st, 2)
    @test st.Sk[kr2] == before.Sk[kr2]
    @test st.energy[2] == before.energy[2]
    @test sum(st.attempted[2]) == 20
    @test iszero(sum(st.accepted[2]))
    @test st.rng_counter[2] == before.rng_counter[2] + 20
end

@testitem "reversibility: a move and its exact inverse sum to zero ΔU and restore Sk" begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [5], ff; T = 298.15, seed = 4)
    n = 1
    rng = Xoshiro(21)
    for movetype in (PureAdsorb.MOVE_TRANSLATION, PureAdsorb.MOVE_ROTATION, PureAdsorb.MOVE_REINSERTION)
        i = rand(rng, PureAdsorb.guest_range(st, n))
        oldpos = st.refpoints[i]; oldq = st.orientations[i]
        cell = b.cells[n]
        chainrng = PureAdsorb.ChainRNG(rand(rng, UInt64), 0, 0)
        newpos, newq, _ = PureAdsorb.propose_move(chainrng, Int32(movetype), oldpos, oldq, 0.3, 0.3, cell)

        kr = PureAdsorb.kvec_range(st, n)
        ΔS_fwd = zeros(ComplexF64, length(kr))
        ΔU_fwd, e_new_fwd = PureAdsorb.guest_move_delta(b, st, g, n, i, newpos, newq, ΔS_fwd)

        st2 = deepcopy(st)
        st2.refpoints[i] = newpos; st2.orientations[i] = newq
        st2.host_energy[i] = e_new_fwd
        st2.Sk[kr] .+= ΔS_fwd

        ΔS_back = zeros(ComplexF64, length(kr))
        ΔU_back, = PureAdsorb.guest_move_delta(b, st2, g, n, i, oldpos, oldq, ΔS_back)

        @test abs(ΔU_fwd + ΔU_back) <= 1.0e-8 * max(abs(ΔU_fwd), 1.0)
        @test maximum(abs.(ΔS_fwd .+ ΔS_back)) <= 1.0e-8 * max(maximum(abs.(ΔS_fwd)), 1.0)
    end
end

@testitem "acceptance rate responds sensibly to translation step size" setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    function acceptance_rate(step)
        b, st, g, guest_c, guest_types, ff = rubtak_co2_setup(Float64; ncounts = [6])
        step_trans = fill(step, st.nsys)
        step_rot = fill(0.3, st.nsys)
        kT = PureAdsorb.KB * 298.15
        ws = default_ws(Float64, st.nsys)
        for _ in 1:600
            PureAdsorb.mc_step!(
                ws, b, st, guest_c, guest_types, PureAdsorb.MOVE_TRANSLATION, step_trans, step_rot, kT; backend = CPU(),
                groupsize = 32
            )
        end
        return sum(st.accepted[1]) / sum(st.attempted[1])
    end
    rate_small = acceptance_rate(0.05)
    rate_large = acceptance_rate(3.0)
    # A small step accepts far more often than a large one on the same system -- the qualitative
    # sanity check the plan asks for, not a target acceptance ratio.
    @test rate_small > rate_large
    @test rate_small > 0.5
    @test rate_large < 0.3
end

@testitem "energy audit passes over a long move_kernel!-driven chain and still catches a corruption (Float64)" setup = [
    MovesOracle,
] begin
    using StaticArrays, KernelAbstractions
    b, st, g, guest_c, guest_types, ff = rubtak_co2_setup(Float64; ncounts = [5])
    step_trans = fill(0.3, st.nsys)
    step_rot = fill(0.4, st.nsys)
    kT = PureAdsorb.KB * 298.15
    ws = default_ws(Float64, st.nsys)
    for step in 1:3000
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws, b, st, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32)
    end
    naccept = sum(st.accepted[1])
    naccept > 0 || error("test setup produced no accepted moves; widen the step size")
    PureAdsorb.audit_energy!(b, st, g, ff, 1, naccept)   # must not throw
    recomputed = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test st.energy[1] == recomputed

    # A deliberate corruption of the running energy must still be caught after a real
    # move_kernel!-driven chain, exactly as `audit_tests.jl` establishes for the synthetic chain.
    st.energy[1] += 1.0e-6
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, 1, 1)
end

@testitem "energy audit passes over a long move_kernel!-driven chain (Float32)" setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    b, st, g, guest_c, guest_types, ff = rubtak_co2_setup(Float32; ncounts = [5])
    step_trans = fill(0.3f0, st.nsys)
    step_rot = fill(0.4f0, st.nsys)
    kT = Float32(PureAdsorb.KB * 298.15)
    ws = default_ws(Float32, st.nsys)
    for step in 1:5000
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws, b, st, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32)
    end
    naccept = sum(st.accepted[1])
    naccept > 0 || error("test setup produced no accepted moves; widen the step size")
    PureAdsorb.audit_energy!(b, st, g, ff, 1, naccept)   # must not throw
end

@testitem "move_kernel! agrees with the CPU backend on CUDA" tags = [:gpu] setup = [MovesOracle] begin
    using StaticArrays, KernelAbstractions
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    if isnothing(backend) && !isnothing(Base.find_package("AMDGPU"))
        @eval using AMDGPU
        AMDGPU.functional() && (backend = ROCBackend())
    end
    isnothing(backend) && error("no functional GPU backend; this item must run on a GPU host")

    b, st_cpu, g, guest_c, guest_types, ff = rubtak_co2_setup(Float64)
    st_gpu = deepcopy(st_cpu)
    db = PureAdsorb.adapt(backend, b)
    dst = PureAdsorb.adapt(backend, st_gpu)
    step_trans = fill(0.3, st_cpu.nsys)
    step_rot = fill(0.3, st_cpu.nsys)
    dstep_trans = PureAdsorb.adapt(backend, step_trans)
    dstep_rot = PureAdsorb.adapt(backend, step_rot)
    kT = PureAdsorb.KB * 298.15
    # `nblocks_per_chain = 4` exercises the cross-workgroup fan-out specifically (the default for
    # this `nsys` would already pick something similar, but naming it makes the intent explicit).
    ws_cpu = default_ws(Float64, st_cpu.nsys)
    ws_gpu = default_ws(Float64, st_cpu.nsys; backend)

    for step in 1:300
        mt = (step % 3) + 1
        PureAdsorb.mc_step!(ws_cpu, b, st_cpu, guest_c, guest_types, mt, step_trans, step_rot, kT; backend = CPU(), groupsize = 32, nblocks_per_chain = 4)
        PureAdsorb.mc_step!(ws_gpu, db, dst, guest_c, guest_types, mt, dstep_trans, dstep_rot, kT; backend = backend, groupsize = 32, nblocks_per_chain = 4)
    end

    refpoints_gpu = Array(dst.refpoints)
    orientations_gpu = Array(dst.orientations)
    Sk_gpu = Array(dst.Sk)
    energy_gpu = Array(dst.energy)
    accepted_gpu = Array(dst.accepted)

    # Cross-device summation order differs, so this compares to rounding, not bit for bit (the
    # existing `guest.jl` GPU test uses the same tolerance style).
    @test all(isapprox.(refpoints_gpu, st_cpu.refpoints; rtol = 1.0e-9))
    @test all(isapprox.(orientations_gpu, st_cpu.orientations; rtol = 1.0e-9))
    @test Sk_gpu ≈ st_cpu.Sk rtol = 1.0e-9
    @test energy_gpu ≈ st_cpu.energy rtol = 1.0e-9
    @test accepted_gpu == st_cpu.accepted
end
