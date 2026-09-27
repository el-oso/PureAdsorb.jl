@testsnippet ExchangeOracle begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions

    function rubtak_co2_exchange_setup(::Type{F}; ncounts = [0], capacities = [4], seed = 33) where {F}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
        b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
        st = PureAdsorb.SystemState(b, g, ncounts, ff; T = F(298.15), seed, capacities)
        guest_c = PureAdsorb.compact_guest(b, g)
        N = length(g.sites)
        guest_types = SVector{N, Int}(b.guest_types)
        return b, st, g, guest_c, guest_types, ff
    end

    # A single LJ type (epsilon=0) shared by host and guest, and zero guest charges: every
    # real-space, guest-guest and reciprocal interaction is EXACTLY zero, so a chain's loading is
    # governed only by the μVT combinatorial prefactor -- an ideal gas. `tail=false` also zeroes
    # `exchange_constant_term`'s tail piece.
    function ideal_gas_setup(::Type{F}; L = F(20), capacities = [25]) where {F}
        cell = SMatrix{3, 3, F}(L * I)
        fw = PureAdsorb.Framework{F}(cell, [SVector(F(0.5), F(0.5), F(0.5))], ["X1"], ["X"], [F(0)])
        ffree = PureAdsorb.ForceField(["X_"], [F(1)], [F(0)]; cutoff = F(8), tail = false)
        gg = PureAdsorb.Guest(SVector{1}(SVector(F(0), F(0), F(0))), SVector(1), SVector(F(0)), F(1), F(1), F(0))
        ewald = EwaldParams(cutoff = F(8), precision = F(1.0e-6))
        b = FrameworkBatch([fw], ffree, gg, ewald; fullk = true)
        st = PureAdsorb.SystemState(b, gg, [0], ffree; T = F(298.15), seed = 11, capacities)
        guest_c = PureAdsorb.compact_guest(b, gg)
        guest_types = SVector{1, Int}(b.guest_types)
        return b, st, gg, guest_c, guest_types, ffree, F(L)^3
    end

    # Attempts `mc_insert!`/`mc_delete!` until the FIRST accepted move, returning its `ΔU` -- an
    # acceptance that never happens in `maxtries` attempts is a test-setup error (widen the
    # fugacity), not something to silently skip. `ff` is used once, here, to precompute the
    # affine coefficients `mc_insert!`/`mc_delete!` take as an argument rather than rederiving on
    # every call (job 1); `ws` (the same `MoveWorkspace` `mc_step!` reuses) is built once too.
    function insert_until_accept!(b, st, gc, gt, ff, fug::F, kT::F; backend = CPU(), maxtries = 500) where {F}
        p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
        ws = PureAdsorb.MoveWorkspace(F, st.nsys, PureAdsorb.default_nblocks_per_chain(F, st.nsys); backend)
        n0 = st.occupancy[1]
        for _ in 1:maxtries
            e0 = st.energy[1]
            PureAdsorb.mc_insert!(ws, b, st, gc, gt, p, q, F[fug], kT; backend)
            st.occupancy[1] == n0 + 1 && return st.energy[1] - e0
        end
        return error("insert_until_accept!: no insertion accepted in $maxtries attempts")
    end
    function delete_until_accept!(b, st, gc, gt, ff, fug::F, kT::F; backend = CPU(), maxtries = 500) where {F}
        p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
        ws = PureAdsorb.MoveWorkspace(F, st.nsys, PureAdsorb.default_nblocks_per_chain(F, st.nsys); backend)
        n0 = st.occupancy[1]
        for _ in 1:maxtries
            e0 = st.energy[1]
            PureAdsorb.mc_delete!(ws, b, st, gc, gt, p, q, F[fug], kT; backend)
            st.occupancy[1] == n0 - 1 && return st.energy[1] - e0
        end
        return error("delete_until_accept!: no deletion accepted in $maxtries attempts")
    end
end

@testitem "log_insertion_prefactor and log_deletion_prefactor are exact inverses" begin
    using Random
    rng = Xoshiro(1)
    for F in (Float64, Float32)
        for _ in 1:200
            f = F(rand(rng) * 10 + 1.0e-6)
            V = F(rand(rng) * 1000 + 1)
            kT = F(rand(rng) * 0.05 + 0.001)
            N = rand(rng, 0:50)
            # `A_ins(N -> N+1)` and its detailed-balance partner `A_del(N+1 -> N)` must be exact
            # negatives in log-space -- `log_deletion_prefactor` is DEFINED as the negative of
            # `log_insertion_prefactor` one guest down, so this pins that construction against
            # regression rather than re-deriving the same formula twice by hand.
            lhs = PureAdsorb.log_insertion_prefactor(f, V, kT, N)
            rhs = PureAdsorb.log_deletion_prefactor(f, V, kT, N + 1)
            @test iszero(lhs + rhs)
        end
    end
end

@testitem "exchange_constant_coeffs is affine in Ng and matches exchange_constant_term at every Ng" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [0])
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c, 1)
    for Ng in 0:6
        exact = PureAdsorb.exchange_constant_term(ff, b, guest_c, 1, Ng)
        @test p + q * Ng ≈ exact atol = 1.0e-10 * max(abs(exact), 1.0)
    end
    # `Ng` is a kernel ARGUMENT (`occupancy[n]`, read live at launch), not baked into `p`/`q`: two
    # chains with different current occupancy sharing the same `p`, `q` must still reproduce
    # `exchange_constant_term` at their OWN occupancy (E4's concern, stated as a property test
    # rather than trusted by inspection of the kernel body).
    @test p ≈ PureAdsorb.exchange_constant_term(ff, b, guest_c, 1, 0)
    @test (p + 5q) ≈ PureAdsorb.exchange_constant_term(ff, b, guest_c, 1, 5)
end

@testitem "an insertion and its exact-inverse deletion sum ΔU to zero and restore Sk (Float64)" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [0], capacities = [4], seed = 33)
    kT = PureAdsorb.KB * 298.15
    Sk0 = copy(st.Sk)
    dU_ins = insert_until_accept!(b, st, guest_c, guest_types, ff, 5.0e6, kT)
    @test st.occupancy[1] == 1   # the ONLY occupied slot: the next delete cannot pick any other guest
    dU_del = delete_until_accept!(b, st, guest_c, guest_types, ff, 1.0e-3, kT)
    @test iszero(st.occupancy[1])
    @test iszero(dU_ins + dU_del)
    @test isapprox(st.Sk, Sk0; atol = 1.0e-9)
    recomputed = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test st.energy[1] ≈ recomputed atol = 1.0e-9 * max(abs(recomputed), 1.0)
end

@testitem "an insertion and its exact-inverse deletion sum ΔU to zero and restore Sk (Float32)" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float32; ncounts = [0], capacities = [4], seed = 33)
    kT = Float32(PureAdsorb.KB * 298.15)
    Sk0 = copy(st.Sk)
    dU_ins = insert_until_accept!(b, st, guest_c, guest_types, ff, 5.0f6, kT)
    @test st.occupancy[1] == 1
    dU_del = delete_until_accept!(b, st, guest_c, guest_types, ff, 1.0f-3, kT)
    @test iszero(st.occupancy[1])
    @test abs(dU_ins + dU_del) <= 1.0f-4 * max(abs(dU_ins), 1.0f0)
    @test isapprox(st.Sk, Sk0; atol = 1.0f-3)
end

@testitem "mc_delete! on an empty chain is a no-op" setup = [ExchangeOracle] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [0], capacities = [4], seed = 5)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    before = deepcopy(st)
    PureAdsorb.mc_delete!(ws, b, st, guest_c, guest_types, p, q, [1.0], kT)   # must not throw
    @test iszero(st.occupancy[1])
    @test st.energy[1] == before.energy[1]
    @test st.Sk == before.Sk
    @test st.rng_counter[1] == before.rng_counter[1] + 1
end

@testitem "mc_insert! throws when a physically-accepted move is forced to reject at capacity" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [0], capacities = [3], seed = 5)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    @test_throws "hit capacity" begin
        for _ in 1:50
            PureAdsorb.mc_insert!(ws, b, st, guest_c, guest_types, p, q, [1.0e12], kT)
        end
    end
end

@testitem "capacity_hits aborts an ideal-gas chain rather than truncating its distribution" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff, V = ideal_gas_setup(Float64; capacities = [25])
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    fV_over_kT = 20.0   # <N> = fV/kT for an ideal gas: comfortably above capacity=25's tail
    fugacity_pa = fV_over_kT * kT / V / PureAdsorb.PASCAL
    rng = Xoshiro(1)
    @test_throws "hit capacity" begin
        for _ in 1:500
            PureAdsorb.mc_exchange!(rng, ws, b, st, guest_c, guest_types, p, q, [fugacity_pa], kT)
        end
    end
    @test st.occupancy[1] == 25   # aborted AT capacity, not merely rejected forever below it
end

@testitem "mc_exchange! draws roughly a fair 50/50 split of insertions and deletions" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [3], capacities = [40], seed = 8)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    rng = Xoshiro(1234)
    ncalls = 400
    n_ins = count(
        _ -> PureAdsorb.mc_exchange!(rng, ws, b, st, guest_c, guest_types, p, q, [2.0e4], kT),
        1:ncalls
    )
    # p_ins = p_del = 1/2 is what log_insertion_prefactor/log_deletion_prefactor assume; a 4-sigma
    # band around n=ncalls/2 (binomial std = sqrt(ncalls)/2 ≈ 10) flags any systematic bias, not
    # just an unlucky draw.
    @test abs(n_ins - ncalls / 2) < 4 * sqrt(ncalls) / 2
end

@testitem "energy audit passes over a long mc_exchange!-driven chain and still catches an injected corruption (Float64)" setup = [
    ExchangeOracle,
] begin
    b, st, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [3], capacities = [15], seed = 42)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    rng = Xoshiro(7)
    for _ in 1:600
        PureAdsorb.mc_exchange!(rng, ws, b, st, guest_c, guest_types, p, q, [2.0e4], kT)
    end
    naccept_ish = 300   # attempts, not accepted moves -- a loose over-estimate is fine for a tolerance scale
    PureAdsorb.audit_energy!(b, st, g, ff, 1, naccept_ish)   # must not throw
    recomputed = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test st.energy[1] == recomputed

    st.energy[1] += 1.0e-3
    @test_throws "energy audit failed" PureAdsorb.audit_energy!(b, st, g, ff, 1, 1)
end

@testitem "mc_insert!/mc_delete! agree between CPU and CUDA" tags = [:gpu] setup = [ExchangeOracle] begin
    using KernelAbstractions
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

    b, st_cpu, g, guest_c, guest_types, ff = rubtak_co2_exchange_setup(Float64; ncounts = [3], capacities = [10], seed = 3)
    st_gpu = deepcopy(st_cpu)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    nbpc = PureAdsorb.default_nblocks_per_chain(Float64, st_cpu.nsys)
    ws_cpu = PureAdsorb.MoveWorkspace(Float64, st_cpu.nsys, nbpc; backend = CPU())
    ws_gpu = PureAdsorb.MoveWorkspace(Float64, st_cpu.nsys, nbpc; backend)
    db = PureAdsorb.adapt(backend, b)
    dst = PureAdsorb.adapt(backend, st_gpu)
    dp = PureAdsorb.adapt(backend, p); dq = PureAdsorb.adapt(backend, q)
    kT = PureAdsorb.KB * 298.15
    rng_cpu = Xoshiro(99)
    rng_gpu = Xoshiro(99)

    for _ in 1:200
        PureAdsorb.mc_exchange!(rng_cpu, ws_cpu, b, st_cpu, guest_c, guest_types, p, q, [2.0e4], kT; backend = CPU())
        PureAdsorb.mc_exchange!(rng_gpu, ws_gpu, db, dst, guest_c, guest_types, dp, dq, [2.0e4], kT; backend)
    end

    @test Array(dst.occupancy) == st_cpu.occupancy
    @test all(isapprox.(Array(dst.refpoints)[1:st_cpu.occupancy[1]], st_cpu.refpoints[1:st_cpu.occupancy[1]]; rtol = 1.0e-8))
    @test Array(dst.energy) ≈ st_cpu.energy rtol = 1.0e-8
    @test Array(dst.Sk) ≈ st_cpu.Sk rtol = 1.0e-8
end
