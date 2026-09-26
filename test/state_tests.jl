@testitem "SystemState requires a fullk batch" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b_sparse = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    @test_throws "fullk=true" PureAdsorb.SystemState(b_sparse, g, [1]; T = 300.0)
end

@testitem "SystemState rejects a mismatched guest count or a mismatched guest" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    @test_throws DimensionMismatch PureAdsorb.SystemState(b, g, [1]; T = 300.0)
    g_wrong = PureAdsorb.Guest(g.sites, g.types, g.charges .+ 1, g.tc, g.pc, g.omega)
    @test_throws "does not match the guest" PureAdsorb.SystemState(b, g_wrong, [1, 1]; T = 300.0)
end

@testitem "SystemState builds a ragged layout, one running structure factor entry per k-vector" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    ncounts = [5, 0, 2]
    st = PureAdsorb.SystemState(b, g, ncounts; T = 298.15, seed = 3)
    @test st.nsys == 3
    for n in 1:3
        @test length(PureAdsorb.guest_range(st, n)) == ncounts[n]
        @test PureAdsorb.nguests(st, n) == ncounts[n]
        @test length(PureAdsorb.kvec_range(st, n)) == b.k_offsets[n + 1] - b.k_offsets[n]
    end
    @test PureAdsorb.guest_range(st, 1) == 1:5
    @test PureAdsorb.guest_range(st, 2) == 6:5   # empty: system 2 has no guests
    @test PureAdsorb.guest_range(st, 3) == 6:7
    @test length(st.refpoints) == length(st.orientations) == sum(ncounts)
    @test length(st.Sk) == length(b.Shost)
    @test all(iszero, st.energy)
    @test all(iszero, st.rng_counter)
    @test length(unique(st.rng_seed)) == 3   # distinct chains get distinct seeds
    @test all(==(zero(SVector{3, Int32})), st.accepted)
    @test all(==(zero(SVector{3, Int32})), st.attempted)
end

@testitem "a system with no guests keeps the running structure factor equal to the host's" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    st = PureAdsorb.SystemState(b, g, [0, 0]; T = 300.0)
    @test isempty(st.refpoints)
    @test isempty(st.orientations)
    @test st.Sk == b.Shost
end

@testitem "initial placement clears widom's own hard-core rejection test against the host" begin
    using LinearAlgebra, StaticArrays
    for T in (Float64, Float32)
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = T)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = T)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = T)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = T(12), precision = T(1.0e-6))
        b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = true)
        Tk = T(298.15)
        st = PureAdsorb.SystemState(b, g, [4, 3]; T = Tk, seed = 11)
        kT = T(PureAdsorb.KB) * Tk
        N = length(g.sites)
        g_compact = PureAdsorb.Guest{T, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
        rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, g_compact, kT)

        ntot = length(st.refpoints)
        sys_of = Int32[n for n in 1:b.nsys for _ in PureAdsorb.guest_range(st, n)]
        rpos = [inv(b.cells[sys_of[i]]) * st.refpoints[i] for i in 1:ntot]
        flags = zeros(UInt8, ntot)
        PureAdsorb.hardcore_kernel!(PureAdsorb.CPU())(
            flags, sys_of, rpos, st.orientations, b, g_compact, rho2, reach0, Int32(ntypes); ndrange = ntot
        )
        @test all(iszero, flags)
    end
end

@testitem "the running structure factor is the host term plus every guest's own" begin
    using LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    st = PureAdsorb.SystemState(b, g, [3, 2]; T = 298.15, seed = 5)
    for n in 1:2
        kr = PureAdsorb.kvec_range(st, n)
        expected = copy(b.Shost[kr])
        for i in PureAdsorb.guest_range(st, n)
            gsites = [st.refpoints[i] + PureAdsorb.rotate(st.orientations[i], s) for s in g.sites]
            expected .+= PureAdsorb.structure_factor(view(b.ks, kr), gsites, g.charges)
        end
        # Summed in a different order than the constructor (per guest here, over every guest's
        # every site there), so only mathematically, not bit-for-bit, equal.
        @test st.Sk[kr] ≈ expected
    end
end

@testitem "SystemState construction is reproducible for a fixed seed, independent runs" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    st1 = PureAdsorb.SystemState(b, g, [4, 4]; T = 298.15, seed = 42)
    st2 = PureAdsorb.SystemState(b, g, [4, 4]; T = 298.15, seed = 42)
    st3 = PureAdsorb.SystemState(b, g, [4, 4]; T = 298.15, seed = 43)
    @test st1.refpoints == st2.refpoints
    @test st1.orientations == st2.orientations
    @test st1.rng_seed == st2.rng_seed
    @test st1.refpoints != st3.refpoints
    @test st1.rng_seed != st3.rng_seed
end

@testitem "SystemState round-trips through adapt on CPU unchanged" begin
    using KernelAbstractions
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3]; T = 298.15, seed = 9)
    st2 = PureAdsorb.adapt(CPU(), st)
    @test typeof(st2) == typeof(st)
    for f in fieldnames(typeof(st))
        @test getfield(st2, f) == getfield(st, f)
    end
end

@testitem "SystemState round-trips through adapt on the GPU unchanged" tags = [:gpu] begin
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
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6); fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3]; T = 298.15, seed = 9)
    dst = PureAdsorb.adapt(backend, st)
    @test !(dst.refpoints isa Array)   # actually moved to the device, not a no-op adapt
    back = PureAdsorb.adapt(CPU(), dst)
    for f in fieldnames(typeof(st))
        @test getfield(back, f) == getfield(st, f)
    end
end
