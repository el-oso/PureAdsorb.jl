@testitem "GPU matches CPU on the same poses" tags = [:gpu] begin
    using StaticArrays, Random, KernelAbstractions
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
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    rc = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4)[1]
    rg = widom(b, g; T = 298.15, ninsert = 20_000, seed = 5, nblocks = 4, backend)[1]
    @test rg.mu_ex ≈ rc.mu_ex rtol = 1.0e-10
    @test rg.K_H ≈ rc.K_H rtol = 1.0e-10
    @test rg.q_st ≈ rc.q_st rtol = 1.0e-10

    # Phase 0 (hardcore_kernel!) on its own, bit-exact against the CPU backend: `widom`'s
    # end-to-end agreement above could in principle mask a phase-0 discrepancy that phase 1
    # happens to compensate, so this checks the rejection flags directly.
    N = length(g.sites)
    g_compact = PureAdsorb.Guest{Float64, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
    kT = PureAdsorb.KB * 298.15
    rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, g_compact, kT)
    nposes = 20_000
    rng = Xoshiro(5)
    sys_of = Vector{Int32}(undef, nposes)
    rpos = Vector{SVector{3, Float64}}(undef, nposes)
    quat = Vector{SVector{4, Float64}}(undef, nposes)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, PureAdsorb.default_run(nposes, b.nsys), b.nsys)
    flags_cpu = zeros(UInt8, nposes)
    PureAdsorb.hardcore_kernel!(CPU())(
        flags_cpu, sys_of, rpos, quat, b, g_compact, rho2, reach0, Int32(ntypes); ndrange = nposes
    )
    KernelAbstractions.synchronize(CPU())
    dbatch = PureAdsorb.adapt(backend, b)
    dsys, drpos, dquat = PureAdsorb.adapt(backend, sys_of), PureAdsorb.adapt(backend, rpos), PureAdsorb.adapt(backend, quat)
    drho2, dreach0 = PureAdsorb.adapt(backend, rho2), PureAdsorb.adapt(backend, reach0)
    dflags = PureAdsorb.adapt(backend, zeros(UInt8, nposes))
    PureAdsorb.hardcore_kernel!(backend)(
        dflags, dsys, drpos, dquat, dbatch, g_compact, drho2, dreach0, Int32(ntypes); ndrange = nposes
    )
    KernelAbstractions.synchronize(backend)
    @test Array(dflags) == flags_cpu
end
