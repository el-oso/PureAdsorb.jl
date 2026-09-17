@testitem "GPU matches CPU on the same poses" tags = [:gpu] begin
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
end
