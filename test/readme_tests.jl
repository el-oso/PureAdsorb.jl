@testitem "README example runs" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    co2 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    batch = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, co2, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    res = widom(batch, co2; T = 298.15, ninsert = 2_000, nblocks = 4, seed = 42)
    @test isfinite(res[1].K_H) && isfinite(res[1].mu_ex) && isfinite(res[1].q_st)
end
