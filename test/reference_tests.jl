@testitem "RUBTAK CO2 matches kUPS within combined error" tags = [:slow] begin
    using JSON
    ref = JSON.parsefile(joinpath(pkgdir(PureAdsorb), "test", "reference", "rubtak_co2_kups.json"))
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    b = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    r = widom(b, g; T = 298.15, ninsert = 1_000_000, seed = 42, nblocks = 20)[1]
    for (ours, err, key) in ((r.mu_ex, r.mu_ex_err, "mu_ex"), (r.K_H, r.K_H_err, "K_H"), (r.q_st, r.q_st_err, "q_st"))
        m, s = ref[key]["mean"], ref[key]["sem"]
        @test abs(ours - m) < 3 * hypot(err, s)
    end
end
