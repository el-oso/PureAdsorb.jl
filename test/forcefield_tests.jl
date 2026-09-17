@testitem "force field mixing and lookup" begin
    ff = ForceField(["A", "B"], [3.0, 4.0], [0.01, 0.04]; cutoff = 12.0)
    @test ff.sigma[1, 2] ≈ 3.5
    @test ff.epsilon[1, 2] ≈ 0.02
    @test PureAdsorb.typeindex(ff, "B") == 2
    @test_throws "unknown LJ type" PureAdsorb.typeindex(ff, "C")
    c = PureAdsorb.tail_coefficient(ff, 1, 2)
    @test c ≈ 0.02 * 3.5^3 * ((3.5 / 12)^9 / 3 - (3.5 / 12)^3)
end

@testitem "read trappe.yaml and co2.yaml" begin
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    @test ff.cutoff == 12.0 && ff.tail
    i = PureAdsorb.typeindex(ff, "Ac_")
    @test ff.sigma[i, i] ≈ 3.0985
    @test ff.epsilon[i, i] ≈ 0.0014311662224050347
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    @test length(g.sites) == 3
    @test g.charges == [0.7, -0.35, -0.35]
    @test g.sites[2] ≈ [-1.16, 0, 0]
    @test g.tc ≈ 303.75
    @test g.pc == 7.84e6
end

@testitem "tail delta for a ghost guest" begin
    ff = ForceField(["A", "B"], [3.0, 4.0], [0.01, 0.04]; cutoff = 12.0)
    V = 1000.0
    counts = [10, 0]          # host: 10 A atoms
    guest = [0, 2]            # guest: 2 B sites
    c = (i, j) -> PureAdsorb.tail_coefficient(ff, i, j)
    expected = (8π / 3 / V) * (2 * (10 * 2 * c(1, 2)) + 4 * c(2, 2))
    @test PureAdsorb.tail_delta(ff, counts, guest, V) ≈ expected
end
