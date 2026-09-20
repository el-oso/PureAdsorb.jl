@testitem "read RUBTAK.cif" begin
    using StaticArrays
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    @test PureAdsorb.natoms(fw) == 114
    @test fw.symbols[1] == "Zr"
    @test fw.charges[1] ≈ 2.38565
    @test fw.frac[1] ≈ SVector(0.37986, 0.37998, 0.61969)
    @test abs(PureAdsorb.total_charge(fw)) < 1.0e-3
    @test fw.cell ≈ PureAdsorb.cell_matrix(14.7619, 14.80147, 14.76539, 59.84578, 60.04729, 59.8131)
    @test fw.replication == (1, 1, 1)
end

@testitem "replicate preserves counts and charge" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    sc = replicate(fw, (3, 3, 3))
    @test PureAdsorb.natoms(sc) == 27 * 114
    @test sc.cell ≈ 3 * fw.cell
    @test PureAdsorb.total_charge(sc) ≈ 27 * PureAdsorb.total_charge(fw) atol = 1.0e-9
    @test all(0 .<= reduce(vcat, collect.(sc.frac)) .< 1)
    @test sc.replication == (3, 3, 3)
end

@testitem "replication factors compose across successive replicate calls" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    sc = replicate(replicate(fw, (2, 1, 1)), (1, 3, 2))
    @test sc.replication == (2, 3, 2)
end

@testitem "read_cif rejects non-P1 and chargeless files" begin
    dir = mktempdir()
    p = joinpath(dir, "bad.cif")
    write(p, "data_x\n_symmetry_space_group_name_H-M 'P 21'\nloop_\n_atom_site_label\n_atom_site_fract_x\n")
    @test_throws "P1" read_cif(p)
    src = read(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"), String)
    write(p, replace(src, "_atom_site_charge\n" => ""))
    @test_throws "_atom_site_charge" read_cif(p)
end
