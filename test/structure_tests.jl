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

@testitem "parse_symop parses CIF symmetry-operation strings" begin
    using StaticArrays
    @test PureAdsorb.parse_symop("x,y,z")(SVector(0.1, 0.2, 0.3)) ≈ SVector(0.1, 0.2, 0.3)
    @test PureAdsorb.parse_symop("-x,-y,z")(SVector(0.1, 0.2, 0.3)) ≈ SVector(0.9, 0.8, 0.3)
    @test PureAdsorb.parse_symop("-x+1/2,y,-z+1/2")(SVector(0.1, 0.2, 0.3)) ≈ SVector(0.4, 0.2, 0.2)
    @test PureAdsorb.parse_symop("1/2-y,1/2+x,z")(SVector(0.1, 0.2, 0.3)) ≈ SVector(0.3, 0.6, 0.3)
    @test PureAdsorb.parse_symop("z,x,y")(SVector(0.1, 0.2, 0.3)) ≈ SVector(0.3, 0.1, 0.2)
end

@testitem "read_cif expands a non-P1 IRMOF-1 CIF to a structure matching an independent P1 file" begin
    # RASPA2's own IRMOF-1.cif (space group F m -3 m, #225, 192 listed operations) against
    # PorousMaterials.jl's already-P1 IRMOF-1.cif, both fetched unchanged (data/NOTICE) and
    # neither modified by hand — job 2's own acceptance bar: atom count against the published
    # formula, density, and symmetry expansion reproducing a known P1 file for the same material.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "raspa2_IRMOF-1.cif"))
    # Zn4O(BDC)3 (BDC = benzene-1,4-dicarboxylate, C8H4O4) per formula unit, Z = 8 formula units
    # in this cubic cell (a well-known fact about IRMOF-1/MOF-5's conventional cell).
    @test PureAdsorb.natoms(fw) == 8 * (4 + 1 + 3 * 4 + 3 * 8 + 3 * 4)   # Zn32 O104 C192 H96 = 424
    counts = Dict{String, Int}()
    for s in fw.symbols
        counts[s] = get(counts, s, 0) + 1
    end
    @test counts == Dict("Zn" => 32, "O" => 104, "C" => 192, "H" => 96)

    masses = Dict("Zn" => 65.38, "O" => 15.999, "C" => 12.011, "H" => 1.008)
    mass_amu = sum(masses[s] for s in fw.symbols)
    density_g_cm3 = (mass_amu / 6.02214076e23) / (PureAdsorb.volume(fw.cell) * 1.0e-24)
    # Published IRMOF-1 crystal density is ~0.59-0.61 g/cm^3 (e.g. Eddaoudi et al. 2002, the
    # structure's own citation, and standard RASPA/PorousMaterials benchmark reports).
    @test 0.55 < density_g_cm3 < 0.65

    fw_pm = read_cif(
        joinpath(pkgdir(PureAdsorb), "data", "pm_IRMOF-1.cif");
        charges = Dict("Zn" => 0.0, "O" => 0.0, "C" => 0.0, "H" => 0.0)
    )
    @test PureAdsorb.natoms(fw_pm) == PureAdsorb.natoms(fw)
    # Every symmetry-expanded atom has a same-element match, within 1e-3 fractional units
    # (periodic), among PM's independently-published, already-P1 atom list.
    function count_matches(fw_a, fw_b)
        n = 0
        for i in eachindex(fw_a.frac)
            for j in eachindex(fw_b.frac)
                fw_b.symbols[j] == fw_a.symbols[i] || continue
                d = fw_a.frac[i] - fw_b.frac[j]
                if all(x -> (x = abs(x - round(x)); x < 1.0e-3), d)
                    n += 1
                    break
                end
            end
        end
        return n
    end
    @test count_matches(fw, fw_pm) == PureAdsorb.natoms(fw)
end

@testitem "read_cif's charges keyword overrides the file's own column and requires every label" begin
    # An incomplete map must fail loudly naming the missing label, never fall back silently to
    # the file's own column.
    @test_throws "no entry for atom label" read_cif(
        joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); charges = Dict("bogus_label" => 0.0)
    )
    # A complete map (every RUBTAK.cif label to a distinct value) overrides the in-file column.
    fw_file = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    override = Dict(l => 9.0 for l in fw_file.labels)
    fw_override = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); charges = override)
    @test all(==(9.0), fw_override.charges)
    @test fw_override.frac == fw_file.frac
end
