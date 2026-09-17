@testitem "batch layout and offsets" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc, sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    @test b.nsys == 2
    @test b.atom_offsets == Int32[0, 3078, 6156]
    @test length(b.ks) == 2 * (b.k_offsets[2] - b.k_offsets[1])
    @test b.constant_offset[1] == b.constant_offset[2]
    @test b.types[1] == PureAdsorb.typeindex(ff, "Zr_")
end

@testitem "batch rejects a cell too small for the cutoff" begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    @test_throws "replicate" FrameworkBatch([fw], ff, g, EwaldParams(cutoff = 12.0))
end

@testitem "constant offset matches the pose-independent terms" begin
    using StaticArrays, LinearAlgebra
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    b = FrameworkBatch([sc], ff, g, EwaldParams(cutoff = 12.0, precision = 1.0e-6))
    α = b.alphas[1]; V = b.volumes[1]
    self = -PureAdsorb.KE * α / sqrt(π) * sum(abs2, g.charges)
    excl = -PureAdsorb.KE * sum(
        g.charges[a] * g.charges[c] * (1 - PureAdsorb.erfc_dev(α * norm(g.sites[a] - g.sites[c]))) / norm(g.sites[a] - g.sites[c])
            for a in 1:3 for c in (a + 1):3
    )
    counts = [count(==(t), b.types) for t in eachindex(ff.names)]
    gcounts = [count(==(t), g.types) for t in eachindex(ff.names)]
    tail = PureAdsorb.tail_delta(ff, counts, gcounts, V)
    @test b.constant_offset[1] ≈ self + excl + tail     # CO2 is neutral: no net-charge term
end
