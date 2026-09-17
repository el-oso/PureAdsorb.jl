@testitem "CPU entry points accept OffsetArray and view inputs" begin
    using OffsetArrays, StaticArrays
    σ = OffsetArray([3.0, 4.0], 0:1); ε = OffsetArray([0.01, 0.04], 0:1)
    ff = ForceField(OffsetArray(["A", "B"], 0:1), σ, ε; cutoff = 12.0)
    @test ff.sigma[1, 2] ≈ 3.5
    counts = view([10, 0, 99], 1:2); gc = view([0, 2, 99], 1:2)
    @test PureAdsorb.tail_delta(ff, counts, gc, 1000.0) ≈ PureAdsorb.tail_delta(ff, [10, 0], [0, 2], 1000.0)
    A = SMatrix{3, 3}(10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0)
    pos = OffsetArray([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)], 0:1)
    q = OffsetArray([1.0, -1.0], 0:1)
    ks, w = PureAdsorb.kvectors(A, 2.0)
    @test length(PureAdsorb.structure_factor(ks, pos, q)) == length(ks)
    @test_throws ArgumentError PureAdsorb.ewald_energy(A, pos, q, [1, 2], 0.3, 4.0, ks, w)
end
