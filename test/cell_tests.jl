@testitem "cell matrix from parameters" begin
    using StaticArrays, LinearAlgebra
    A = PureAdsorb.cell_matrix(14.7619, 14.80147, 14.76539, 59.84578, 60.04729, 59.8131)
    @test norm(A[:, 1]) ≈ 14.7619
    @test norm(A[:, 2]) ≈ 14.80147
    @test norm(A[:, 3]) ≈ 14.76539
    @test acosd(dot(A[:, 2], A[:, 3]) / (norm(A[:, 2]) * norm(A[:, 3]))) ≈ 59.84578 rtol = 1.0e-6
    @test PureAdsorb.volume(A) ≈ abs(det(A))
    L = PureAdsorb.perpendicular_lengths(A)
    @test L[1] ≈ PureAdsorb.volume(A) / norm(cross(A[:, 2], A[:, 3]))
    @test PureAdsorb.min_multiplicity(A, 12.0) == ntuple(i -> ceil(Int, 24.0 / L[i]), 3)
    B = PureAdsorb.reciprocal_basis(A)
    @test B' * A ≈ 2π * I
end

@testitem "cell_matrix rejects degenerate cells" begin
    @test_throws "cell" PureAdsorb.cell_matrix(1, 1, 1, 90, 90, 180)
    @test_throws "cell" PureAdsorb.cell_matrix(1, 1, 1, 10, 10, 170)
end

@testitem "minimum image is exact inside the cutoff" begin
    using StaticArrays, LinearAlgebra, Random
    A = 3 * PureAdsorb.cell_matrix(14.7619, 14.80147, 14.76539, 59.84578, 60.04729, 59.8131)
    invA = inv(A)
    rc = 12.0
    rng = Xoshiro(1)
    for _ in 1:2000
        Δ = A * (rand(rng, SVector{3, Float64}) .- 0.5) * 3      # spans several images
        d = PureAdsorb.minimum_image(A, invA, Δ)
        best = minimum(norm(Δ + A * SVector(i, j, k)) for i in -2:2, j in -2:2, k in -2:2)
        if best < rc
            @test norm(d) ≈ best
        end
    end
end
