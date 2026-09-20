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
    @test_throws "cell" PureAdsorb.cell_matrix(0, 1, 1, 90, 90, 90)
    @test_throws "cell" PureAdsorb.cell_matrix(-1, 1, 1, 90, 90, 90)
end

@testitem "grid_dims and stencil_reaches match their formulas" begin
    using StaticArrays
    L = SVector(10.0, 21.0, 7.5)
    @test PureAdsorb.grid_dims(L, 3.0) == SVector{3, Int32}(3, 7, 2)
    @test PureAdsorb.grid_dims(L, 100.0) == SVector{3, Int32}(1, 1, 1)   # never below one cell
    n = PureAdsorb.grid_dims(L, 3.0)
    @test PureAdsorb.stencil_reaches(L, n, 14.0) == SVector{3, Int32}(
        ceil(Int, 14.0 * 3 / 10.0), ceil(Int, 14.0 * 7 / 21.0), ceil(Int, 14.0 * 2 / 7.5)
    )
end

@testitem "home_cell and home_cell_dev agree and land in range" begin
    using Random
    n = Int32(9)
    for f in (0.0, 0.05, 0.5, 1 / 9, 8 / 9 + 1.0e-9, prevfloat(1.0))
        h = PureAdsorb.home_cell(f, n)
        hd = PureAdsorb.home_cell_dev(f, n)
        @test h == hd
        @test 0 <= h < n
        @test h == clamp(floor(Int, f * n), 0, n - 1)
    end
    rng = Xoshiro(3)
    for _ in 1:2000
        f = rand(rng)
        @test PureAdsorb.home_cell(f, n) == PureAdsorb.home_cell_dev(f, n)
    end
end

@testitem "cell_coord wraps fractional coordinates outside [0,1)" begin
    using StaticArrays
    n = SVector{3, Int32}(4, 4, 4)
    @test PureAdsorb.cell_coord(SVector(0.0, 0.0, 0.0), n) == SVector{3, Int32}(0, 0, 0)
    @test PureAdsorb.cell_coord(SVector(1.0 - 1.0e-12, 1.0 - 1.0e-12, 1.0 - 1.0e-12), n) == SVector{3, Int32}(3, 3, 3)
    # wrapping: a coordinate of 1.1 and 0.1 land in the same cell
    @test PureAdsorb.cell_coord(SVector(1.1, -0.9, 0.1), n) == PureAdsorb.cell_coord(SVector(0.1, 0.1, 0.1), n)
end

@testitem "cell_linear is a bijection onto 0:n1*n2*n3-1" begin
    using StaticArrays
    n = SVector{3, Int32}(3, 4, 5)
    seen = falses(3 * 4 * 5)
    for k in 0:4, j in 0:3, i in 0:2
        c = PureAdsorb.cell_linear(Int32(i), Int32(j), Int32(k), n[1], n[2])
        @test 0 <= c < 3 * 4 * 5
        @test !seen[c + 1]
        seen[c + 1] = true
    end
    @test all(seen)
end

@testitem "wrap_cell is the identity in range and wraps just outside it" begin
    n = Int32(6)
    for x in Int32(0):Int32(n - 1)
        @test PureAdsorb.wrap_cell(x, n) == x
    end
    @test PureAdsorb.wrap_cell(Int32(-1), n) == n - 1
    @test PureAdsorb.wrap_cell(Int32(-3), n) == n - 3
    @test iszero(PureAdsorb.wrap_cell(n, n))
    @test PureAdsorb.wrap_cell(n + Int32(2), n) == 2
end

@testitem "stencil_start_count visits every cell once when the reach spans the axis" begin
    n = Int32(5)
    for h in Int32(0):Int32(n - 1)
        start, count = PureAdsorb.stencil_start_count(h, Int32(10), n)
        @test iszero(start)
        @test count == n
        visited = Set(PureAdsorb.wrap_cell(start + Int32(t), n) for t in Int32(0):Int32(count - 1))
        @test visited == Set(Int32(0):Int32(n - 1))
    end
end

@testitem "stencil_start_count covers exactly the m cells either side of home" begin
    n = Int32(20)
    m = Int32(2)
    for h in Int32(0):Int32(n - 1)
        start, count = PureAdsorb.stencil_start_count(h, m, n)
        @test count == 2m + 1
        visited = [PureAdsorb.wrap_cell(start + Int32(t), n) for t in Int32(0):Int32(count - 1)]
        @test length(unique(visited)) == length(visited)   # no cell visited twice
        expected = Set(PureAdsorb.wrap_cell(h + d, n) for d in (-m):m)
        @test Set(visited) == expected
    end
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
