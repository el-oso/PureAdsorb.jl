@testitem "alpha and kmax follow kUPS" begin
    using SpecialFunctions
    α = PureAdsorb.ewald_alpha(12.0, 1.0e-6)
    @test erfc(α * 12.0) ≈ 12.0 * 5.0e-7 rtol = 1.0e-9
    @test PureAdsorb.ewald_kmax(α, 1.0e-6) ≈ 2α * sqrt(-log(5.0e-7))
end

@testitem "erfc_dev matches SpecialFunctions" begin
    using SpecialFunctions
    for x in 0:0.01:6
        @test PureAdsorb.erfc_dev(x) ≈ erfc(x) rtol = 1.0e-11
    end
    @test PureAdsorb.erfc_dev(-1.0) ≈ erfc(-1.0) rtol = 1.0e-11
    @test PureAdsorb.erfc_dev(8.0) ≈ erfc(8.0) rtol = 1.0e-9
end

@testitem "Madelung constant of NaCl" begin
    using StaticArrays, LinearAlgebra
    a = 5.64
    A = SMatrix{3, 3}(a, 0, 0, 0, a, 0, 0, 0, a)
    base = (
        ((0, 0, 0), 1.0), ((0.5, 0.5, 0), 1.0), ((0.5, 0, 0.5), 1.0), ((0, 0.5, 0.5), 1.0),
        ((0.5, 0, 0), -1.0), ((0, 0.5, 0), -1.0), ((0, 0, 0.5), -1.0), ((0.5, 0.5, 0.5), -1.0),
    )
    for (rc, prec) in ((a * 0.49, 1.0e-8), (2a, 1.0e-8))
        m = PureAdsorb.min_multiplicity(A, rc)
        Asc = A * Diagonal(SVector(m))
        pos = SVector{3, Float64}[]
        q = Float64[]
        for k in 0:(m[3] - 1), j in 0:(m[2] - 1), i in 0:(m[1] - 1), (f, c) in base
            push!(pos, A * (SVector(f...) + SVector(i, j, k)))
            push!(q, c)
        end
        α = PureAdsorb.ewald_alpha(rc, prec)
        ks, w = PureAdsorb.kvectors(Asc, PureAdsorb.ewald_kmax(α, prec))
        E = PureAdsorb.ewald_energy(Asc, pos, q, collect(eachindex(pos)), α, rc, ks, w)
        @test E / (4 * prod(m)) ≈ -1.747565 * PureAdsorb.KE / (a / 2) rtol = 1.0e-5   # per ion pair
    end
end

@testitem "energy independent of alpha" begin
    using StaticArrays, Random
    A = SMatrix{3, 3}(20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0)
    rng = Xoshiro(3)
    pos = [A * rand(rng, SVector{3, Float64}) for _ in 1:40]
    q = [isodd(i) ? 0.5 : -0.5 for i in 1:40]
    mol = collect(1:40)
    Es = map((6.0, 8.0, 9.5)) do rc
        α = PureAdsorb.ewald_alpha(rc, 1.0e-8)
        ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1.0e-8))
        PureAdsorb.ewald_energy(A, pos, q, mol, α, rc, ks, w)
    end
    @test all(e -> isapprox(e, Es[1]; rtol = 1.0e-6), Es)
end

@testitem "kvectors bound uses lattice vector lengths" begin
    using StaticArrays, LinearAlgebra
    A = PureAdsorb.cell_matrix(14.7619, 14.80147, 14.76539, 59.84578, 60.04729, 59.8131)
    α = PureAdsorb.ewald_alpha(12.0, 1.0e-6)
    kmax = PureAdsorb.ewald_kmax(α, 1.0e-6)
    B = PureAdsorb.reciprocal_basis(A)

    lat = ntuple(i -> norm(A[:, i]), 3)
    perp = PureAdsorb.perpendicular_lengths(A)
    n_lat = ntuple(i -> ceil(Int, kmax * lat[i] / (2π)), 3)
    n_perp = ntuple(i -> ceil(Int, kmax * perp[i] / (2π)), 3)
    @test all(n_lat[i] >= n_perp[i] for i in 1:3)
    @test any(n_lat[i] > n_perp[i] for i in 1:3)   # the perpendicular-length bound undercounts here

    N = maximum(n_lat)
    ks_brute = SVector{3, Float64}[]
    w_brute = Float64[]
    for n1 in 0:(2N), n2 in (-2N):(2N), n3 in (-2N):(2N)
        (iszero(n1) && iszero(n2) && iszero(n3)) && continue
        k = B * SVector(n1, n2, n3)
        norm(k) <= kmax || continue
        push!(ks_brute, k)
        push!(w_brute, iszero(n1) ? 1.0 : 2.0)
    end

    ks, w = PureAdsorb.kvectors(A, kmax)
    @test length(ks) == length(ks_brute)
    @test sum(w) == sum(w_brute)
end

@testitem "intramolecular exclusion removes the pair" begin
    using StaticArrays
    A = SMatrix{3, 3}(30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0)
    pos = [SVector(15.0, 15.0, 15.0), SVector(16.16, 15.0, 15.0)]
    q = [0.7, -0.35]
    α = PureAdsorb.ewald_alpha(12.0, 1.0e-8)
    ks, w = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(α, 1.0e-8))
    Esame = PureAdsorb.ewald_energy(A, pos, q, [1, 1], α, 12.0, ks, w)
    Ediff = PureAdsorb.ewald_energy(A, pos, q, [1, 2], α, 12.0, ks, w)
    @test Ediff - Esame ≈ PureAdsorb.KE * 0.7 * -0.35 / 1.16 rtol = 1.0e-3   # periodic images make this approximate
end
