@testitem "mulhi64 matches a widened 128-bit reference" begin
    using Random
    rng = Xoshiro(1)
    for _ in 1:100_000
        a, b = rand(rng, UInt64), rand(rng, UInt64)
        @test PureAdsorb.mulhi64(a, b) == UInt64((UInt128(a) * UInt128(b)) >> 64)
    end
    for (a, b) in (
            (UInt64(0), UInt64(0)), (typemax(UInt64), typemax(UInt64)),
            (UInt64(1), typemax(UInt64)), (typemax(UInt64), UInt64(1)),
        )
        @test PureAdsorb.mulhi64(a, b) == UInt64((UInt128(a) * UInt128(b)) >> 64)
    end
end

@testitem "ChainRNG depends on every key component and nothing else" begin
    base = PureAdsorb.ChainRNG(UInt64(7), 3, 2)
    same = PureAdsorb.ChainRNG(UInt64(7), 3, 2)
    @test PureAdsorb.next_bits(base)[1] == PureAdsorb.next_bits(same)[1]

    b0 = PureAdsorb.next_bits(base)[1]
    @test PureAdsorb.next_bits(PureAdsorb.ChainRNG(UInt64(8), 3, 2))[1] != b0
    @test PureAdsorb.next_bits(PureAdsorb.ChainRNG(UInt64(7), 4, 2))[1] != b0
    @test PureAdsorb.next_bits(PureAdsorb.ChainRNG(UInt64(7), 3, 5))[1] != b0

    # successive draws from one stream advance the counter and differ from each other
    _, r1 = PureAdsorb.next_bits(base)
    v2, r2 = PureAdsorb.next_bits(r1)
    v3, _ = PureAdsorb.next_bits(r2)
    @test length(unique((b0, v2, v3))) == 3
end

@testitem "rand_uniform is uniform in [0,1) to a CLT-justified tolerance" begin
    for F in (Float64, Float32)
        n = 2_000_000
        s, s2 = 0.0, 0.0
        rng = PureAdsorb.ChainRNG(UInt64(123), 1, 1)
        for _ in 1:n
            u, rng = PureAdsorb.rand_uniform(rng, F)
            uf = Float64(u)
            s += uf
            s2 += uf^2
        end
        m = s / n
        v = s2 / n - m^2
        # Uniform(0,1) has mean 1/2, variance 1/12, and fourth central moment 1/80 (E[(X-1/2)^4]
        # over X-1/2 ~ Uniform(-1/2,1/2)). Both the sample mean and sample variance are
        # asymptotically normal with these exactly known asymptotic variances, so a 5-sigma band
        # has a two-sided false-positive rate of about 5.7e-7 per check -- tight enough to catch a
        # biased or narrowed generator, loose enough not to fire on sampling noise alone.
        se_mean = sqrt((1 / 12) / n)
        se_var = sqrt((1 / 80 - (1 / 12)^2) / n)
        @test abs(m - 0.5) < 5se_mean
        @test abs(v - 1 / 12) < 5se_var
    end
end

@testitem "rand_range covers 1:n uniformly, without division by a runtime value" begin
    n_range = 7
    n = 2_000_000
    counts = zeros(Int, n_range)
    function fill_counts!(counts, n, n_range)
        rng = PureAdsorb.ChainRNG(UInt64(55), 2, 9)
        for _ in 1:n
            v, rng = PureAdsorb.rand_range(rng, n_range)
            @test 1 <= v <= n_range
            counts[v] += 1
        end
        return nothing
    end
    fill_counts!(counts, n, n_range)
    p = 1 / n_range
    se = sqrt(n * p * (1 - p))   # binomial standard error of each bucket's count
    for c in counts
        @test abs(c - n * p) < 5se
    end

    # n = 1 is a degenerate range with exactly one outcome
    v, _ = PureAdsorb.rand_range(PureAdsorb.ChainRNG(UInt64(1), 1, 1), 1)
    @test v == 1

    # the argument's integer type comes back unchanged, including Int32 (the type this project's
    # index arrays use)
    v32, _ = PureAdsorb.rand_range(PureAdsorb.ChainRNG(UInt64(1), 1, 1), Int32(5))
    @test v32 isa Int32
end

@testitem "rand_normal has mean 0 and variance 1 to a CLT-justified tolerance" begin
    for F in (Float64, Float32)
        n = 2_000_000
        s, s2 = 0.0, 0.0
        rng = PureAdsorb.ChainRNG(UInt64(321), 4, 4)
        for _ in 1:n
            z, rng = PureAdsorb.rand_normal(rng, F)
            zf = Float64(z)
            s += zf
            s2 += zf^2
        end
        m = s / n
        v = s2 / n - m^2
        # Standard normal: Var(sample mean) = 1/n exactly; Var(sample variance) = 2/(n-1)
        # (Var(X^2) = E[X^4] - 1 = 3 - 1 = 2 for a standard normal). Same 5-sigma justification
        # as the uniform test above.
        se_mean = sqrt(1 / n)
        se_var = sqrt(2 / (n - 1))
        @test abs(m) < 5se_mean
        @test abs(v - 1) < 5se_var
    end
end

@testitem "rand_quaternion is uniform on SO(3): a rotation-invariance check, not just unit norm" begin
    using StaticArrays, LinearAlgebra
    for F in (Float64, Float32)
        n = 1_000_000
        v0 = SVector{3, F}(0, 0, 1)
        sx = sy = sz = 0.0
        sxx = syy = szz = 0.0
        sxy = sxz = syz = 0.0
        rng = PureAdsorb.ChainRNG(UInt64(9), 5, 1)
        normtol = F === Float64 ? 1.0e-10 : 1.0e-4
        for _ in 1:n
            q, rng = PureAdsorb.rand_quaternion(rng, F)
            @test abs(norm(q) - 1) < normtol
            rv = PureAdsorb.rotate(q, v0)
            x, y, z = Float64(rv[1]), Float64(rv[2]), Float64(rv[3])
            sx += x
            sy += y
            sz += z
            sxx += x * x
            syy += y * y
            szz += z * z
            sxy += x * y
            sxz += x * z
            syz += y * z
        end
        mx, my, mz = sx / n, sy / n, sz / n
        cxx, cyy, czz = sxx / n - mx^2, syy / n - my^2, szz / n - mz^2
        cxy, cxz, cyz = sxy / n - mx * my, sxz / n - mx * mz, syz / n - my * mz
        # A vector rotated by a Haar-uniform SO(3) element is uniform on the sphere of the same
        # radius, for ANY fixed starting vector: its mean is zero and its covariance is (1/3)I,
        # both basis-independent (rotation-invariant) properties -- unlike checking one marginal
        # in one fixed frame, which a non-uniform-but-axis-symmetric sampler could still pass.
        # Exact moments of a point uniform on the unit sphere: E[x_i^2] = 1/3, E[x_i^4] = 1/5 (each
        # coordinate marginal is Uniform(-1,1) by Archimedes' hat-box theorem), and, from the
        # general moment formula for the uniform sphere measure, E[x_i^2 x_j^2] = 1/15 for i != j.
        # So Var(x_i^2) = 1/5 - 1/9 = 4/45 and Var(x_i x_j) = 1/15 - 0^2 = 1/15 for i != j.
        se_mean = sqrt((1 / 3) / n)
        se_diag = sqrt((4 / 45) / n)
        se_offdiag = sqrt((1 / 15) / n)
        for m in (mx, my, mz)
            @test abs(m) < 5se_mean
        end
        for c in (cxx, cyy, czz)
            @test abs(c - 1 / 3) < 5se_diag
        end
        for c in (cxy, cxz, cyz)
            @test abs(c) < 5se_offdiag
        end
    end
end

@testitem "distinct chains give independent, uncorrelated streams" begin
    n = 1_000_000
    seedA = PureAdsorb.splitmix64(UInt64(99), UInt64(1))
    seedB = PureAdsorb.splitmix64(UInt64(99), UInt64(2))
    function paired_sums(seedA, seedB, n)
        rngA = PureAdsorb.ChainRNG(seedA, 1, 1)
        rngB = PureAdsorb.ChainRNG(seedB, 1, 1)
        sA = sB = sAB = sA2 = sB2 = 0.0
        for _ in 1:n
            a, rngA = PureAdsorb.rand_uniform(rngA, Float64)
            b, rngB = PureAdsorb.rand_uniform(rngB, Float64)
            sA += a
            sB += b
            sAB += a * b
            sA2 += a^2
            sB2 += b^2
        end
        return sA, sB, sAB, sA2, sB2
    end
    sA, sB, sAB, sA2, sB2 = paired_sums(seedA, seedB, n)
    mA, mB = sA / n, sB / n
    vA, vB = sA2 / n - mA^2, sB2 / n - mB^2
    cov = sAB / n - mA * mB
    r = cov / sqrt(vA * vB)
    # Under independence, the sample correlation coefficient of n pairs has asymptotic standard
    # error 1/sqrt(n) (Fisher); a 5-sigma band is 5/sqrt(n) =~ 0.005 at this n.
    @test abs(r) < 5 / sqrt(n)
end

@testitem "identical draws for the same chain across two different batch sizes" begin
    using KernelAbstractions, StaticArrays
    seed = UInt64(2024)
    chain_idx = collect(1:5)
    cycles_probe = Int32.(7 .+ chain_idx)
    moves_probe = Int32.(13 .+ chain_idx)
    nrange_probe = Int32.(2 .+ chain_idx)
    seeds_probe = [PureAdsorb.splitmix64(seed, UInt64(n)) for n in chain_idx]

    function draw_all(rng_seed, cycles, moves, nrange)
        m = length(rng_seed)
        u = zeros(Float64, m)
        r = zeros(Int32, m)
        z = zeros(Float64, m)
        q = Vector{SVector{4, Float64}}(undef, m)
        PureAdsorb.rng_draw_kernel!(CPU())(u, r, z, q, rng_seed, cycles, moves, nrange; ndrange = m)
        KernelAbstractions.synchronize(CPU())
        return u, r, z, q
    end

    # "small batch": only the probed chains.
    u_small, r_small, z_small, q_small = draw_all(seeds_probe, cycles_probe, moves_probe, nrange_probe)

    # "large batch": the same probed chains embedded at scattered positions among many more
    # unrelated ones, a much larger total batch size.
    nbig = 400
    positions = [3, 50, 120, 250, 399]
    rs = [PureAdsorb.splitmix64(seed, UInt64(1000 + i)) for i in 1:nbig]
    cs = Int32.(1:nbig)
    ms = Int32.(2:(nbig + 1))
    ns = fill(Int32(9), nbig)
    for (k, p) in enumerate(positions)
        rs[p] = seeds_probe[k]
        cs[p] = cycles_probe[k]
        ms[p] = moves_probe[k]
        ns[p] = nrange_probe[k]
    end
    u_big, r_big, z_big, q_big = draw_all(rs, cs, ms, ns)

    for (k, p) in enumerate(positions)
        @test u_small[k] == u_big[p]
        @test r_small[k] == r_big[p]
        @test z_small[k] == z_big[p]
        @test q_small[k] == q_big[p]
    end
end

@testitem "identical draws whether a run of moves is one chunk or split across several" begin
    using KernelAbstractions, StaticArrays
    seed = UInt64(77)
    ntot = 20
    rs = fill(PureAdsorb.splitmix64(seed, UInt64(1)), ntot)   # one chain throughout
    cs = fill(Int32(5), ntot)
    ms = Int32.(1:ntot)   # move index is the position: each entry is a distinct move on that chain
    ns = fill(Int32(6), ntot)

    function draw_range(rs, cs, ms, ns)
        m = length(rs)
        u = zeros(Float64, m)
        r = zeros(Int32, m)
        z = zeros(Float64, m)
        q = Vector{SVector{4, Float64}}(undef, m)
        PureAdsorb.rng_draw_kernel!(CPU())(u, r, z, q, rs, cs, ms, ns; ndrange = m)
        KernelAbstractions.synchronize(CPU())
        return u, r, z, q
    end
    combine(chunks) = (vcat((c[1] for c in chunks)...), vcat((c[2] for c in chunks)...), vcat((c[3] for c in chunks)...), vcat((c[4] for c in chunks)...))
    at(rng) = draw_range(view(rs, rng), view(cs, rng), view(ms, rng), view(ns, rng))

    one_chunk = draw_range(rs, cs, ms, ns)

    two_chunks = combine((at(1:7), at(8:ntot)))
    @test one_chunk[1] == two_chunks[1]
    @test one_chunk[2] == two_chunks[2]
    @test one_chunk[3] == two_chunks[3]
    @test one_chunk[4] == two_chunks[4]

    three_chunks = combine((at(1:3), at(4:11), at(12:ntot)))
    @test one_chunk[1] == three_chunks[1]
    @test one_chunk[2] == three_chunks[2]
    @test one_chunk[3] == three_chunks[3]
    @test one_chunk[4] == three_chunks[4]
end

@testitem "GPU matches CPU exactly for the same keys" tags = [:gpu] begin
    using KernelAbstractions, StaticArrays, Random
    backend = nothing
    if !isnothing(Base.find_package("CUDA"))
        @eval using CUDA
        CUDA.functional() && (backend = CUDABackend())
    end
    if isnothing(backend) && !isnothing(Base.find_package("AMDGPU"))
        @eval using AMDGPU
        AMDGPU.functional() && (backend = ROCBackend())
    end
    isnothing(backend) && error("no functional GPU backend; this item must run on a GPU host")

    n = 50_000
    seed = UInt64(31)
    rng = Xoshiro(2)
    rs = [PureAdsorb.splitmix64(seed, UInt64(i)) for i in 1:n]
    cs = Int32.(1:n)
    ms = Int32.(rand(rng, UInt32, n) .% UInt32(97))
    ns = Int32.(2 .+ (1:n) .% 11)

    u_cpu = zeros(Float64, n)
    r_cpu = zeros(Int32, n)
    z_cpu = zeros(Float64, n)
    q_cpu = Vector{SVector{4, Float64}}(undef, n)
    PureAdsorb.rng_draw_kernel!(CPU())(u_cpu, r_cpu, z_cpu, q_cpu, rs, cs, ms, ns; ndrange = n)
    KernelAbstractions.synchronize(CPU())

    du = PureAdsorb.adapt(backend, zeros(Float64, n))
    dr = PureAdsorb.adapt(backend, zeros(Int32, n))
    dz = PureAdsorb.adapt(backend, zeros(Float64, n))
    dq = PureAdsorb.adapt(backend, Vector{SVector{4, Float64}}(undef, n))
    drs, dcs, dms, dns = PureAdsorb.adapt(backend, rs), PureAdsorb.adapt(backend, cs), PureAdsorb.adapt(backend, ms), PureAdsorb.adapt(backend, ns)
    PureAdsorb.rng_draw_kernel!(backend)(du, dr, dz, dq, drs, dcs, dms, dns; ndrange = n)
    KernelAbstractions.synchronize(backend)

    # `rand_uniform`/`rand_range` are pure bit shifts, masks, multiplies and adds on integers --
    # exactly reproducible across backends under IEEE-754 -- so these two compare bit for bit.
    @test Array(du) == u_cpu
    @test Array(dr) == r_cpu
    # `rand_normal`/`rand_quaternion` call `log`/`sqrt`/`cospi`/`sinpi`, whose CUDA device
    # implementations are not bound to match the host's `libm` bit for bit (both are within their
    # documented accuracy, just not identical). Measured on this host: at most 2 ULPs of
    # disagreement over 50,000 draws, on both `z` and every quaternion component -- consistent
    # with that library difference and nothing else, so `atol` covers a handful of ULPs rather
    # than papering over a real discrepancy.
    @test all(abs(a - b) <= 4eps(Float64) for (a, b) in zip(Array(dz), z_cpu))
    @test all(maximum(abs.(a - b)) <= 4eps(Float64) for (a, b) in zip(Array(dq), q_cpu))
end
