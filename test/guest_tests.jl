@testsnippet GuestOracle begin
    using StaticArrays, LinearAlgebra, Random

    # Independent, test-only total-energy oracle: a literal (non-incremental) Ewald sum over
    # every host-plus-guest position via `ewald_energy` (a pre-existing, general primitive that
    # predates and is not part of the incremental machinery this file validates — `guest.jl`
    # never calls it), minus the host-only Ewald energy to drop the constant `U_host-host` term
    # the same way `total_energy` does, plus a hand-written direct-pair Lennard-Jones loop over
    # every host-guest and guest-guest pair (no cell list, no shortcuts). `precision` sets the
    # oracle's OWN k-vector cutoff independently of the batch's; passing it equal to the batch's
    # own `EwaldParams.precision` isolates the formula/bookkeeping from Ewald's own truncation
    # error (see the guest-move throughput bench's finding that the two are not interchangeable
    # once precision differs).
    function oracle_total_energy(batch, state, guest::PureAdsorb.Guest{T, N}, ff, n::Integer, precision) where {T, N}
        fw = batch.framework_of[n]
        A = batch.cells[fw]; invA = batch.invcells[fw]
        a0 = batch.atom_offsets[fw]; natoms = batch.atom_offsets[fw + 1] - a0
        hpos = batch.positions[(a0 + 1):(a0 + natoms)]
        hq = batch.charges[(a0 + 1):(a0 + natoms)]
        htype = batch.compact_to_orig[batch.types[(a0 + 1):(a0 + natoms)]]
        gr = PureAdsorb.guest_range(state, n)
        Ng = length(gr)
        alpha = batch.alphas[fw]; ewald_cutoff = batch.ewald_cutoff
        ks, w, = PureAdsorb.kvectors(A, PureAdsorb.ewald_kmax(alpha, precision))

        gpos = SVector{3, T}[]; gq = T[]; gmol = Int[]; gtype_orig = Int[]
        for (idx, i) in enumerate(gr), s in 1:N
            push!(gpos, state.refpoints[i] + PureAdsorb.rotate(state.orientations[i], guest.sites[s]))
            push!(gq, guest.charges[s]); push!(gmol, natoms + idx); push!(gtype_orig, guest.types[s])
        end
        allpos = vcat(hpos, gpos); allq = vcat(hq, gq); allmol = vcat(collect(1:natoms), gmol)
        Ecoul = PureAdsorb.ewald_energy(A, allpos, allq, allmol, alpha, ewald_cutoff, ks, w) -
            PureAdsorb.ewald_energy(A, hpos, hq, collect(1:natoms), alpha, ewald_cutoff, ks, w)

        rc2 = ff.cutoff^2
        Elj = zero(T)
        for a in eachindex(gpos)
            for h in eachindex(hpos)
                r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - hpos[h]))
                r * r < rc2 || continue
                sig = ff.sigma[gtype_orig[a], htype[h]]; eps = ff.epsilon[gtype_orig[a], htype[h]]
                x = (sig / r)^6
                Elj += 4 * eps * (x^2 - x)
            end
            for bb in eachindex(gpos)
                bb > a || continue
                gmol[a] == gmol[bb] && continue
                r = norm(PureAdsorb.minimum_image(A, invA, gpos[a] - gpos[bb]))
                r * r < rc2 || continue
                sig = ff.sigma[gtype_orig[a], gtype_orig[bb]]; eps = ff.epsilon[gtype_orig[a], gtype_orig[bb]]
                x = (sig / r)^6
                Elj += 4 * eps * (x^2 - x)
            end
        end

        ntypes_ff = length(ff.names)
        gcounts = zeros(Int, ntypes_ff)
        for t in guest.types
            gcounts[t] += 1
        end
        host_counts = zeros(Int, ntypes_ff)
        for t in htype
            host_counts[t] += 1
        end
        Etail = PureAdsorb.tail_delta(ff, host_counts, Ng .* gcounts, batch.volumes[fw])
        return Elj + Ecoul + Etail
    end

    # kUPS's own host/empty.cif is a 30 Å cubic P1 cell with one non-interacting dummy site
    # (`_atom_site_charge` is absent from that file, so `read_cif` cannot parse it directly);
    # this reproduces the same physical setup — sigma=1, epsilon=0, charge=0 — directly, giving a
    # pure-CO2-in-vacuum-with-PBC system whose host contributes identically zero to every term.
    function empty_box_setup(::Type{T}) where {T}
        ff0 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = T)
        names = vcat(ff0.names, ["X1_"])
        sigma = vcat([ff0.sigma[i, i] for i in eachindex(ff0.names)], T(1))
        epsilon = vcat([ff0.epsilon[i, i] for i in eachindex(ff0.names)], T(0))
        # `ForceField`'s Lorentz-Berthelot mixing needs only the per-type (diagonal) values.
        ff = ForceField(names, sigma, epsilon; cutoff = T(12), tail = true)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = T)
        A = SMatrix{3, 3}(T(30), T(0), T(0), T(0), T(30), T(0), T(0), T(0), T(30))
        fw = Framework{T}(A, [SVector(T(0), T(0), T(0))], ["X1"], ["X1"], [T(0)])
        return ff, g, fw
    end
end

@testitem "total_energy matches an independent oracle: RUBTAK 3x3x3 with guests (B2)" setup = [GuestOracle] begin
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    # precision=1e-8 (not kUPS's usual 1e-6) so the k-vector table is converged enough for a
    # 1e-10-relative comparison against the oracle at matching alpha/kmax; α·ewald_cutoff = 3.83
    # stays under `PAIR_ERFC_XMAX` (4.0), so `FrameworkBatch` still accepts it.
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-8)
    b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15, seed = 9)
    for n in 1:2
        ref = oracle_total_energy(b, st, g, ff, n, 1.0e-8)
        prod = PureAdsorb.total_energy(b, st, g, ff, n)
        @test prod ≈ ref rtol = 1.0e-10
    end
end

@testitem "total_energy matches an independent oracle: pure CO2 in an empty box (B1)" setup = [GuestOracle] begin
    ff, g, fw = empty_box_setup(Float64)
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-8)
    b = FrameworkBatch([fw], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [50], ff; T = 298.15, seed = 3)
    ref = oracle_total_energy(b, st, g, ff, 1, 1.0e-8)
    prod = PureAdsorb.total_energy(b, st, g, ff, 1)
    @test prod ≈ ref rtol = 1.0e-10
end

@testitem "total_energy matches an independent oracle in Float32, where precision allows" setup = [GuestOracle] begin
    # Float32's ~1.2e-7 unit roundoff, amplified by the cancellation between the Coulomb sum's
    # large same-sign real-space and reciprocal-space pieces, floors agreement well above 1e-10
    # (measured ~5e-4 worst case here); 1e-3 leaves comfortable margin without being vacuous.
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-8)
    b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15f0, seed = 9)
    for n in 1:2
        ref = oracle_total_energy(b, st, g, ff, n, 1.0f-8)
        prod = PureAdsorb.total_energy(b, st, g, ff, n)
        @test prod ≈ ref rtol = 1.0f-3
    end

    ffe, ge, fwe = empty_box_setup(Float32)
    be = FrameworkBatch([fwe], ffe, ge, ewald; fullk = true)
    ste = PureAdsorb.SystemState(be, ge, [20], ffe; T = 298.15f0, seed = 4)
    refe = oracle_total_energy(be, ste, ge, ffe, 1, 1.0f-8)
    prode = PureAdsorb.total_energy(be, ste, ge, ffe, 1)
    @test prode ≈ refe rtol = 1.0f-3
end

@testitem "guest_self_terms asserts every intramolecular distance is inside the cutoff (R1)" begin
    using StaticArrays
    # A stretched-out three-site guest (10 Å apart) against a 5 Å cutoff: R1's equivalence
    # between the two exclusion conventions requires every intramolecular pair to lie inside the
    # real-space cutoff, so this must throw rather than silently returning a wrong exclusion term.
    stretched = PureAdsorb.Guest(
        SVector(SVector(0.0, 0.0, 0.0), SVector(10.0, 0.0, 0.0), SVector(20.0, 0.0, 0.0)),
        SVector(1, 2, 2), SVector(0.7, -0.35, -0.35), 300.0, 1.0e6, 0.2
    )
    @test_throws "intramolecular distance" PureAdsorb.guest_self_terms(stretched, 0.3, 5.0)
    # CO2's own sites (max separation 2.32 Å) are comfortably inside a 12 Å cutoff.
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    self, excl = PureAdsorb.guest_self_terms(g, 0.267, 12.0)
    @test self < 0   # the Gaussian self-energy correction is always negative
end

@testitem "guest_move_delta equals the difference of two total_energy calls, many random moves" setup = [GuestOracle] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15, seed = 9)

    rng = Xoshiro(77)
    for _ in 1:300
        n = rand(rng, 1:2)
        gr = PureAdsorb.guest_range(st, n)
        isempty(gr) && continue
        i = rand(rng, gr)
        kr = PureAdsorb.kvec_range(st, n)
        ΔS = zeros(ComplexF64, length(kr))
        E_before = PureAdsorb.total_energy(b, st, g, ff, n)
        oldpos = st.refpoints[i]; oldq = st.orientations[i]
        movekind = rand(rng, 1:3)
        newpos, newq = if movekind == 1   # translation: Gaussian displacement (R3)
            oldpos + SVector{3, Float64}(randn(rng, 3)), oldq
        elseif movekind == 2   # rotation about the reference point
            oldpos, normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
        else   # reinsertion: fresh uniform position and orientation
            b.cells[b.framework_of[n]] * SVector{3, Float64}(rand(rng, 3)), normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5))
        end
        ΔU, = PureAdsorb.guest_move_delta(b, st, g, n, i, newpos, newq, ΔS)
        st2 = deepcopy(st)
        st2.refpoints[i] = newpos
        st2.orientations[i] = newq
        st2.Sk[kr] .+= ΔS
        E_after = PureAdsorb.total_energy(b, st2, g, ff, n)
        # Accumulated rounding, not an arbitrary bound: two independent `total_energy` calls each
        # carry O(eps(Float64)) relative error against the true configuration energy, so their
        # difference and the incremental ΔU (itself an independent computation) can disagree by a
        # few times eps(Float64) times the larger energy scale involved.
        scale = max(abs(E_before), abs(E_after), 1.0)
        @test abs(ΔU - (E_after - E_before)) <= 100 * eps(Float64) * scale
    end
end

@testitem "guest_move_delta equals the difference of two total_energy calls in Float32" setup = [GuestOracle] begin
    using StaticArrays, LinearAlgebra, Random
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = Float32)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
    b = FrameworkBatch([sc, sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [4, 3], ff; T = 298.15f0, seed = 9)

    rng = Xoshiro(55)
    for _ in 1:200
        n = rand(rng, 1:2)
        gr = PureAdsorb.guest_range(st, n)
        isempty(gr) && continue
        i = rand(rng, gr)
        kr = PureAdsorb.kvec_range(st, n)
        ΔS = zeros(ComplexF32, length(kr))
        E_before = PureAdsorb.total_energy(b, st, g, ff, n)
        oldpos = st.refpoints[i]; oldq = st.orientations[i]
        newpos = oldpos + SVector{3, Float32}(randn(rng, Float32, 3))
        newq = normalize(SVector{4, Float32}(rand(rng, Float32, 4) .- 0.5f0))
        ΔU, = PureAdsorb.guest_move_delta(b, st, g, n, i, newpos, newq, ΔS)
        st2 = deepcopy(st)
        st2.refpoints[i] = newpos
        st2.orientations[i] = newq
        st2.Sk[kr] .+= ΔS
        E_after = PureAdsorb.total_energy(b, st2, g, ff, n)
        scale = max(abs(E_before), abs(E_after), 1.0f0)
        # A near-overlap can drive both `E_before`/`E_after` into the millions of eV, at which
        # scale a few hundred `eps(Float32)` is the honest rounding floor, not a bug.
        @test abs(ΔU - (E_after - E_before)) <= 400 * eps(Float32) * scale
    end
end

@testitem "realspace_move_kernel! and recip_move_kernel! agree between CPU and CUDA" tags = [:gpu] begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions
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

    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"))
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
    b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    st = PureAdsorb.SystemState(b, g, [50], ff; T = 298.15, seed = 1)
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_compact = PureAdsorb.Guest{Float64, N}(g.sites, guest_types, g.charges, g.tc, g.pc, g.omega)

    nmoves = 500
    rng = Xoshiro(3)
    gr = PureAdsorb.guest_range(st, 1)
    sys_of = fill(Int32(1), nmoves)
    gidx = Int32[rand(rng, gr) for _ in 1:nmoves]
    oldpos = [st.refpoints[i] for i in gidx]
    oldq = [st.orientations[i] for i in gidx]
    newpos = [st.refpoints[i] + SVector{3, Float64}(randn(rng, 3)) for i in gidx]
    newq = [normalize(SVector{4, Float64}(rand(rng, 4) .- 0.5)) for _ in 1:nmoves]

    ΔU_real_cpu = zeros(Float64, nmoves)
    ΔU_recip_cpu = zeros(Float64, nmoves)
    PureAdsorb.realspace_move_kernel!(CPU())(
        ΔU_real_cpu, sys_of, gidx, oldpos, oldq, newpos, newq, b, guest_compact, st.refpoints, st.orientations, st.guest_offsets, guest_types;
        ndrange = nmoves
    )
    PureAdsorb.recip_move_kernel!(CPU())(ΔU_recip_cpu, sys_of, oldpos, oldq, newpos, newq, b, g, st.Sk, st.k_offsets; ndrange = nmoves)
    KernelAbstractions.synchronize(CPU())

    db = PureAdsorb.adapt(backend, b)
    drefpoints = PureAdsorb.adapt(backend, st.refpoints)
    dorientations = PureAdsorb.adapt(backend, st.orientations)
    dguest_offsets = PureAdsorb.adapt(backend, st.guest_offsets)
    dSk = PureAdsorb.adapt(backend, st.Sk)
    dk_offsets = PureAdsorb.adapt(backend, st.k_offsets)
    dsys_of, dgidx = PureAdsorb.adapt(backend, sys_of), PureAdsorb.adapt(backend, gidx)
    doldpos, doldq = PureAdsorb.adapt(backend, oldpos), PureAdsorb.adapt(backend, oldq)
    dnewpos, dnewq = PureAdsorb.adapt(backend, newpos), PureAdsorb.adapt(backend, newq)
    dΔU_real = PureAdsorb.adapt(backend, zeros(Float64, nmoves))
    dΔU_recip = PureAdsorb.adapt(backend, zeros(Float64, nmoves))

    PureAdsorb.realspace_move_kernel!(backend)(
        dΔU_real, dsys_of, dgidx, doldpos, doldq, dnewpos, dnewq, db, guest_compact, drefpoints, dorientations, dguest_offsets, guest_types;
        ndrange = nmoves
    )
    PureAdsorb.recip_move_kernel!(backend)(dΔU_recip, dsys_of, doldpos, doldq, dnewpos, dnewq, db, g, dSk, dk_offsets; ndrange = nmoves)
    KernelAbstractions.synchronize(backend)

    # A near-overlap proposal drives the real-space energy into the 1e14+ range, where a few ULP
    # of cross-device summation-order difference is a large absolute number but a tiny relative
    # one; excluding the largest 1% by |CPU value| isolates the ordinary (non-overlap) agreement.
    real_diffs = abs.(Array(dΔU_real) .- ΔU_real_cpu) ./ max.(abs.(ΔU_real_cpu), 1.0)
    ordinary = sortperm(abs.(ΔU_real_cpu))[1:(nmoves - cld(nmoves, 100))]
    @test maximum(real_diffs[ordinary]) < 1.0e-9
    @test Array(dΔU_recip) ≈ ΔU_recip_cpu rtol = 1.0e-9
end
