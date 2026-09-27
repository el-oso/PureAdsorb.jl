@testsnippet UnitsOracle begin
    using StaticArrays, LinearAlgebra, Random, KernelAbstractions, Unitful

    function units_probe_setup(::Type{F}; ncounts = [5], capacities = [40], seed = 33) where {F}
        fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
        ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
        g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
        sc = replicate(fw, (3, 3, 3))
        ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
        b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
        st = SystemState(b, g, ncounts, ff; T = F(298.15), seed, capacities)
        guest_c = PureAdsorb.compact_guest(b, g)
        N = length(g.sites)
        guest_types = SVector{N, Int}(b.guest_types)
        return b, st, g, guest_c, guest_types, ff
    end
end

@testitem "peng_robinson_fugacity's Unitful boundary matches the bare-Float call" setup = [UnitsOracle] begin
    using Unitful
    tc, pc, omega = 303.75, 7.84e6, 0.22394
    for (P, T) in ((5.0e6, 298.0), (1.0, 400.0), (3.92e7, 243.0))
        bare = peng_robinson_fugacity(P, T, tc, pc, omega)
        boundary = peng_robinson_fugacity(P * u"Pa", T * u"K", tc, pc, omega)
        @test ustrip(u"Pa", boundary.f) == bare.f
        @test boundary.phi == bare.phi
        @test boundary.Z == bare.Z
        @test unit(boundary.f) == u"Pa"
    end
end

@testitem "mc_insert!/mc_delete!/mc_exchange!'s Unitful fugacity matches the bare-Float call exactly" setup = [
    UnitsOracle,
] begin
    using Unitful
    b, st_bare, g, gc, gt, ff = units_probe_setup(Float64)
    st_unitful = deepcopy(st_bare)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
    ws_bare = PureAdsorb.MoveWorkspace(Float64, st_bare.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st_bare.nsys))
    ws_unitful = PureAdsorb.MoveWorkspace(Float64, st_unitful.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st_unitful.nsys))
    kT = PureAdsorb.KB * 298.15
    fug_pa = 2.0e4

    rng_bare = Xoshiro(1); rng_unitful = Xoshiro(1)
    for _ in 1:30
        PureAdsorb.mc_insert!(ws_bare, b, st_bare, gc, gt, p, q, [fug_pa], kT)
        PureAdsorb.mc_insert!(ws_unitful, b, st_unitful, gc, gt, p, q, [fug_pa * u"Pa"], kT)
        PureAdsorb.mc_delete!(ws_bare, b, st_bare, gc, gt, p, q, [fug_pa], kT)
        PureAdsorb.mc_delete!(ws_unitful, b, st_unitful, gc, gt, p, q, [fug_pa * u"Pa"], kT)
        PureAdsorb.mc_exchange!(rng_bare, ws_bare, b, st_bare, gc, gt, p, q, [fug_pa], kT)
        PureAdsorb.mc_exchange!(rng_unitful, ws_unitful, b, st_unitful, gc, gt, p, q, [fug_pa * u"Pa"], kT)
    end
    @test st_bare.occupancy == st_unitful.occupancy
    @test st_bare.energy == st_unitful.energy
    @test st_bare.refpoints == st_unitful.refpoints
    @test st_bare.Sk == st_unitful.Sk
end

@testitem "the Unitful boundary never reaches FrameworkBatch or SystemState's arrays" setup = [UnitsOracle] begin
    using Unitful: Quantity
    b, st, g, gc, gt, ff = units_probe_setup(Float64)
    p, q = PureAdsorb.exchange_constant_coeffs(ff, b, gc)
    ws = PureAdsorb.MoveWorkspace(Float64, st.nsys, PureAdsorb.default_nblocks_per_chain(Float64, st.nsys))
    kT = PureAdsorb.KB * 298.15
    rng = Xoshiro(1)
    for _ in 1:10
        PureAdsorb.mc_exchange!(rng, ws, b, st, gc, gt, p, q, [2.0e4 * Unitful.u"Pa"], kT)
    end
    for x in (b, st)
        for fname in fieldnames(typeof(x))
            v = getfield(x, fname)
            v isa AbstractArray || continue
            @test !(eltype(v) <: Quantity)
        end
    end
end
