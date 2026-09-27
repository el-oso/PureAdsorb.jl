@testitem "Peng-Robinson fugacity coefficient approaches 1 as P -> 0" begin
    # CO2's own critical properties (data/co2.yaml); an arbitrary supercritical temperature so
    # there is exactly one real root regardless of pressure.
    tc, pc, omega = 303.75, 7.84e6, 0.22394
    for P in (1.0, 1.0e-3, 1.0e-6, 1.0e-9)
        r = PureAdsorb.peng_robinson_fugacity(P, 400.0, tc, pc, omega)
        @test r.Z ≈ 1 atol = 10 * P / pc
        @test r.phi ≈ 1 atol = 10 * P / pc
        @test r.f ≈ P rtol = 10 * P / pc
    end
end

@testitem "Peng-Robinson fugacity agrees with kUPS's own module to 1e-10 relative" begin
    # Reference values from kUPS's `mcmc.fugacity.peng_robinson_log_fugacity` (JAX float64,
    # `jax_enable_x64`), generated under the standing kUPS reference-oracle exception (`uv run`
    # from `~/src/kups`, `JULIA_GUARD=off` on that command only). CO2 and methane's own critical
    # properties (`data/co2.yaml`, kUPS's own `examples/adsorbate/ch4.yaml`); temperatures span
    # 0.6-2x the critical temperature, pressures 1 Pa to 5x the critical pressure, for each.
    points = [
        (T = 182.25, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999997988312043, ln_f = -2.0116877911189023e-7),
        (T = 182.25, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9982352670341536, ln_f = 9.076162025541716),
        (T = 182.25, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.10179323154970214, ln_f = 11.52331366835566),
        (T = 182.25, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.7686415729378658, ln_f = 12.201275671992724),
        (T = 243.0, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999071767531, ln_f = -9.28232440239905e-8),
        (T = 243.0, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9991864396671702, ln_f = 9.077112151791814),
        (T = 243.0, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.09205622080102857, ln_f = 14.1454945983309),
        (T = 243.0, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.6528725182781026, ln_f = 14.736679750534805),
        (T = 288.5625, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.999999943491023, ln_f = -5.650897602434026e-8),
        (T = 288.5625, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9995048646766468, ln_f = 9.077430416606148),
        (T = 288.5625, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.6273345878399122, ln_f = 15.117732710249584),
        (T = 288.5625, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.6311651438197722, ln_f = 15.77682930322424),
        (T = 303.75, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999516020398, ln_f = -4.839796008568253e-8),
        (T = 303.75, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9995759613611251, ln_f = 9.077501490278937),
        (T = 303.75, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.7109409800508826, ln_f = 15.172633369012635),
        (T = 303.75, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.6350195138703313, ln_f = 16.02707425149436),
        (T = 318.9375, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999583813812, ln_f = -4.1618618160008786e-8),
        (T = 318.9375, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9996353782662367, ln_f = 9.07756089147879),
        (T = 318.9375, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.7657978731522459, ln_f = 15.214835816698551),
        (T = 318.9375, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.6445187619245725, ln_f = 16.240893579896035),
        (T = 334.125, P = 0.001, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999999640886, ln_f = -6.9077552790180485),
        (T = 364.5, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999730463659, ln_f = -2.6953633946104564e-8),
        (T = 364.5, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9997638864371499, ln_f = 9.07768937653927),
        (T = 364.5, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.8623198183208057, ln_f = 15.298746295232773),
        (T = 364.5, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.7048399086631209, ln_f = 16.71188741490943),
        (T = 455.625, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.999999988272758, ln_f = -1.1727241988415948e-8),
        (T = 455.625, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9998972834301845, ln_f = 9.077822764918789),
        (T = 455.625, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9449460815734447, ln_f = 15.378575983111359),
        (T = 455.625, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.8751409636137532, ln_f = 17.186761977125265),
        (T = 607.5, P = 1.0, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999999976661802, ln_f = -2.333819836790541e-9),
        (T = 607.5, P = 8759.774899571317, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9999795632889888, ln_f = 9.077905046980131),
        (T = 607.5, P = 5.054612647198428e6, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 0.9905210474004233, ln_f = 15.425178181899616),
        (T = 607.5, P = 3.92e7, tc = 303.75, pc = 7.84e6, omega = 0.22394, Z = 1.0263529236162376, ln_f = 17.456300086859084),
        (T = 114.3384, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.999999701046828, ln_f = -2.989531368518618e-7),
        (T = 114.3384, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9981322017587797, ln_f = 8.736650404082756),
        (T = 114.3384, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.1055940245246065, ln_f = 11.810645330885254),
        (T = 114.3384, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.7881489821392419, ln_f = 12.509380557162073),
        (T = 152.4512, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.99999985267467, ln_f = -1.4732532314768862e-7),
        (T = 152.4512, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.99908034009872, ln_f = 8.737597447140333),
        (T = 152.4512, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.09652827915104144, ln_f = 13.845737928774597),
        (T = 152.4512, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.667680322557493, ln_f = 14.456040102816134),
        (T = 181.0358, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999054497809, ln_f = -9.455021673876218e-8),
        (T = 181.0358, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9994099556166111, ln_f = 8.737926877107768),
        (T = 181.0358, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.6408394273298627, ln_f = 14.59207927885716),
        (T = 181.0358, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.6364385649381779, ln_f = 15.287637335934994),
        (T = 190.564, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999174986922, ln_f = -8.250130598905578e-8),
        (T = 190.564, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9994851814470843, ln_f = 8.73800207439235),
        (T = 190.564, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.7109409800508818, ln_f = 14.639276909977262),
        (T = 190.564, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.6350195138703313, ln_f = 15.493717792458986),
        (T = 200.0922, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999276744174, ln_f = -7.232558168289944e-8),
        (T = 200.0922, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9995487044028745, ln_f = 8.738065577182406),
        (T = 200.0922, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.759295195826503, ln_f = 14.676415452904978),
        (T = 200.0922, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.6378621254009954, ln_f = 15.67278602001059),
        (T = 209.6204, P = 0.001, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999999363308, ln_f = -6.907755279045806),
        (T = 228.6768, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999501291583, ln_f = -4.987084146102847e-8),
        (T = 228.6768, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.999688855104178, ln_f = 8.738205695997081),
        (T = 228.6768, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.8477547376063894, ln_f = 14.752395874755974),
        (T = 228.6768, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.6703713554165768, ln_f = 16.08230383023921),
        (T = 285.846, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999745383262, ln_f = -2.5461673828433525e-8),
        (T = 285.846, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9998411658599928, ln_f = 8.738357991327241),
        (T = 285.846, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9277304111072766, ln_f = 14.828478844000625),
        (T = 285.846, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.7963576774851009, ln_f = 16.533870081088388),
        (T = 381.128, P = 1.0, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999999910234243, ln_f = -8.97657590244056e-9),
        (T = 381.128, P = 6238.635898490325, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9999440095377546, ln_f = 8.738460835428494),
        (T = 381.128, P = 2.9652008274228387e6, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.9759624978496172, ln_f = 14.877117423296601),
        (T = 381.128, P = 2.2996e7, tc = 190.564, pc = 4.5992e6, omega = 0.01142, Z = 0.948944994553984, ln_f = 16.823911020239205),
    ]
    for p in points
        r = PureAdsorb.peng_robinson_fugacity(p.P, p.T, p.tc, p.pc, p.omega)
        @test r.Z ≈ p.Z rtol = 1.0e-10
        # An additive comparison in log-space is a relative comparison on `f` itself (`f`
        # spans several orders of magnitude across this grid and is never exactly zero).
        @test log(r.f) ≈ p.ln_f atol = 1.0e-10
    end
end

@testitem "three real roots: the largest is picked when the vapor phase is the stable one" begin
    # Both state points are below the critical temperature and produce three real roots
    # (`P/Pc` around 0.13); at this pressure the vapor branch has the lower Gibbs energy, so
    # kUPS's own minimum-fugacity-coefficient selection lands on the LARGEST root. Reference
    # values from kUPS's `mcmc.fugacity.peng_robinson_log_fugacity` (JAX float64), the same
    # oracle the grid test above uses.
    r_co2 = PureAdsorb.peng_robinson_fugacity(1.0172008982397867e6, 243.0, 303.75, 7.84e6, 0.22394)
    @test r_co2.Z ≈ 0.8983272226792529 rtol = 1.0e-10
    @test log(r_co2.f) ≈ 13.734741913105998 atol = 1.0e-10

    r_ch4 = PureAdsorb.peng_robinson_fugacity(274075.6434850021, 152.4512, 190.564, 4.5992e6, 0.01142)
    @test r_ch4.Z ≈ 0.9584833878417425 rtol = 1.0e-10
    @test log(r_ch4.f) ≈ 12.480225316281716 atol = 1.0e-10
end

@testitem "three real roots, above the coexistence pressure: the SMALLEST is selected" begin
    # These two state points also have three real roots, at a higher pressure than the previous
    # item's (same temperatures): above the liquid-vapor coexistence pressure, the smaller,
    # liquid-like-density root has the lower Gibbs energy and kUPS's own selection (minimum
    # fugacity coefficient among the valid roots) picks it instead. Naively always returning the
    # largest real root — a common but wrong shortcut for "the vapor root" — would give the wrong
    # answer at both of these; this pins that `peng_robinson_fugacity` does not take that
    # shortcut. Roots (from an independent `numpy.roots` solve of the same cubic) and kUPS's own
    # selected `Z`/`ln_f` are both float64.
    co2_roots = (0.02899837914793681, 0.11661826542122994, 0.8349333554308334)
    r_co2 = PureAdsorb.peng_robinson_fugacity(1.568e6, 243.0, 303.75, 7.84e6, 0.22394)
    @test r_co2.Z ≈ 0.02899837914793693 rtol = 1.0e-10
    @test log(r_co2.f) ≈ 14.081517421342674 atol = 1.0e-10
    @test r_co2.Z ≈ minimum(co2_roots) rtol = 1.0e-8
    @test !isapprox(r_co2.Z, maximum(co2_roots); rtol = 1.0e-3)

    ch4_roots = (0.032927482555359676, 0.39717333521310005, 0.5439658488982075)
    r_ch4 = PureAdsorb.peng_robinson_fugacity(919840.0, 114.3384, 190.564, 4.5992e6, 0.01142)
    @test r_ch4.Z ≈ 0.0329274825553596 rtol = 1.0e-10
    @test log(r_ch4.f) ≈ 11.737619926415173 atol = 1.0e-10
    @test r_ch4.Z ≈ minimum(ch4_roots) rtol = 1.0e-8
    @test !isapprox(r_ch4.Z, maximum(ch4_roots); rtol = 1.0e-3)
end

@testitem "peng_robinson_fugacity rejects non-positive P, Tgas, tc or pc" begin
    # A root with Z > B always exists for positive P/Tgas/tc/pc (fugacity.jl's docstring proves
    # it), so the only reachable failures are the upfront positivity checks.
    @test_throws "P=" PureAdsorb.peng_robinson_fugacity(0.0, 300.0, 303.75, 7.84e6, 0.22394)
    @test_throws "P=" PureAdsorb.peng_robinson_fugacity(-1.0, 300.0, 303.75, 7.84e6, 0.22394)
    @test_throws "Tgas=" PureAdsorb.peng_robinson_fugacity(1.0e5, 0.0, 303.75, 7.84e6, 0.22394)
    @test_throws "tc=" PureAdsorb.peng_robinson_fugacity(1.0e5, 300.0, 0.0, 7.84e6, 0.22394)
    @test_throws "pc=" PureAdsorb.peng_robinson_fugacity(1.0e5, 300.0, 303.75, 0.0, 0.22394)
end

@testitem "the cubic always has a root above B for positive P, Tgas, tc, pc" begin
    # Direct check of the identity `fugacity.jl`'s docstring proves and relies on: the monic
    # cubic evaluates to exactly -2*B^2 at Z=B, independent of A, so a root with Z > B always
    # exists. Swept over a range of acentric factors, temperatures and pressures well beyond any
    # real gas's (but short of the magnitude where `phi = exp(ln_phi)` itself overflows Float64,
    # a separate, expected limitation of returning `phi`/`f` rather than kUPS's `ln_f`) to confirm
    # `peng_robinson_fugacity` never spuriously throws "no root has Z > B" for a positive
    # P/Tgas/tc/pc.
    for omega in (-0.5, -0.1, 0.0, 0.2, 0.5, 1.0, 2.0), Tg in (50.0, 100.0, 300.0, 1000.0, 2000.0),
            P in (1.0, 1.0e3, 1.0e5, 1.0e8)

        r = PureAdsorb.peng_robinson_fugacity(P, Tg, 190.564, 4.5992e6, omega)
        @test isfinite(r.Z) && isfinite(r.phi) && isfinite(r.f)
    end
end

@testitem "peng_robinson_fugacity(P, T, guest) reads tc/pc/omega off Guest" begin
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"))
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff)
    r1 = PureAdsorb.peng_robinson_fugacity(1.0e6, 300.0, g)
    r2 = PureAdsorb.peng_robinson_fugacity(1.0e6, 300.0, g.tc, g.pc, g.omega)
    @test r1 == r2
end

@testitem "peng_robinson_fugacity is generic over Float32 and Float64" begin
    for T in (Float32, Float64)
        r = PureAdsorb.peng_robinson_fugacity(T(1.0e6), T(300.0), T(303.75), T(7.84e6), T(0.22394))
        @test r.f isa T
        @test r.phi isa T
        @test r.Z isa T
        @test isfinite(r.f) && isfinite(r.phi) && isfinite(r.Z)
    end
end
