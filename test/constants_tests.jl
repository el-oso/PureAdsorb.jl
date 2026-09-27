@testitem "constants match kUPS" begin
    @test PureAdsorb.KB ≈ 8.6173303e-5 rtol = 1.0e-7
    @test PureAdsorb.KE ≈ 14.3996454 rtol = 1.0e-8
end

@testitem "PASCAL converts Pa to eV/Å³" begin
    # 1/(1e30 * 1.6021766208e-19), matching kUPS's own PASCAL (application/mcmc/data.py:427-431).
    @test PureAdsorb.PASCAL ≈ 6.241509125883258e-12 rtol = 1.0e-12
    # Standard atmosphere (101325 Pa) in eV/Å³, cross-checked against the direct J/m³ -> eV/Å³
    # unit conversion rather than against PASCAL's own formula.
    atm_pa = 101325.0
    @test PureAdsorb.PASCAL * atm_pa ≈ atm_pa / (1.0e30 * 1.6021766208e-19) rtol = 1.0e-14
end
