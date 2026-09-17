@testitem "constants match kUPS" begin
    @test PureAdsorb.KB ≈ 8.6173303e-5 rtol = 1.0e-7
    @test PureAdsorb.KE ≈ 14.3996454 rtol = 1.0e-8
end
