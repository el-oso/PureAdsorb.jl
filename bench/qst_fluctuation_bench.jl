# Measures the fluctuation q_st's correlation coefficient corr(U,N) and its own precision against
# the loading's, for docs/src/theory.md's grand-canonical section: the fluctuation formula's
# relative error scales as sqrt((1-corr_UN^2)/(n*corr_UN^2)), so a correlation this close to ±1
# makes q_st converge almost as fast as a direct mean despite being a ratio of fluctuations.
# RUBTAK 3x3x3 + CO2, 298.15 K, fugacity 2e4 Pa, capacity 30 -- the same case
# `test/gcmc_tests.jl`'s own q_st/corr_UN test uses, run here with far more production cycles.
using PureAdsorb, LinearAlgebra, JSON, Dates
BLAS.set_num_threads(1)

F = Float64
fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
st = SystemState(b, g, [3], ff; T = F(298.15), seed = 15, capacities = [150])

results = run_gcmc!(
    b, st, g, ff; T = 298.15, n_warmup = 200, n_production = 5000, n_audit = 200, step_trans = [0.3],
    step_rot = [0.3], fugacity = [2.0e4], exchange_prob = 0.5, seed = 16, nblocks = 10
)
r = results[1]
println("loading=$(r.loading) +/- $(r.loading_err), q_st=$(r.q_st) +/- $(r.q_st_err), corr_UN=$(r.corr_UN)")
println("relative error: loading=$(r.loading_err / r.loading), q_st=$(r.q_st_err / abs(r.q_st))")

commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), commit,
    description = "docs/src/theory.md: q_st fluctuation formula's convergence, RUBTAK 3x3x3 + CO2, " *
        "298.15 K, fugacity 2e4 Pa, capacity 30, 3 initial guests, n_warmup=200, n_production=5000.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(@__DIR__, "results", "pureadsorb_qst_fluctuation_neuromancer_cpu_f64_20260927_$(commit).json")
open(outpath, "w") do io
    JSON.print(
        io, (;
            meta, loading = r.loading, loading_err = r.loading_err, q_st = r.q_st, q_st_err = r.q_st_err,
            corr_UN = r.corr_UN, energy = r.energy, energy_err = r.energy_err, max_occupancy = r.max_occupancy,
            capacity = r.capacity, ncycles = r.ncycles,
        )
    )
end
println("wrote $outpath")
