# Milestone C task 9: PureAdsorb's own half of the GCMC-vs-kUPS comparison, matched to kUPS's
# shipped `examples/mcmc_rigid.yaml` case (RUBTAK 3x3x3 + CO2, 298.15 K, pressure 1e4 Pa -- the
# yaml's own comment calls this "10 bar", which is wrong: 1e4 Pa is 0.1 bar, matched literally
# here rather than corrected -- 100% exchange, since translation/rotation/reinsertion are 0 there
# and exchange_prob is left at RunConfig's 1/2 default, the only nonzero weight). Reads kUPS's own
# numbers from `bench/run_kups_gcmc.sh main`'s raw output (run separately, since kUPS is a one-off
# `uv`-managed checkout outside this repo) and writes the combined comparison this project's other
# `*_vs_kups_*.json` files use: PureAdsorb's own result, kUPS's (with its host-only baseline
# subtracted from its full-system energy and its heat-of-adsorption sign corrected -- kUPS's GCMC
# analyzer computes cov(U,N)/var(N) - kT, the negative of its own Widom analyzer's kT - <dU*W>/<W>
# convention that PureAdsorb's `fluctuation_qst` follows), and each quantity's difference in
# combined standard errors.
using PureAdsorb, LinearAlgebra, JSON, Dates
BLAS.set_num_threads(1)

F = Float64
fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
b = FrameworkBatch([sc], ff, g, ewald; fullk = true)

T = F(298.15)
P_pa = F(10_000)  # the yaml's own literal value; its comment "10 bar" is wrong (this is 0.1 bar)
fug = peng_robinson_fugacity(P_pa, T, g)
capacity = 300  # >6x the ~40-50 loading Henry's law + the ideal-gas anchor predict at this point

st = SystemState(b, g, [0], ff; T, seed = 42, capacities = [capacity])
results = run_gcmc!(
    b, st, g, ff; T, n_warmup = 1000, n_production = 10_000, n_audit = 500, step_trans = [0.3],
    step_rot = [0.3], fugacity = [fug.f], exchange_prob = 1.0, min_cycle_length = 20, seed = 42, nblocks = 10
)
r = results[1]
println(
    "loading=$(r.loading) +/- $(r.loading_err), energy=$(r.energy) +/- $(r.energy_err), " *
        "q_st=$(r.q_st) +/- $(r.q_st_err), max_occupancy=$(r.max_occupancy)/$(r.capacity)"
)

kups_raw_path = joinpath(@__DIR__, "results", "kups_gcmc_main_neuromancer4070_f64_20260927.json")
kups = JSON.parsefile(kups_raw_path)
k_energy_full = kups["main"]["energy"]["mean"]; k_energy_full_sem = kups["main"]["energy"]["sem"]
k_host = kups["baseline"]["energy"]["mean"]; k_host_sem = kups["baseline"]["energy"]["sem"]
k_energy = k_energy_full - k_host
k_energy_sem = sqrt(k_energy_full_sem^2 + k_host_sem^2)
k_loading = kups["main"]["loading"]["mean"]; k_loading_sem = kups["main"]["loading"]["sem"]
k_qst_raw = kups["main"]["heat_of_adsorption"]["mean"]; k_qst_sem = kups["main"]["heat_of_adsorption"]["sem"]
k_qst = -k_qst_raw  # sign correction, see this file's own header comment

combined_se(a, b) = sqrt(a^2 + b^2)
loading_diff = r.loading - k_loading
loading_cse = combined_se(r.loading_err, k_loading_sem)
energy_diff = r.energy - k_energy
energy_cse = combined_se(r.energy_err, k_energy_sem)
qst_diff = r.q_st - k_qst
qst_cse = combined_se(r.q_st_err, k_qst_sem)

println(
    "loading: diff/cse=$(loading_diff / loading_cse), energy: diff/cse=$(energy_diff / energy_cse), " *
        "q_st: diff/cse=$(qst_diff / qst_cse)"
)

commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
out = (;
    pureadsorb = (;
        loading = (; mean = r.loading, sem = r.loading_err, nblocks = 10),
        energy_eV = (; mean = r.energy, sem = r.energy_err, nblocks = 10),
        q_st_eV = (; mean = r.q_st, sem = r.q_st_err, corr_UN = r.corr_UN),
        max_occupancy = r.max_occupancy, capacity = r.capacity,
        n_warmup_cycles = 1000, n_production_cycles = 10_000, min_cycle_length = 20,
        exchange_prob = 1.0, seed = 42, backend = "cpu",
    ),
    kups = (;
        energy_full_system_eV = kups["main"]["energy"],
        energy_host_only_eV = kups["baseline"]["energy"],
        energy_guest_dependent_eV = (; mean = k_energy, sem = k_energy_sem),
        loading = kups["main"]["loading"],
        heat_of_adsorption_raw_eV = kups["main"]["heat_of_adsorption"],
        heat_of_adsorption_sign_corrected_eV = (; mean = k_qst, sem = k_qst_sem),
    ),
    comparison = (;
        loading = (; diff = loading_diff, combined_se = loading_cse, diff_over_combined_se = loading_diff / loading_cse),
        energy_eV = (; diff = energy_diff, combined_se = energy_cse, diff_over_combined_se = energy_diff / energy_cse),
        q_st_eV = (; diff = qst_diff, combined_se = qst_cse, diff_over_combined_se = qst_diff / qst_cse),
    ),
    meta = (;
        case = "kUPS examples/mcmc_rigid.yaml UNCHANGED: RUBTAK 3x3x3 + CO2, 298.15 K, real-space/Ewald " *
            "cutoff 12 A, Ewald precision 1e-6, pressure=1e4 Pa (yaml comment says '10 bar', wrong: this " *
            "is 0.1 bar, matched literally), 100% exchange (translation/rotation/reinsertion are 0; " *
            "exchange_prob unset there defaults to 0.5, the only nonzero weight), num_warmup_cycles=1000, " *
            "num_cycles=10000, min_cycle_length=20, seed=42",
        fugacity_pa = fug.f, fugacity_coefficient_phi = fug.phi,
        note_fugacity = "phi~0.9995 at this pressure: this comparison does not exercise the equation-of-" *
            "state path beyond a 0.05% correction; ideal_gas_tests.jl's CO2-at-5-MPa case validates that " *
            "separately (a ~20% effect there).",
        note_qst_sign = "kUPS's GCMC analyzer (application/mcmc/analysis.py:122-126) computes " *
            "cov(U,N)/var(N) - kT; its own Widom analyzer (analysis.py:326-332) computes kT - <dU*W>/<W>, " *
            "the opposite sign. PureAdsorb's fluctuation_qst follows the Widom convention, so " *
            "heat_of_adsorption_sign_corrected_eV (kUPS's raw value negated) is what is compared above.",
        note_energy_baseline = "kUPS reports the FULL system energy including U_host-host, a constant " *
            "PureAdsorb's total_energy never computes since it cancels in every difference. " *
            "energy_host_only_eV comes from a separate init_adsorbates=[0], exchange_prob=0, 100-cycle " *
            "run of the same host at the same seed/cutoff/precision.",
        note_nblocks = "kUPS's n_blocks is chosen automatically by its own optimal_block_average (4 " *
            "here); PureAdsorb's is fixed at 10 (run_gcmc!'s own default) -- different block-count rules, " *
            "as in every other cross-code comparison in this project.",
        host = gethostname(), gpu_kups = "NVIDIA GeForce RTX 4070", backend_pureadsorb = "cpu",
        julia = string(VERSION), ewald_cutoff_A = 12.0, ewald_precision = 1.0e-6, temperature_K = 298.15,
        kups_commit = "e183c9aae820b8c98333f8f9ac27a7ac9cfa213d", pureadsorb_commit = commit,
        date = string(today()), kups_raw_source = basename(kups_raw_path),
    ),
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(@__DIR__, "results", "pureadsorb_gcmc_vs_kups_neuromancer4070_f64_$(Dates.format(today(), "yyyymmdd"))_$(commit).json")
open(outpath, "w") do io
    JSON.print(io, out, 2)
end
println("wrote $outpath")
