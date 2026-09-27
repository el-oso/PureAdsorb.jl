# Methane in IRMOF-1 at 298.15 K, built and run as ONE batch via `run_isotherm!`
# (`src/isotherm.jl`), mirroring `isotherm_bench.jl`'s own CO2/RUBTAK isotherm.
#
# The framework (`data/IRMOF-1_P1.cif`) carries an all-zero charge column, unchanged from
# RASPA2's own canonical IRMOF-1.cif (`data/NOTICE`): methane (`data/ch4.yaml`) is a single
# uncharged TraPPE-UA site, so every Coulomb term guest charges enter is identically zero
# regardless of what the framework's charges are (verified directly against the production code
# path, not assumed -- see the job's own report). Framework Lennard-Jones parameters for Zn, O,
# C and H, and the CH4 site itself, are the UFF/TraPPE values already in `data/trappe.yaml`.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
using PureAdsorb, StaticArrays, JSON, LinearAlgebra, Dates, KernelAbstractions, Random, Printf
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cuda")
precision_name = get(ENV, "PA_PRECISION", "f64")
F = precision_name == "f32" ? Float32 : Float64
if backend_name == "cuda"
    using CUDA
    backend = CUDABackend()
    gpu = CUDA.name(CUDA.device())
elseif backend_name == "rocm"
    using AMDGPU
    backend = ROCBackend()
    gpu = AMDGPU.HIP.name(AMDGPU.device())
else
    backend = KernelAbstractions.CPU()
    gpu = ""
end

npress = parse(Int, get(ENV, "PA_NPRESS", "10"))
nreplicas = parse(Int, get(ENV, "PA_NREPLICAS", "3"))
Plo = parse(F, get(ENV, "PA_PLO", "5e4"))          # 0.5 bar
Phi = parse(F, get(ENV, "PA_PHI", "3.6477e6"))     # 36 atm, the Eddaoudi et al. (2002) benchmark pressure
capacity = parse(Int, get(ENV, "PA_CAPACITY", "2500"))
n_warmup = parse(Int, get(ENV, "PA_NWARMUP", "500"))
n_production = parse(Int, get(ENV, "PA_NPRODUCTION", "1000"))
T = F(298.15)

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "IRMOF-1_P1.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "ch4.yaml"), ff; T = F)
sc = replicate(fw, (2, 2, 2))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
pressures = exp.(range(log(Plo), log(Phi); length = npress))

# Framework mass of the simulated (2,2,2) supercell, for converting loading (guests/system) to
# mol/kg: standard atomic weights (IUPAC), not force-field parameters.
const ATOMIC_MASS = Dict("Zn" => 65.38, "O" => 15.999, "C" => 12.011, "H" => 1.008)
const NA = 6.02214076e23
framework_mass_kg = sum(ATOMIC_MASS[s] for s in sc.symbols) / NA / 1000

nsys = npress * nreplicas
t_build_one = @elapsed begin
    b1 = FrameworkBatch([sc], ff, g, ewald; fullk = true)
    SystemState(b1, g, [0], ff; T, seed = 1, capacities = [capacity])
end
t_build_iso = @elapsed begin
    biso = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
    SystemState(biso, g, zeros(Int, nsys), ff; T, seed = 1, capacities = fill(capacity, nsys))
end
println("t_build(nsys=1) = $(round(t_build_one; digits = 3)) s")
println("t_build(nsys=$nsys, $npress pressures x $nreplicas replicas, nframeworks=$(PureAdsorb.nframeworks(biso))) = $(round(t_build_iso; digits = 3)) s")
flush(stdout)

t_run = @elapsed begin
    global iso = run_isotherm!(
        sc, ff, g, ewald; T, pressures, nreplicas, capacity, n_warmup, n_production,
        n_audit = n_warmup + n_production + 1, step_trans = F(1.0), step_rot = F(0.0), exchange_prob = 0.8,
        min_cycle_length = 20, seed = 42, nblocks = 10, backend
    )
end
println("t_run (GCMC, nsys=$nsys, $(n_warmup + n_production) cycles) = $(round(t_run; digits = 1)) s")
loading_molkg = iso.loading ./ (NA * framework_mass_kg)
loading_molkg_err = iso.loading_err ./ (NA * framework_mass_kg)
for p in eachindex(iso.pressure)
    @printf(
        "P=%.4e Pa (%.3f bar)  loading=%.3f +/- %.3f guests/box  %.4f +/- %.4f mol/kg  max_occ=%d/%d\n",
        iso.pressure[p], iso.pressure[p] / 1.0e5, iso.loading[p], iso.loading_err[p],
        loading_molkg[p], loading_molkg_err[p], iso.max_occupancy[p], iso.capacity[p]
    )
end
flush(stdout)

commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, npress, nreplicas, nsys, capacity, n_warmup, n_production, T, commit,
    framework_mass_kg, t_build_nsys1_s = t_build_one, t_build_isotherm_s = t_build_iso, t_run_s = t_run,
    description = "Job 2: CH4 in IRMOF-1 (2x2x2 supercell) at 298.15 K, a $npress-point isotherm " *
        "($Plo-$Phi Pa) x $nreplicas replicas run as one batch via run_isotherm!.",
)
mkpath(joinpath(@__DIR__, "..", "results"))
outpath = joinpath(
    @__DIR__, "..", "results",
    "pureadsorb_isotherm_ch4_irmof1_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(
        io, (;
            meta, pressure = iso.pressure, loading = iso.loading, loading_err = iso.loading_err,
            loading_molkg, loading_molkg_err, energy = iso.energy, energy_err = iso.energy_err,
            max_occupancy = iso.max_occupancy, capacity = iso.capacity,
        ), 2
    )
end
println("wrote $outpath")
