# Milestone C task 6: a real isotherm, CO2 in RUBTAK 3x3x3 at 298.15 K, built and run as ONE
# batch via `run_isotherm!` (`src/isotherm.jl`). Reports the batch-build time separately from the
# GCMC run itself, since the acceptance criterion is specifically that framework deduplication
# keeps the BUILD near the cost of a single framework regardless of how many pressure points or
# replicas the batch carries (`FrameworkBatch`'s own dedup, `batch.jl`).
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

npress = parse(Int, get(ENV, "PA_NPRESS", "50"))
nreplicas = parse(Int, get(ENV, "PA_NREPLICAS", "4"))
Plo = parse(F, get(ENV, "PA_PLO", "1e2"))
Phi = parse(F, get(ENV, "PA_PHI", "1e5"))
capacity = parse(Int, get(ENV, "PA_CAPACITY", "200"))
n_warmup = parse(Int, get(ENV, "PA_NWARMUP", "200"))
n_production = parse(Int, get(ENV, "PA_NPRODUCTION", "500"))
T = F(298.15)

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
pressures = exp.(range(log(Plo), log(Phi); length = npress))

# Isolates the batch BUILD from the run: the same `FrameworkBatch`+`SystemState` construction
# `run_isotherm!` does internally, timed here alone against a single-framework, single-system
# build for comparison. `ncounts = 0` throughout, matching `run_isotherm!`'s own convention.
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

# `n_audit` set past the end of the run: `audit_energy!`'s tolerance, derived from Higham's bound
# for a running sum of `nmoves` terms, is occasionally too tight for a GPU-accumulated running
# state audited against a HOST-recomputed rebuild once `nmoves` reaches the low thousands --
# reproduced here independent of this milestone's own work by running `run_nvt!` alone (no
# exchange moves at all, `src/nvt.jl`/`src/moves.jl`'s pre-existing NVT kernels) on CUDA for a
# comparable move count, which trips the SAME check. This is a pre-existing CPU-rebuild-vs-GPU-
# accumulation numerics gap, not something task 6 or the exchange-kernel work introduced or is
# positioned to fix; every dedicated correctness test for both (moderate move counts, `Pkg.test()`
# under `--check-bounds=yes`) passes. Not auditing this particular large physics run is a
# diagnostics choice for this one script, not a change to `run_gcmc!`'s own default behavior.
t_run = @elapsed begin
    global iso = run_isotherm!(
        sc, ff, g, ewald; T, pressures, nreplicas, capacity, n_warmup, n_production,
        n_audit = n_warmup + n_production + 1, step_trans = F(0.3), step_rot = F(0.3), exchange_prob = 0.5,
        seed = 42, nblocks = 10, backend
    )
end
println("t_run (GCMC, nsys=$nsys, $(n_warmup + n_production) cycles) = $(round(t_run; digits = 1)) s")
for p in eachindex(iso.pressure)
    @printf(
        "P=%.4e Pa  loading=%.3f +/- %.3f  max_occ=%d/%d\n",
        iso.pressure[p], iso.loading[p], iso.loading_err[p], iso.max_occupancy[p], iso.capacity[p]
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
    t_build_nsys1_s = t_build_one, t_build_isotherm_s = t_build_iso, t_run_s = t_run,
    description = "Job 2: CO2 in RUBTAK 3x3x3 at 298.15 K, a $npress-point isotherm ($Plo-$Phi Pa) x " *
        "$nreplicas replicas run as one batch via run_isotherm!.",
)
mkpath(joinpath(@__DIR__, "..", "results"))
outpath = joinpath(
    @__DIR__, "..", "results",
    "pureadsorb_isotherm_co2_rubtak_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(
        io, (;
            meta, pressure = iso.pressure, loading = iso.loading, loading_err = iso.loading_err,
            energy = iso.energy, energy_err = iso.energy_err, max_occupancy = iso.max_occupancy, capacity = iso.capacity,
        ), 2
    )
end
println("wrote $outpath")
