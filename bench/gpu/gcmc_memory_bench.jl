# Job 1 (task 10): PureAdsorb's own per-system device footprint for a GCMC `SystemState`, to set
# against kUPS's batched-state memory ceiling (`bench/results/README.md`'s "kUPS GCMC memory
# ceiling" section: 13.18 GiB requested at nsys=16, RUBTAK 3x3x3 + CO2, RTX 4070). Marginal bytes
# per system are computed analytically from field lengths and element sizes, the same pattern
# `bench/widom_scaling.jl`'s `bytes_per_system` and `bench/gpu/cellwidth_sweep.jl`'s
# `bytes_per_framework` already use for `FrameworkBatch`; a `SystemState`'s two largest fields,
# `Sk` and `sk_abs_accum`, are sized by the FULL k-vector table (`fullk = true`, required once any
# guest is present, `docs/src/theory.md`'s "Which k-vectors each term needs"), not the
# host-coupled subset a guest-free `FrameworkBatch` keeps.
#
# `PA_CAPACITY` matches the isotherm benchmark's own per-system capacity (200,
# `pureadsorb_isotherm_co2_rubtak_neuromancer_cuda_f64_20260927_873c9ed.json`).
using PureAdsorb, JSON, Dates
using PureAdsorb: NMOVETYPES

capacity = parse(Int, get(ENV, "PA_CAPACITY", "200"))

function bytes_per_system(::Type{F}, capacity::Integer, nk::Integer) where {F}
    fs = sizeof(F)
    return capacity * (3 * fs + 4 * fs + fs) +   # refpoints + orientations + host_energy
        nk * (2 * fs + fs) +                     # Sk (Complex{F}) + sk_abs_accum
        2 * fs +                                 # energy + energy_abs_accum
        2 * sizeof(UInt64) +                     # rng_seed + rng_counter
        2 * NMOVETYPES * sizeof(Int32) +         # accepted + attempted
        2 * sizeof(Int32)                        # marginal guest_offsets + k_offsets entries
end

fw64 = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float64)
ff64 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float64)
g64 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff64; T = Float64)
sc64 = replicate(fw64, (3, 3, 3))
ewald64 = EwaldParams(cutoff = 12.0, precision = 1.0e-6)
b64 = FrameworkBatch([sc64], ff64, g64, ewald64; fullk = true)
nk = length(b64.ks)

fw32 = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = Float32)
ff32 = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = Float32)
g32 = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff32; T = Float32)
sc32 = replicate(fw32, (3, 3, 3))
ewald32 = EwaldParams(cutoff = 12.0f0, precision = 1.0f-6)
b32 = FrameworkBatch([sc32], ff32, g32, ewald32; fullk = true)
nk32 = length(b32.ks)
nk == nk32 || throw(AssertionError("k-vector count must not depend on precision: $nk vs $nk32"))

bytes_f64 = bytes_per_system(Float64, capacity, nk)
bytes_f32 = bytes_per_system(Float32, capacity, nk)
println("nk=$nk capacity=$capacity")
println("bytes/system (Float64) = $bytes_f64 ($(round(bytes_f64 / 2^10; digits = 1)) KiB)")
println("bytes/system (Float32) = $bytes_f32 ($(round(bytes_f32 / 2^10; digits = 1)) KiB)")

commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), capacity, nk, commit,
    description = "Job 1 (task 10): SystemState's own per-system device footprint (RUBTAK 3x3x3 + " *
        "CO2, fullk=true), computed analytically from field lengths -- refpoints/orientations/" *
        "host_energy scale with capacity, Sk/sk_abs_accum scale with the full k-vector table.",
)
mkpath(joinpath(@__DIR__, "..", "results"))
outpath = joinpath(@__DIR__, "..", "results", "pureadsorb_gcmc_memory_$(meta.host)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json")
open(outpath, "w") do io
    JSON.print(io, (; meta, bytes_per_system_f64 = bytes_f64, bytes_per_system_f32 = bytes_f32), 2)
end
println("wrote $outpath")
