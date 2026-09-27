# Measures mc_exchange!'s per-call cost on CUDA, RUBTAK 3x3x3 + CO2, nsys=1: `mc_insert!` and
# `mc_delete!` take the pose-independent exchange coefficients (`exchange_constant_coeffs`) as a
# precomputed device array, so neither derives them from `batch` -- a scalar-indexing computation
# that would otherwise force a full device-to-host copy of the whole `FrameworkBatch` on every
# call -- inside the hot path. That round trip is not the only cost here:
# `mc_insert_kernel!`/`mc_delete_kernel!`'s own single-work-item-per-chain reciprocal-space loop
# (`exchange_bench_decompose.jl`) costs far more than `mc_step!`'s ~187 us/move on the same
# hardware; this file reports the actual number rather than assuming it matches. Same wall-clock
# warm-up discipline as mc_step_bench.jl: the card idles at 210 MHz against a 3105 MHz boost and
# needs 100-150 sustained calls to reach it.
using PureAdsorb, StaticArrays, Chairmarks, CUDA, LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)

F = Float64
backend = CUDABackend()
nsys = 1

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
kT = F(PureAdsorb.KB * 298.15)

b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
st = SystemState(b, g, [10], ff; T = F(298.15), seed = 1, capacities = [40])
N = length(g.sites)
guest_types = SVector{N, Int}(b.guest_types)
guest_c = PureAdsorb.compact_guest(b, g)

p, q = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
db = PureAdsorb.adapt(backend, b)
dst = PureAdsorb.adapt(backend, st)
dp = PureAdsorb.adapt(backend, p); dq = PureAdsorb.adapt(backend, q)
ws = PureAdsorb.MoveWorkspace(F, nsys, PureAdsorb.default_nblocks_per_chain(F, nsys); backend)
fugacity = F[2.0e4]

# Warm up on wall-clock time, not a fixed call count (see this file's header).
PureAdsorb.mc_exchange!(Xoshiro(1), ws, db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
CUDA.synchronize()
t_warmup = time()
while time() - t_warmup < 0.5
    PureAdsorb.mc_exchange!(Xoshiro(1), ws, db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
    CUDA.synchronize()
end

rng = Xoshiro(1)
bm = @be PureAdsorb.mc_exchange!($rng, $ws, $db, $dst, $guest_c, $guest_types, $dp, $dq, $fugacity, $kT; backend = $backend) seconds = 10 samples = 10 evals = 1
times_s = [s.time for s in bm.samples]
per_call = median(times_s)
println("mc_exchange! nsys=$nsys: $(per_call * 1.0e6) us/call (median of $(length(times_s)) samples)")

using JSON, Dates
commit = try
    readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
catch
    "unknown"
end
meta = (;
    host = gethostname(), julia = string(VERSION), date = string(now()), gpu = CUDA.name(CUDA.device()),
    backend = "cuda", precision = "f64", nsys, commit,
    description = "Job 1: mc_exchange! per-call cost, RUBTAK 3x3x3 + CO2, nsys=1, 10 initial guests, " *
        "capacity 40 -- before/after removing the per-call adapt(CPU(), batch) in mc_insert!/mc_delete!.",
)
mkpath(joinpath(@__DIR__, "..", "results"))
tag = get(ENV, "PA_EXCHANGE_BENCH_TAG", "unknown")
outpath = joinpath(@__DIR__, "..", "results", "pureadsorb_exchange_percall_neuromancer4070_cuda_f64_$(tag)_$(commit).json")
open(outpath, "w") do io
    JSON.print(io, (; meta, times_s, per_call_s = per_call))
end
println("wrote $outpath")
