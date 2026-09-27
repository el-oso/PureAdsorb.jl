# Decomposes mc_insert!'s per-call cost, RUBTAK 3x3x3 + CO2 at nsys=1, into its component host and
# device operations: the small per-call device allocations (`adapt` for the fugacity and
# capacity-hits arrays), the `mc_insert_kernel!` launch itself, and the capacity-hits readback
# (`Array(dhits)`). This isolates that the kernel launch dominates -- one GPU thread per chain
# serially summing the reciprocal-space loop over every k-vector, no workgroup fan-out (this file's
# own comment on the μVT exchange moves, `src/moves.jl`) -- from the small per-call host/device
# transfers `mc_insert!`/`mc_delete!` still make.
using PureAdsorb, StaticArrays, Chairmarks, CUDA, LinearAlgebra, Random, Statistics
const adapt = PureAdsorb.adapt
BLAS.set_num_threads(1)

F = Float64
backend = CUDABackend()

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
fugacity = F[2.0e4]

# Warm up on wall-clock time.
for _ in 1:5
    PureAdsorb.mc_insert!(db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
end
CUDA.synchronize()
t_warmup = time()
while time() - t_warmup < 1.0
    PureAdsorb.mc_insert!(db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
    CUDA.synchronize()
end

nsys = 1
println("--- component costs (median of 200 calls each, warmed) ---")

t1 = median([(@elapsed (adapt(backend, F(PureAdsorb.PASCAL) .* F.(fugacity)); CUDA.synchronize())) for _ in 1:200])
println("dfug = adapt(backend, ...):        $(t1 * 1.0e6) us")

t2 = median([(@elapsed (adapt(backend, zeros(UInt8, nsys)); CUDA.synchronize())) for _ in 1:200])
println("dhits = adapt(backend, zeros...):  $(t2 * 1.0e6) us")

dfug = adapt(backend, F(PureAdsorb.PASCAL) .* F.(fugacity))
dhits = adapt(backend, zeros(UInt8, nsys))
t3 = median(
    [
        (
            @elapsed begin
                PureAdsorb.mc_insert_kernel!(backend)(
                    dst.refpoints, dst.orientations, dst.host_energy, dst.occupancy, dst.energy, dst.energy_abs_accum,
                    dst.Sk, dst.sk_abs_accum, dst.rng_counter, dhits, db, guest_c, guest_types, dst.guest_offsets,
                    dst.k_offsets, dst.rng_seed, dfug, kT, dp, dq; ndrange = nsys
                )
                CUDA.synchronize()
            end
        ) for _ in 1:200
    ]
)
println("mc_insert_kernel! launch+sync:      $(t3 * 1.0e6) us")

t4 = median([(@elapsed Array(dhits)) for _ in 1:200])
println("Array(dhits) (D2H + implicit sync): $(t4 * 1.0e6) us")

t5 = median([(@elapsed PureAdsorb.mc_insert!(db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)) for _ in 1:200])
println("full mc_insert! call:               $(t5 * 1.0e6) us")
