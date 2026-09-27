# The production shape, timing `mc_step!` (task 5's workgroup-per-chain kernel, with the
# `nblocks_per_chain` cross-workgroup fan-out) instead of `chain_sweep_bench.jl`'s
# one-thread-per-proposal `realspace_move_kernel!`/`recip_move_kernel!` pair. Builds `PA_NSYS`
# independent systems (RUBTAK 3x3x3 + `PA_NGUESTS` CO2 each, its own private slice of `Sk` — no
# sharing) and times one `mc_step!` call (one Metropolis move attempt per chain) per sample.
# `nblocks_per_chain` defaults to `default_nblocks_per_chain(F, nsys)` at each `nsys`, so the
# reported cost is what a caller gets by just using `mc_step!`'s own default fan-out, not a
# hand-tuned point. `PA_NBLOCKS_PER_CHAIN` overrides it (a fixed value at every `nsys`) for
# isolating the fan-out's own effect.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NSYS_LIST` is a comma-separated list of chain counts to sweep (default "1,64,256,1024,4096").
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cpu")
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

nsys_list = parse.(Int, split(get(ENV, "PA_NSYS_LIST", "1,64,256,1024,4096"), ","))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "50"))
nblocks_override = haskey(ENV, "PA_NBLOCKS_PER_CHAIN") ? parse(Int, ENV["PA_NBLOCKS_PER_CHAIN"]) : nothing
groupsize = parse(Int, get(ENV, "PA_GROUPSIZE", "256"))
bench_seconds = 10
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
kT = F(PureAdsorb.KB * 298.15)

function run_one(nsys)
    nblocks = something(nblocks_override, PureAdsorb.default_nblocks_per_chain(F, nsys))
    t_build = @elapsed begin
        b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        st = SystemState(b, g, fill(nguests, nsys), ff; T = F(298.15), seed = 1)
    end
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_c = PureAdsorb.compact_guest(b, g)
    step_trans = fill(F(0.3), nsys)
    step_rot = fill(F(0.3), nsys)

    db = PureAdsorb.adapt(backend, b)
    dst = PureAdsorb.adapt(backend, st)
    dstep_trans = PureAdsorb.adapt(backend, step_trans)
    dstep_rot = PureAdsorb.adapt(backend, step_rot)
    ws = PureAdsorb.MoveWorkspace(F, nsys, nblocks; backend)

    # Warm up (compiles every kernel) before timing.
    PureAdsorb.mc_step!(
        ws, db, dst, guest_c, guest_types, PureAdsorb.MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, groupsize,
        nblocks_per_chain = nblocks
    )
    KernelAbstractions.synchronize(backend)

    bm = @be mc_step!(
        $ws, $db, $dst, $guest_c, $guest_types, PureAdsorb.MOVE_TRANSLATION, $dstep_trans, $dstep_rot, $kT; backend = $backend,
        groupsize = $groupsize, nblocks_per_chain = $nblocks
    ) seconds = bench_seconds samples = bench_samples evals = 1

    times_s = [s.time for s in bm.samples]
    per_move = median(times_s) / nsys
    println(
        "nsys=$nsys nblocks_per_chain=$nblocks t_build=$(round(t_build; digits = 2))s " *
            "$(per_move * 1.0e6) us/move"
    )
    flush(stdout)
    return (; nsys, nblocks_per_chain = nblocks, groupsize, t_build, times_s, per_move_s = per_move)
end

results = [run_one(nsys) for nsys in nsys_list]

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nguests, nsys_list, groupsize, bench_seconds, bench_samples, commit,
    description = "Task 5: cost per move (mc_step!, workgroup-per-chain with nblocks_per_chain fan-out) against " *
        "chain count. RUBTAK 3x3x3 + $nguests CO2 per system, one Metropolis translation attempt per chain per call. " *
        "nblocks_per_chain is PureAdsorb.default_nblocks_per_chain(F, nsys) unless PA_NBLOCKS_PER_CHAIN overrides it.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_mcstep_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, results), 2)
end
println("wrote $outpath")
