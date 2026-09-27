# Job 1: `mc_insert!`/`mc_delete!`'s per-call cost after giving the μVT exchange kernels the same
# workgroup-per-chain, `nblocks_per_chain` cross-workgroup fan-out `mc_step!` already has
# (`src/moves.jl`'s own comment on the μVT exchange moves). Same production shape as
# `mc_step_bench.jl`: `PA_NSYS` independent RUBTAK 3x3x3 + CO2 systems, each with its own private
# slice of `Sk`, `nblocks_per_chain` defaulting to `default_nblocks_per_chain(F, nsys)` so the
# reported cost is what a caller gets from `mc_insert!`/`mc_delete!`'s own default fan-out.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NSYS_LIST` is a comma-separated list of chain counts to sweep (default "1,64,256").
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

nsys_list = parse.(Int, split(get(ENV, "PA_NSYS_LIST", "1,64,256"), ","))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "10"))
capacity = parse(Int, get(ENV, "PA_CAPACITY", "40"))
bench_seconds = 10
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
kT = F(PureAdsorb.KB * 298.15)

function run_one(nsys, mover::Symbol)
    nblocks = PureAdsorb.default_nblocks_per_chain(F, nsys)
    t_build = @elapsed begin
        b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        st = SystemState(b, g, fill(nguests, nsys), ff; T = F(298.15), seed = 1, capacities = fill(capacity, nsys))
    end
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_c = PureAdsorb.compact_guest(b, g)
    p_host, q_host = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    fugacity = fill(F(2.0e4), nsys)   # Pa: well inside the chain's capacity for both moves

    db = PureAdsorb.adapt(backend, b)
    dst = PureAdsorb.adapt(backend, st)
    dp = PureAdsorb.adapt(backend, p_host); dq = PureAdsorb.adapt(backend, q_host)
    ws = PureAdsorb.MoveWorkspace(F, nsys, nblocks; backend)
    occ0 = PureAdsorb.adapt(backend, st.occupancy)   # snapshot: pins Ng at `nguests` across every call below

    # A workgroup-fanned exchange move is fast enough that the wall-clock-timed warm-up below fits
    # thousands of calls into 0.5 s (unlike `mc_step!`'s translation, an exchange move changes
    # occupancy on every acceptance, so thousands of unconstrained calls can walk it all the way to
    # `capacity` -- a real failure mode measured here at this fugacity/temperature, where CO2's real
    # adsorption in RUBTAK equilibrates well above `capacity=$capacity`). Resetting occupancy back
    # to its `nguests` snapshot after every call pins the loading this bench measures at, regardless
    # of which way the true equilibrium lies; it leaves `Sk`/`energy`/`host_energy` slightly out of
    # sync with that reset (an accepted move's structure-factor change is not undone), which affects
    # nothing here since only per-call KERNEL cost is timed, never a physical result.
    call() = begin
        r = mover === :insert ?
            PureAdsorb.mc_insert!(ws, db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend) :
            PureAdsorb.mc_delete!(ws, db, dst, guest_c, guest_types, dp, dq, fugacity, kT; backend)
        copyto!(dst.occupancy, occ0)
        r
    end

    # Warm on wall-clock time, not a fixed call count: the card idles at 210 MHz against a 3105
    # MHz boost and needs 100-150 sustained calls (tens of milliseconds) to reach it -- a fixed
    # small warm-up count once gave a 15x-too-slow reading on this hardware (`mc_step_bench.jl`'s
    # own comment).
    call()
    KernelAbstractions.synchronize(backend)
    t_warmup = time()
    while time() - t_warmup < 0.5
        call()
        KernelAbstractions.synchronize(backend)
    end

    bm = @be call() seconds = bench_seconds samples = bench_samples evals = 1
    times_s = [s.time for s in bm.samples]
    per_call = median(times_s)
    println("$mover nsys=$nsys nblocks_per_chain=$nblocks t_build=$(round(t_build; digits = 2))s $(per_call * 1.0e6) us/call")
    flush(stdout)
    return (; nsys, nblocks_per_chain = nblocks, t_build, times_s, per_call_s = per_call)
end

insert_results = [run_one(nsys, :insert) for nsys in nsys_list]
delete_results = [run_one(nsys, :delete) for nsys in nsys_list]

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
tag = get(ENV, "PA_EXCHANGE_BENCH_TAG", "unknown")
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nguests, capacity, nsys_list, bench_seconds, bench_samples, commit, tag,
    description = "Job 1: mc_insert!/mc_delete! per-call cost against chain count, after giving the exchange " *
        "kernels mc_step!'s workgroup-per-chain + nblocks_per_chain fan-out. RUBTAK 3x3x3 + CO2, $nguests initial " *
        "guests/system, capacity $capacity. nblocks_per_chain is PureAdsorb.default_nblocks_per_chain(F, nsys).",
)
mkpath(joinpath(@__DIR__, "..", "results"))
outpath = joinpath(
    @__DIR__, "..", "results",
    "pureadsorb_exchange_workgroup_$(meta.host)_$(backend_name)_$(precision_name)_$(tag)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, insert_results, delete_results), 2)
end
println("wrote $outpath")
