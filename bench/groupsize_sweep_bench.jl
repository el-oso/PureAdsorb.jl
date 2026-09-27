# Sweeps `mc_step!`'s `groupsize` at a few chain counts and both precisions to pick a default on
# evidence (cleanup 1 of Milestone B task 7): `mc_step_bench.jl`'s own default (256) was never
# compared against the alternatives it was chosen alongside. Same case, same warm-up discipline
# (wall-clock ramp, not a fixed call count) as `mc_step_bench.jl`; this script only adds the
# groupsize axis and writes one combined JSON per (backend, precision) instead of mc_step_bench.jl's
# per-run file, since the groupsize axis would otherwise collide on that script's filename.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NSYS_LIST` is a comma-separated list of chain counts to sweep (default "1,64,256").
# `PA_GROUPSIZE_LIST` is a comma-separated list of groupsizes to sweep (default "32,64,128,256").
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics
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
groupsize_list = parse.(Int, split(get(ENV, "PA_GROUPSIZE_LIST", "32,64,128,256"), ","))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "50"))
bench_seconds = 10
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
kT = F(PureAdsorb.KB * 298.15)

function run_one(nsys, groupsize)
    nblocks = PureAdsorb.default_nblocks_per_chain(F, nsys)
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

    # Warm on wall-clock time, not a fixed call count: this card idles at 210 MHz against a
    # 3,105 MHz boost and needs 100-150 sustained calls (tens of milliseconds) to reach it
    # (`mc_step_bench.jl`'s own note; a single warm-up call already produced a 15x-too-slow
    # reading in this project once).
    PureAdsorb.mc_step!(
        ws, db, dst, guest_c, guest_types, PureAdsorb.MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, groupsize,
        nblocks_per_chain = nblocks
    )
    KernelAbstractions.synchronize(backend)
    t_warmup = time()
    while time() - t_warmup < 0.5
        PureAdsorb.mc_step!(
            ws, db, dst, guest_c, guest_types, PureAdsorb.MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, groupsize,
            nblocks_per_chain = nblocks
        )
        KernelAbstractions.synchronize(backend)
    end

    bm = @be mc_step!(
        $ws, $db, $dst, $guest_c, $guest_types, PureAdsorb.MOVE_TRANSLATION, $dstep_trans, $dstep_rot, $kT; backend = $backend,
        groupsize = $groupsize, nblocks_per_chain = $nblocks
    ) seconds = bench_seconds samples = bench_samples evals = 1

    times_s = [s.time for s in bm.samples]
    per_move = median(times_s) / nsys
    println("nsys=$nsys groupsize=$groupsize t_build=$(round(t_build; digits = 2))s $(per_move * 1.0e6) us/move")
    flush(stdout)
    return (; nsys, groupsize, nblocks_per_chain = nblocks, t_build, times_s, per_move_s = per_move)
end

results = [run_one(nsys, groupsize) for nsys in nsys_list for groupsize in groupsize_list]

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nguests, nsys_list, groupsize_list, bench_seconds, bench_samples, commit,
    description = "Cleanup 1 (Milestone B task 7): mc_step! cost per move against groupsize, at each nsys in " *
        "nsys_list, RUBTAK 3x3x3 + $nguests CO2 per system, one Metropolis translation attempt per chain per call.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_groupsizesweep_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, results), 2)
end
println("wrote $outpath")
