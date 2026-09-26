# The production shape (per P5.3): one chain per system, each with its OWN private slice of `Sk`
# (no sharing), sweeping the number of chains to find where occupancy saturates. Unlike
# `guest_bench.jl`'s single-framework, 65536-shared-`Sk` measurement (an L1 broadcast, explicitly
# not a chain — see `bench/results/README.md`), this builds `PA_NSYS` independent systems, each
# with `PA_NGUESTS` guests, and launches exactly ONE move proposal per system per kernel call
# (`ndrange = nsys`), which is what a real batch of chains looks like at the kernel-launch level.
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
bench_seconds = 10
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
N = length(g.sites)

function run_one(nsys)
    t_build = @elapsed begin
        b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        st = SystemState(b, g, fill(nguests, nsys), ff; T = F(298.15), seed = 1)
    end
    nk_total = length(b.ks)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_compact = PureAdsorb.Guest{F, N}(g.sites, guest_types, g.charges, g.tc, g.pc, g.omega)

    # One proposal per system: a random guest in that system, a Gaussian displacement.
    rng = Xoshiro(0)
    sys_of = Int32.(1:nsys)
    gidx = Int32[rand(rng, PureAdsorb.guest_range(st, n)) for n in 1:nsys]
    oldpos = [st.refpoints[i] for i in gidx]
    oldq = [st.orientations[i] for i in gidx]
    newpos = [st.refpoints[i] + SVector{3, F}((rand(rng, F, 3) .- F(0.5)) .* F(2)) for i in gidx]
    newq = [normalize(SVector{4, F}(rand(rng, F, 4) .- F(0.5))) for _ in 1:nsys]

    backend === KernelAbstractions.CPU() || (
        sys_of = PureAdsorb.adapt(backend, sys_of); gidx = PureAdsorb.adapt(backend, gidx);
        oldpos = PureAdsorb.adapt(backend, oldpos); oldq = PureAdsorb.adapt(backend, oldq);
        newpos = PureAdsorb.adapt(backend, newpos); newq = PureAdsorb.adapt(backend, newq)
    )
    db = PureAdsorb.adapt(backend, b)
    drefpoints = PureAdsorb.adapt(backend, st.refpoints)
    dorientations = PureAdsorb.adapt(backend, st.orientations)
    dguest_offsets = PureAdsorb.adapt(backend, st.guest_offsets)
    dSk = PureAdsorb.adapt(backend, st.Sk)
    dk_offsets = PureAdsorb.adapt(backend, st.k_offsets)
    dΔU_real = PureAdsorb.adapt(backend, zeros(F, nsys))
    dΔU_recip = PureAdsorb.adapt(backend, zeros(F, nsys))

    kern_real = PureAdsorb.realspace_move_kernel!(backend)
    kern_recip = PureAdsorb.recip_move_kernel!(backend)

    kern_real(dΔU_real, sys_of, gidx, oldpos, oldq, newpos, newq, db, guest_compact, drefpoints, dorientations, dguest_offsets, guest_types; ndrange = nsys)
    KernelAbstractions.synchronize(backend)
    kern_recip(dΔU_recip, sys_of, oldpos, oldq, newpos, newq, db, g, dSk, dk_offsets; ndrange = nsys)
    KernelAbstractions.synchronize(backend)

    bm_real = @be (
        kern_real($dΔU_real, $sys_of, $gidx, $oldpos, $oldq, $newpos, $newq, $db, $guest_compact, $drefpoints, $dorientations, $dguest_offsets, $guest_types; ndrange = $nsys);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1
    bm_recip = @be (
        kern_recip($dΔU_recip, $sys_of, $oldpos, $oldq, $newpos, $newq, $db, $g, $dSk, $dk_offsets; ndrange = $nsys);
        KernelAbstractions.synchronize($backend)
    ) seconds = bench_seconds samples = bench_samples evals = 1

    real_times_s = [s.time for s in bm_real.samples]
    recip_times_s = [s.time for s in bm_recip.samples]
    real_per_move = median(real_times_s) / nsys
    recip_per_move = median(recip_times_s) / nsys
    total_per_move = real_per_move + recip_per_move
    println(
        "nsys=$nsys t_build=$(round(t_build; digits = 2))s real=$(real_per_move * 1.0e9) ns/move " *
            "recip=$(recip_per_move * 1.0e9) ns/move total=$(total_per_move * 1.0e9) ns/move"
    )
    flush(stdout)
    return (;
        nsys, nk_total, t_build, real_times_s, recip_times_s, real_per_move_s = real_per_move, recip_per_move_s = recip_per_move,
        total_per_move_s = total_per_move,
    )
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
    precision = precision_name, nguests, nsys_list, bench_seconds, bench_samples, commit,
    description = "P5.3: cost per move against chain count, one chain per system with its OWN private Sk slice " *
        "(no sharing), one move proposal per system per kernel launch (ndrange = nsys). RUBTAK 3x3x3 + " *
        "$nguests CO2 per system. Decides whether occupancy saturates at a batch size anyone would actually run.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_chainsweep_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, results), 2)
end
println("wrote $outpath")
