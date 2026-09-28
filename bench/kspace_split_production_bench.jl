# Gate 5 of the kspace-split wiring: does the k-space split's speedup, measured at 1.8-2.0x on
# `guest_move_delta` alone (a real-space-plus-reciprocal move-delta call, not a full production
# kernel launch), survive onto the actual production paths a user runs -- `mc_step!` (one NVT
# translation attempt per chain) and `mc_exchange!` (one μVT insertion-or-deletion attempt per
# chain), both through their normal three-kernel-launch structure (`evaluate_*_kernel!`,
# `decide_*_kernel!`, `apply_*_sk_kernel!`)? Times each with `ewald_gg = nothing` (split off, the
# k-vector table `fullk = true` gives) against `ewald_gg` set (split on), at `nsys = 1` (one
# chain, dominated by kernel-launch and reduction overhead, where the k-loop is a small fraction
# of the total) and at a batch (`nsys = 256`, where per-chain work dominates and the k-loop's own
# share of the kernel should matter more), both Float32 and Float64.
#
# `ewald_gg.cutoff = 16.5` (precision matching the host's own) was chosen by direct measurement
# (not the unverified "1,810 k-vectors" figure quoted in the task): for RUBTAK 3x3x3 + CO2 at
# `ewald.cutoff = 12, precision = 1e-6`, the host-coupled cross table is 190 k-vectors regardless
# of `ewald_gg` (`FrameworkBatch`'s own docstring), and the guest-guest self table falls from the
# unsplit `fullk` table's 4587 k-vectors to 1692 at `cutoff_gg = 16.5` -- close to, not identical
# to, the quoted figure, and the closest feasible point to it before RUBTAK 3x3x3's own minimum
# image bound rejects a wider cutoff.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default), "f32", or "both".
# `PA_NSYS_LIST` is a comma-separated list of chain counts to sweep (default "1,256").
using PureAdsorb, StaticArrays, Chairmarks, JSON, LinearAlgebra, Dates, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cpu")
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

precisions = get(ENV, "PA_PRECISION", "both") == "both" ? (Float64, Float32) : (get(ENV, "PA_PRECISION", "f64") == "f32" ? (Float32,) : (Float64,))
nsys_list = parse.(Int, split(get(ENV, "PA_NSYS_LIST", "1,256"), ","))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "50"))
groupsize = parse(Int, get(ENV, "PA_GROUPSIZE", "256"))
cutoff_gg = parse(Float64, get(ENV, "PA_CUTOFF_GG", "16.5"))
bench_seconds = 10
bench_samples = 10
warmup_seconds = 0.5

# Warms the device to its boosted clock state on wall-clock time (`bench/mc_step_bench.jl`'s own
# rationale and measured ramp: 100-150 calls, tens of milliseconds, on an RTX 4070) rather than a
# fixed call count, so this stays portable to the CPU backend.
function warm_up!(f!::F, seconds) where {F}
    f!()
    KernelAbstractions.synchronize(backend)
    t0 = time()
    while time() - t0 < seconds
        f!()
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end

function build(::Type{F}, nsys, split::Bool) where {F}
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
    ewald_gg = split ? EwaldParams(cutoff = F(cutoff_gg), precision = F(1.0e-6)) : nothing
    b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = !split, ewald_gg)
    capacity = nguests + 10
    st = SystemState(b, g, fill(nguests, nsys), ff; T = F(298.15), seed = 1, capacities = fill(capacity, nsys))
    return b, st, g, ff
end

function bench_case(::Type{F}, nsys::Integer, split::Bool) where {F}
    b, st, g, ff = build(F, nsys, split)
    n_cross = length(PureAdsorb.batch_kvec_range(b, 1))
    n_self = length(PureAdsorb.batch_kvec_gg_range(b, 1))
    N = length(g.sites)
    guest_types = SVector{N, Int}(b.guest_types)
    guest_c = PureAdsorb.compact_guest(b, g)
    step_trans = fill(F(0.3), nsys)
    step_rot = fill(F(0.3), nsys)
    kT = F(PureAdsorb.KB * 298.15)
    p_host, q_host = PureAdsorb.exchange_constant_coeffs(ff, b, guest_c)
    fugacity = fill(2.0e4, nsys)   # Pa: `bench/gpu/exchange_workgroup_bench.jl`'s own choice, well inside capacity

    nblocks = PureAdsorb.default_nblocks_per_chain(F, nsys)
    db = PureAdsorb.adapt(backend, b)
    dst_nvt = PureAdsorb.adapt(backend, deepcopy(st))
    dst_gcmc = PureAdsorb.adapt(backend, deepcopy(st))
    dstep_trans = PureAdsorb.adapt(backend, step_trans)
    dstep_rot = PureAdsorb.adapt(backend, step_rot)
    dp = PureAdsorb.adapt(backend, p_host); dq = PureAdsorb.adapt(backend, q_host)
    ws_nvt = PureAdsorb.MoveWorkspace(F, nsys, nblocks; backend)
    ws_gcmc = PureAdsorb.MoveWorkspace(F, nsys, nblocks; backend)
    rng_gcmc = Xoshiro(0)
    # An exchange move changes occupancy on every acceptance, so an unconstrained warm-up/bench
    # loop drains or saturates it within the warm-up alone -- at that point each call becomes a
    # cheap reject with no sustained work, and the GPU never reaches (or falls back out of) its
    # boost clock between calls, exactly the 15x-too-slow trap `mc_step_bench.jl`'s own comment
    # warns about (measured here: idle-clock exchange calls read several ms against ~200-400 us
    # warm). `occ0`/`copyto!` after every call (`bench/gpu/exchange_workgroup_bench.jl`'s own
    # convention) pins `Ng` at `nguests` regardless of which way acceptance drifts, so every timed
    # call does the same real work and the GPU clock has something to stay boosted on.
    occ0_gcmc = PureAdsorb.adapt(backend, st.occupancy)   # `st` itself is never mutated below
    call_gcmc!() = begin
        r = PureAdsorb.mc_exchange!(
            rng_gcmc, ws_gcmc, db, dst_gcmc, guest_c, guest_types, dp, dq, fugacity, kT; backend, groupsize,
            nblocks_per_chain = nblocks
        )
        copyto!(dst_gcmc.occupancy, occ0_gcmc)
        r
    end

    warm_up!(warmup_seconds) do
        PureAdsorb.mc_step!(
            ws_nvt, db, dst_nvt, guest_c, guest_types, PureAdsorb.MOVE_TRANSLATION, dstep_trans, dstep_rot, kT;
            backend, groupsize, nblocks_per_chain = nblocks
        )
    end
    bm_nvt = @be PureAdsorb.mc_step!(
        $ws_nvt, $db, $dst_nvt, $guest_c, $guest_types, PureAdsorb.MOVE_TRANSLATION, $dstep_trans, $dstep_rot, $kT;
        backend = $backend, groupsize = $groupsize, nblocks_per_chain = $nblocks
    ) seconds = bench_seconds samples = bench_samples evals = 1
    t_nvt = median(s.time for s in bm_nvt.samples) / nsys

    warm_up!(call_gcmc!, warmup_seconds)
    bm_gcmc = @be call_gcmc!() seconds = bench_seconds samples = bench_samples evals = 1
    t_gcmc = median(s.time for s in bm_gcmc.samples) / nsys

    return (; nsys, split, n_cross, n_self, nblocks_per_chain = nblocks, nvt_us = t_nvt * 1.0e6, gcmc_us = t_gcmc * 1.0e6)
end

results = NamedTuple[]
for F in precisions, nsys in nsys_list
    r_off = bench_case(F, nsys, false)
    r_on = bench_case(F, nsys, true)
    speedup_nvt = r_off.nvt_us / r_on.nvt_us
    speedup_gcmc = r_off.gcmc_us / r_on.gcmc_us
    println(
        "$F nsys=$nsys: NVT $(round(r_off.nvt_us; digits = 2))us -> $(round(r_on.nvt_us; digits = 2))us " *
            "($(round(speedup_nvt; digits = 2))x)   GCMC $(round(r_off.gcmc_us; digits = 2))us -> " *
            "$(round(r_on.gcmc_us; digits = 2))us ($(round(speedup_gcmc; digits = 2))x)   " *
            "k: $(r_off.n_cross + r_off.n_self) -> $(r_on.n_cross + r_on.n_self)"
    )
    flush(stdout)
    push!(results, (; precision = string(F), nsys, off = r_off, on = r_on, speedup_nvt, speedup_gcmc))
end

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    nguests, nsys_list, groupsize, cutoff_gg, bench_seconds, bench_samples, commit,
    description = "Gate 5: mc_step!/mc_exchange! per-move cost with the k-space split off (fullk=true) " *
        "against on (ewald_gg at cutoff_gg), RUBTAK 3x3x3 + $nguests CO2 per system.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_kspace_split_production_$(meta.host)_$(backend_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, results), 2)
end
println("wrote $outpath")
