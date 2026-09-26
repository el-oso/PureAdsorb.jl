# Transcendental (sincos) fraction of the reciprocal move kernel's cost, per P5.1. A bench-local
# copy of the (post-hoist) reciprocal path replaces `cis(x)` with `Complex(one(T) - x*x/2, x)`:
# same memory traffic (one k-vector, N site positions, one running Sk entry, per k-vector), a
# comparable flop count (one multiply-subtract and one multiply in place of `cis`'s sincos), but
# numerically wrong. The measured time difference against the real kernel is the fraction of the
# kernel's cost spent evaluating `cis` itself, isolated from everything else the kernel does.
# Must run after the rotation hoist (`src/guest.jl`), or it measures unhoisted rotations instead.
#
# `PA_BACKEND` selects the KernelAbstractions backend: "cpu" (default), "cuda", "rocm".
# `PA_PRECISION` selects the element type: "f64" (default) or "f32".
# `PA_NMOVES` sets the number of independent move proposals launched per kernel call (default
# 65536, matching `guest_bench.jl`'s chunk size).
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

nmoves = parse(Int, get(ENV, "PA_NMOVES", "65536"))
nguests = parse(Int, get(ENV, "PA_NGUESTS", "50"))
bench_seconds = 20
bench_samples = 10

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
b = FrameworkBatch([sc], ff, g, ewald; fullk = true)
nk = length(b.ks)

st = SystemState(b, g, [nguests], ff; T = F(298.15), seed = 1)

rng = Xoshiro(0)
gr = PureAdsorb.guest_range(st, 1)
sys_of = fill(Int32(1), nmoves)
gidx = Int32[rand(rng, gr) for _ in 1:nmoves]
oldpos = [st.refpoints[i] for i in gidx]
oldq = [st.orientations[i] for i in gidx]
newpos = [st.refpoints[i] + SVector{3, F}((rand(rng, F, 3) .- F(0.5)) .* F(2)) for i in gidx]
newq = [normalize(SVector{4, F}(rand(rng, F, 4) .- F(0.5))) for _ in 1:nmoves]

# A deliberately wrong per-k structure-factor kernel: `Complex(1 - x^2/2, x)` reads the same
# `guest_sites_at`-hoisted site positions and writes the same memory (one `ΔU` per proposal) as
# the real reciprocal kernel, differing only in the transcendental evaluation itself.
@inline function fake_reciprocal_move_delta_k(
        k::SVector{3, T}, charges::SVector{N, T}, old_sites::SVector{N, SVector{3, T}}, new_sites::SVector{N, SVector{3, T}}, Sk_i::Complex{T}
    ) where {N, T}
    Sold = zero(Complex{T}); Snew = zero(Complex{T})
    for s in 1:N
        xo = dot(k, old_sites[s]); xn = dot(k, new_sites[s])
        Sold += charges[s] * Complex(one(T) - xo * xo / 2, xo)
        Snew += charges[s] * Complex(one(T) - xn * xn / 2, xn)
    end
    ds = Snew - Sold
    return ds, 2 * real(conj(Sk_i) * ds) + abs2(ds)
end

function fake_reciprocal_move_delta_energy(
        guest::PureAdsorb.Guest{T, N}, oldpos::SVector{3, T}, oldq::SVector{4, T}, newpos::SVector{3, T}, newq::SVector{4, T},
        ks, kprefactor, Sk
    ) where {T, N}
    old_sites = PureAdsorb.guest_sites_at(guest, oldpos, oldq)
    new_sites = PureAdsorb.guest_sites_at(guest, newpos, newq)
    ΔU = zero(T)
    for i in eachindex(ks)
        _, contribution = fake_reciprocal_move_delta_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
        ΔU += kprefactor[i] * contribution
    end
    return T(PureAdsorb.KE) * ΔU
end

@kernel function fake_recip_move_kernel!(ΔU, @Const(sys_of), @Const(oldpos), @Const(oldq), @Const(newpos), @Const(newq), batch, guest, Sk, k_offsets)
    m = @index(Global)
    n = sys_of[m]
    kr = (k_offsets[n] + 1):k_offsets[n + 1]
    ΔU[m] = fake_reciprocal_move_delta_energy(
        guest, oldpos[m], oldq[m], newpos[m], newq[m], view(batch.ks, kr), view(batch.kprefactor, kr), view(Sk, kr)
    )
end

backend === KernelAbstractions.CPU() || (
    global sys_of = PureAdsorb.adapt(backend, sys_of); global oldpos = PureAdsorb.adapt(backend, oldpos); global oldq = PureAdsorb.adapt(backend, oldq);
    global newpos = PureAdsorb.adapt(backend, newpos); global newq = PureAdsorb.adapt(backend, newq)
)
db = PureAdsorb.adapt(backend, b)
dSk = PureAdsorb.adapt(backend, st.Sk)
dk_offsets = PureAdsorb.adapt(backend, st.k_offsets)
dΔU_real = PureAdsorb.adapt(backend, zeros(F, nmoves))
dΔU_fake = PureAdsorb.adapt(backend, zeros(F, nmoves))

kern_real = PureAdsorb.recip_move_kernel!(backend)
kern_fake = fake_recip_move_kernel!(backend)

kern_real(dΔU_real, sys_of, oldpos, oldq, newpos, newq, db, g, dSk, dk_offsets; ndrange = nmoves)
KernelAbstractions.synchronize(backend)   # warm-up: compile
kern_fake(dΔU_fake, sys_of, oldpos, oldq, newpos, newq, db, g, dSk, dk_offsets; ndrange = nmoves)
KernelAbstractions.synchronize(backend)   # warm-up: compile

bm_real = @be (
    kern_real($dΔU_real, $sys_of, $oldpos, $oldq, $newpos, $newq, $db, $g, $dSk, $dk_offsets; ndrange = $nmoves);
    KernelAbstractions.synchronize($backend)
) seconds = bench_seconds samples = bench_samples evals = 1
bm_fake = @be (
    kern_fake($dΔU_fake, $sys_of, $oldpos, $oldq, $newpos, $newq, $db, $g, $dSk, $dk_offsets; ndrange = $nmoves);
    KernelAbstractions.synchronize($backend)
) seconds = bench_seconds samples = bench_samples evals = 1

real_times_s = [s.time for s in bm_real.samples]
fake_times_s = [s.time for s in bm_fake.samples]
real_per_move_s = median(real_times_s) / nmoves
fake_per_move_s = median(fake_times_s) / nmoves
sincos_fraction = (real_per_move_s - fake_per_move_s) / real_per_move_s
println(
    "backend=$backend_name precision=$precision_name nk=$nk real=$(real_per_move_s * 1.0e9) ns/move " *
        "fake=$(fake_per_move_s * 1.0e9) ns/move sincos_fraction=$sincos_fraction"
)
flush(stdout)

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()), gpu, backend = backend_name,
    precision = precision_name, nk, nguests, nmoves, bench_seconds, bench_samples, commit,
    description = "P5.1: reciprocal move kernel per-move cost with the real cis(x) versus a deliberately wrong, " *
        "comparable-cost Complex(1-x^2/2, x) in its place, both post-hoist (src/guest.jl). The relative difference " *
        "is the transcendental (sincos) fraction of the reciprocal kernel's own cost.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_sincosfraction_$(meta.host)_$(backend_name)_$(precision_name)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, real_times_s, fake_times_s, real_per_move_s, fake_per_move_s, sincos_fraction), 2)
end
println("wrote $outpath")
