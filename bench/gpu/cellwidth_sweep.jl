# One-off script (not part of the regular suite): phase-0 kernel time and bytes/framework across a cellwidth grid, for RUBTAK 3x3x3 + CO2 (E3's cellwidth choice) -- run as PA_BACKEND=cuda|rocm PA_PRECISION=f64|f32 julia --project=bench/gpu bench/gpu/cellwidth_sweep.jl
using PureAdsorb, StaticArrays, Chairmarks, LinearAlgebra, KernelAbstractions, Statistics, Random
BLAS.set_num_threads(1)

backend_name = get(ENV, "PA_BACKEND", "cuda")
precision_name = get(ENV, "PA_PRECISION", "f64")
F = precision_name == "f32" ? Float32 : Float64
if backend_name == "cuda"
    using CUDA
    backend = CUDABackend()
else
    using AMDGPU
    backend = ROCBackend()
end

fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
sc = replicate(fw, (3, 3, 3))
ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
kernel_chunk = 2^16
N = length(g.sites)

for cellwidth in F.((2, 3, 4))
    b = FrameworkBatch([sc], ff, g, ewald; cellwidth)
    bytes_per_framework = length(b.positions) * (sizeof(eltype(b.positions)) + sizeof(Int32) + sizeof(F)) +
        length(b.ks) * (sizeof(eltype(b.ks)) + sizeof(F) + sizeof(Complex{F})) +
        length(b.cell_offsets) * sizeof(Int32) + 2 * sizeof(eltype(b.ncells)) +
        sizeof(F) + N * size(b.sigma, 1) * sizeof(F)

    g_compact = PureAdsorb.Guest{F, N}(g.sites, SVector{N, Int}(b.guest_types), g.charges, g.tc, g.pc, g.omega)
    kT = F(PureAdsorb.KB * 298.15)
    rho2, reach0, ntypes = PureAdsorb.build_rejection_tables(b, g_compact, kT)
    rng = Xoshiro(0)
    sys_of = Vector{Int32}(undef, kernel_chunk)
    rpos = Vector{SVector{3, F}}(undef, kernel_chunk)
    quat = Vector{SVector{4, F}}(undef, kernel_chunk)
    flags = Vector{UInt8}(undef, kernel_chunk)
    PureAdsorb.random_poses!(rng, sys_of, rpos, quat, 1, PureAdsorb.default_run(kernel_chunk, 1), 1)
    dbatch = PureAdsorb.adapt(backend, b)
    dsys, drpos, dquat = PureAdsorb.adapt(backend, sys_of), PureAdsorb.adapt(backend, rpos), PureAdsorb.adapt(backend, quat)
    dflags = PureAdsorb.adapt(backend, flags)
    drho2, dreach0 = PureAdsorb.adapt(backend, rho2), PureAdsorb.adapt(backend, reach0)
    kern0 = PureAdsorb.hardcore_kernel!(backend)

    kern0(dflags, dsys, drpos, dquat, dbatch, g_compact, drho2, dreach0, Int32(ntypes); ndrange = kernel_chunk)
    KernelAbstractions.synchronize(backend)
    copyto!(flags, dflags)
    rejected_frac = count(!iszero, flags) / kernel_chunk

    bm = @be (
        kern0($dflags, $dsys, $drpos, $dquat, $dbatch, $g_compact, $drho2, $dreach0, $(Int32(ntypes)); ndrange = $kernel_chunk);
        KernelAbstractions.synchronize($backend)
    ) seconds = 20 samples = 10 evals = 1
    times = [s.time for s in bm.samples]
    println(
        "cellwidth=$cellwidth phase0_median=$(round(1000 * median(times); digits = 1)) ms " *
            "rejected=$(round(100 * rejected_frac; digits = 1))% bytes/framework=$bytes_per_framework"
    )
    flush(stdout)
end
