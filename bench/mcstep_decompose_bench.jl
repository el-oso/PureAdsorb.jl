# Item 2: re-derive the cis (sincos) fraction and the visit-vs-pair real-space split on the
# ACTUAL mc_step! pipeline (evaluate_move_kernel! -> decide_move_kernel! -> apply_sk_kernel!),
# which replaced the one-thread-per-move kernel the earlier bench/sincos_fraction_bench.jl and
# bench/visit_vs_pair_bench.jl scripts measured. Same disabling-a-term-and-differencing method,
# applied to the new kernel. Float64 and Float32, nsys in {1, 1024}, RUBTAK 3x3x3 + 50 CO2/chain.
using PureAdsorb, StaticArrays, KernelAbstractions, CUDA, Statistics, LinearAlgebra, JSON, Dates
using PureAdsorb: MoveWorkspace, DEFAULT_GROUPSIZE, default_nblocks_per_chain, MOVE_TRANSLATION,
    host_guest_realspace_energy_range, guest_guest_move_delta, select_and_propose, guest_sites_at,
    compact_guest, onehot_movetype
import KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize, @uniform, @ndrange

backend = CUDABackend()

function warm_up!(f!; min_seconds = 0.5)
    f!()
    t0 = time()
    n = 0
    while time() - t0 < min_seconds
        f!()
        n += 1
    end
    return n
end

# Same fake structure-factor kernel as bench/sincos_fraction_bench.jl (P5.1): reads the same
# hoisted site positions and writes the same memory as the real reciprocal term, but replaces
# cis(x) with a comparable-cost, numerically wrong Complex(1-x^2/2, x).
@inline function fake_recip_k(k::SVector{3, T}, charges::SVector{N, T}, old_sites, new_sites, Sk_i::Complex{T}) where {N, T}
    Sold = zero(Complex{T}); Snew = zero(Complex{T})
    for s in 1:N
        xo = dot(k, old_sites[s]); xn = dot(k, new_sites[s])
        Sold += charges[s] * Complex(one(T) - xo * xo / 2, xo)
        Snew += charges[s] * Complex(one(T) - xn * xn / 2, xn)
    end
    ds = Snew - Sold
    return 2 * real(conj(Sk_i) * ds) + abs2(ds)
end
function fake_recip_energy(guest::PureAdsorb.Guest{T, N}, oldpos, oldq, newpos, newq, ks, kprefactor, Sk) where {T, N}
    old_sites = guest_sites_at(guest, oldpos, oldq)
    new_sites = guest_sites_at(guest, newpos, newq)
    ΔU = zero(T)
    for i in eachindex(ks)
        ΔU += kprefactor[i] * fake_recip_k(ks[i], guest.charges, old_sites, new_sites, Sk[i])
    end
    return T(PureAdsorb.KE) * ΔU
end

# Copy of evaluate_move_kernel! (src/moves.jl) with only the reciprocal term's `cis` replaced.
@kernel function fake_evaluate_move_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::PureAdsorb.Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(k_offsets), @Const(rng_seed), @Const(rng_counter),
        movetype::Int32, @Const(step_trans), @Const(step_rot), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    i, oldpos, oldq, newpos, newq, _, _ = select_and_propose(
        n, guest_offsets, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[batch.framework_of[n]],
        refpoints, orientations, length(refpoints)
    )

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    fw = batch.framework_of[n]
    A = batch.cells[fw]
    gr0 = guest_offsets[n]
    a0 = batch.atom_offsets[fw]
    natoms = batch.atom_offsets[fw + 1] - a0
    invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    e_new_partial = host_guest_realspace_energy_range(
        newpos, newq, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, (a0 + lane):L:(a0 + natoms), A, invA, alpha
    )

    gr_lo = gr0 + one(gr0); gr_hi = guest_offsets[n + 1]
    ΔU_gg_partial = guest_guest_move_delta(
        guest, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    kr_lo = k_offsets[n] + one(eltype(k_offsets)); kr_hi = k_offsets[n + 1]
    kr_lo_b = batch.k_offsets[fw] + one(eltype(batch.k_offsets)); kr_hi_b = batch.k_offsets[fw + 1]
    ΔU_recip_partial = fake_recip_energy(
        guest, oldpos, oldq, newpos, newq, view(batch.ks, (kr_lo_b + lane - 1):L:kr_hi_b),
        view(batch.kprefactor, (kr_lo_b + lane - 1):L:kr_hi_b), view(Sk, (kr_lo + lane - 1):L:kr_hi)
    )

    buf1 = @localmem T (G,)
    buf2 = @localmem T (G,)
    buf1[tid] = e_new_partial
    buf2[tid] = ΔU_gg_partial + ΔU_recip_partial
    @synchronize()
    for step in 1:nsteps
        stride = G >> step
        if tid <= stride
            buf1[tid] += buf1[tid + stride]
            buf2[tid] += buf2[tid + stride]
        end
        @synchronize()
    end
    grp2 = @index(Group, NTuple)
    b2, n2 = grp2[2], grp2[3]
    tid == 1 && (partial1[b2, n2] = buf1[1]; partial2[b2, n2] = buf2[1])
end

# Copy of evaluate_move_kernel! with the reciprocal term skipped entirely (not merely made
# numerically wrong): this is what isolates the real-space term's OWN cost (visits plus LJ/Ewald
# pair arithmetic) from reciprocal work when both live in the same kernel launch — running the
# cutoff ablation (`with_cutoffs`, below) against the REAL kernel cannot do this, since even at
# both cutoffs near zero the real kernel still pays for the reciprocal k-loop and decide/apply
# overhead, conflating them with the real-space "visit" residual.
@kernel function no_recip_evaluate_move_kernel!(
        partial1, partial2, @Const(refpoints), @Const(orientations), @Const(Sk), batch, guest::PureAdsorb.Guest{T, N},
        guest_types::SVector{N, Int}, @Const(guest_offsets), @Const(k_offsets), @Const(rng_seed), @Const(rng_counter),
        movetype::Int32, @Const(step_trans), @Const(step_rot), nblocks_per_chain::Int32, ::Val{G}
    ) where {T, N, G}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    b, n = grp[2], grp[3]
    @uniform nsteps = trailing_zeros(G)

    i, oldpos, oldq, newpos, newq, _, _ = select_and_propose(
        n, guest_offsets, rng_seed, rng_counter, movetype, step_trans, step_rot, batch.cells[batch.framework_of[n]],
        refpoints, orientations, length(refpoints)
    )

    L = nblocks_per_chain * G
    lane = (b - one(Int32)) * G + tid

    fw = batch.framework_of[n]
    A = batch.cells[fw]
    gr0 = guest_offsets[n]
    a0 = batch.atom_offsets[fw]
    natoms = batch.atom_offsets[fw + 1] - a0
    invA = batch.invcells[fw]; alpha = batch.alphas[fw]
    e_new_partial = host_guest_realspace_energy_range(
        newpos, newq, guest, batch.sigma, batch.epsilon, batch.cutoff, batch.ewald_cutoff,
        batch.positions, batch.types, batch.charges, (a0 + lane):L:(a0 + natoms), A, invA, alpha
    )

    gr_lo = gr0 + one(gr0); gr_hi = guest_offsets[n + 1]
    ΔU_gg_partial = guest_guest_move_delta(
        guest, refpoints, orientations, (gr_lo + lane - 1):L:gr_hi, i, oldpos, oldq, newpos, newq,
        batch.sigma, batch.epsilon, guest_types, batch.cutoff, batch.ewald_cutoff, A, invA, alpha
    )

    buf1 = @localmem T (G,)
    buf2 = @localmem T (G,)
    buf1[tid] = e_new_partial
    buf2[tid] = ΔU_gg_partial
    @synchronize()
    for step in 1:nsteps
        stride = G >> step
        if tid <= stride
            buf1[tid] += buf1[tid + stride]
            buf2[tid] += buf2[tid + stride]
        end
        @synchronize()
    end
    grp2 = @index(Group, NTuple)
    b2, n2 = grp2[2], grp2[3]
    tid == 1 && (partial1[b2, n2] = buf1[1]; partial2[b2, n2] = buf2[1])
end

# Runs the full 3-kernel mc_step! pipeline, using `evalkern!` in place of the real
# evaluate_move_kernel! (decide/apply are unmodified — their own `cis` cost, in apply_sk_kernel!,
# is not isolated here: it only ever touches ACCEPTED chains' k-vectors, a small fraction of the
# always-executed evaluate step's own reciprocal work).
function mc_step_variant!(evalkern!, ws, batch, state, guest, guest_types, movetype, step_trans, step_rot, kT; backend, groupsize = DEFAULT_GROUPSIZE, nblocks_per_chain = default_nblocks_per_chain(eltype(step_trans), state.nsys))
    G = Int(groupsize); nbpc = Int32(nblocks_per_chain); mt = Int32(movetype); nsys = state.nsys
    evalkern!(backend)(
        ws.partial1, ws.partial2, state.refpoints, state.orientations, state.Sk, batch, guest, guest_types,
        state.guest_offsets, state.k_offsets, state.rng_seed, state.rng_counter, mt, step_trans, step_rot, nbpc, Val(G);
        ndrange = (G, nblocks_per_chain, nsys), workgroupsize = (G, 1, 1)
    )
    KernelAbstractions.synchronize(backend)
    PureAdsorb.decide_move_kernel!(backend)(
        state.refpoints, state.orientations, state.host_energy, state.energy, state.energy_abs_accum,
        state.rng_counter, state.accepted, state.attempted, ws.accept_flag, ws.move_oldpos, ws.move_oldq,
        ws.move_newpos, ws.move_newq, ws.partial1, ws.partial2, batch, guest, state.guest_offsets, state.rng_seed, mt,
        step_trans, step_rot, kT, nbpc; ndrange = nsys
    )
    KernelAbstractions.synchronize(backend)
    PureAdsorb.apply_sk_kernel!(backend)(
        state.Sk, state.sk_abs_accum, batch, ws.accept_flag, ws.move_oldpos, ws.move_oldq, ws.move_newpos,
        ws.move_newq, guest, state.k_offsets; ndrange = (nblocks_per_chain * G, nsys)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# `with_cutoffs`: a copy of `b` with only cutoff/ewald_cutoff changed (visit-vs-pair split).
with_cutoffs(batch::PureAdsorb.FrameworkBatch, cutoff, ewald_cutoff) = PureAdsorb.FrameworkBatch(
    batch.positions, batch.types, batch.charges, batch.atom_offsets, batch.cells, batch.invcells, batch.volumes, batch.alphas,
    batch.ks, batch.kprefactor, batch.Shost, batch.k_offsets, batch.constant_offset, batch.self_term_halfrange,
    batch.ncells, batch.cell_offsets, batch.cellgrid_offsets, batch.sigma, batch.epsilon, batch.compact_to_orig,
    batch.guest_types, batch.guest_types_orig, batch.guest_sites_orig, batch.guest_charges_orig, batch.bs, batch.kmin,
    cutoff, ewald_cutoff, batch.fullk, batch.nsys, batch.framework_of
)
const NEAR_ZERO = 1.0e-6


# GPU-kernel timing noise (OS scheduling, host-side dispatch jitter) is one-sided — it can only
# ADD to the true cost, never subtract from it — so the minimum over many samples is a tighter
# estimate of the kernel's actual cost than the median (the same reasoning BenchmarkTools/
# Chairmarks report minimum for). Reports (min, median) so both are on record.
function timed_ns(step!, nsys; nsamples = 500)
    warm_up!(step!)
    times = Vector{Float64}(undef, nsamples)
    for i in 1:nsamples
        t0 = time_ns()
        step!()
        times[i] = time_ns() - t0
    end
    return minimum(times) / nsys, median(times) / nsys
end

results = []
for F in (Float64, Float32)
    fw = read_cif(joinpath(pkgdir(PureAdsorb), "data", "RUBTAK.cif"); T = F)
    ff = read_forcefield(joinpath(pkgdir(PureAdsorb), "data", "trappe.yaml"); T = F)
    g = read_guest(joinpath(pkgdir(PureAdsorb), "data", "co2.yaml"), ff; T = F)
    sc = replicate(fw, (3, 3, 3))
    ewald = EwaldParams(cutoff = F(12), precision = F(1.0e-6))
    N = length(g.sites)

    for nsys in (1, 1024)
        b = FrameworkBatch(fill(sc, nsys), ff, g, ewald; fullk = true)
        st = SystemState(b, g, fill(50, nsys), ff; T = F(298.15), seed = 1)
        guest_c = compact_guest(b, g)
        guest_types = SVector{N, Int}(b.guest_types)
        step_trans = fill(F(0.3), nsys); step_rot = fill(F(0.3), nsys)
        kT = F(PureAdsorb.KB * 298.15)
        nbpc = default_nblocks_per_chain(F, nsys)
        ws = MoveWorkspace(F, nsys, nbpc; backend)

        # --- real kernel, real batch: baseline ---
        db = PureAdsorb.adapt(backend, b)
        dst = PureAdsorb.adapt(backend, st)
        dstep_trans = PureAdsorb.adapt(backend, step_trans); dstep_rot = PureAdsorb.adapt(backend, step_rot)
        real_step!() = mc_step!(ws, db, dst, guest_c, guest_types, MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, nblocks_per_chain = nbpc)
        real_min, real_med = timed_ns(real_step!, nsys)

        # --- fake-cis kernel, real batch: cis's own share of a full move ---
        dst2 = PureAdsorb.adapt(backend, deepcopy(st))
        fake_step!() = mc_step_variant!(fake_evaluate_move_kernel!, ws, db, dst2, guest_c, guest_types, MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, nblocks_per_chain = nbpc)
        fake_min, fake_med = timed_ns(fake_step!, nsys)
        sincos_fraction = (real_min - fake_min) / real_min

        # --- no-reciprocal kernel, real batch: the reciprocal term's TOTAL share of a full move
        # (not just cis's own share) — skipping the k-loop entirely, not merely making it wrong,
        # is what lets the cutoff ablation below isolate real-space cost cleanly (see
        # `no_recip_evaluate_move_kernel!`'s comment).
        dst5 = PureAdsorb.adapt(backend, deepcopy(st))
        no_recip_step!() = mc_step_variant!(no_recip_evaluate_move_kernel!, ws, db, dst5, guest_c, guest_types, MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, nblocks_per_chain = nbpc)
        no_recip_min, no_recip_med = timed_ns(no_recip_step!, nsys)
        reciprocal_fraction = (real_min - no_recip_min) / real_min

        # --- no-reciprocal kernel, cutoffs collapsed: visit-vs-pair split WITHIN real space,
        # each expressed as its own share of the FULL move (real_min). `no_recip_both0_min` is
        # real-space "visits" (the atom loop still runs; the pair-energy branches never fire)
        # plus decide/apply/launch overhead, with reciprocal work already subtracted out by using
        # the no-reciprocal kernel throughout — unlike the real kernel, whose reciprocal term and
        # decide/apply overhead would otherwise be indistinguishable from "visits" at both
        # cutoffs near zero.
        b_ewald0 = with_cutoffs(b, b.cutoff, F(NEAR_ZERO))
        b_both0 = with_cutoffs(b, F(NEAR_ZERO), F(NEAR_ZERO))
        db_ewald0 = PureAdsorb.adapt(backend, b_ewald0)
        db_both0 = PureAdsorb.adapt(backend, b_both0)
        dst3 = PureAdsorb.adapt(backend, deepcopy(st))
        dst4 = PureAdsorb.adapt(backend, deepcopy(st))
        step_ewald0!() = mc_step_variant!(no_recip_evaluate_move_kernel!, ws, db_ewald0, dst3, guest_c, guest_types, MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, nblocks_per_chain = nbpc)
        step_both0!() = mc_step_variant!(no_recip_evaluate_move_kernel!, ws, db_both0, dst4, guest_c, guest_types, MOVE_TRANSLATION, dstep_trans, dstep_rot, kT; backend, nblocks_per_chain = nbpc)
        ewald0_min, ewald0_med = timed_ns(step_ewald0!, nsys)
        both0_min, both0_med = timed_ns(step_both0!, nsys)
        lj_fraction = (ewald0_min - both0_min) / real_min
        ewald_arith_fraction = (no_recip_min - ewald0_min) / real_min
        visit_and_overhead_fraction = both0_min / real_min
        realspace_fraction = 1 - reciprocal_fraction

        println(
            "F=$F nsys=$nsys real=$(round(real_min; digits = 1))(med $(round(real_med; digits = 1)))ns/move " *
                "sincos_fraction=$(round(sincos_fraction; digits = 3)) reciprocal_fraction=$(round(reciprocal_fraction; digits = 3)) | " *
                "realspace_fraction=$(round(realspace_fraction; digits = 3)) = lj_arith_frac=$(round(lj_fraction; digits = 3)) + " *
                "ewald_arith_frac=$(round(ewald_arith_fraction; digits = 3)) + visit_and_overhead_frac=$(round(visit_and_overhead_fraction; digits = 3)) " *
                "(sum=$(round(reciprocal_fraction + lj_fraction + ewald_arith_fraction + visit_and_overhead_fraction; digits = 3)))"
        )
        flush(stdout)
        push!(
            results, (;
                precision = string(F), nsys, real_min_ns = real_min, real_median_ns = real_med,
                fake_cis_min_ns = fake_min, sincos_fraction, no_recip_min_ns = no_recip_min, reciprocal_fraction,
                realspace_fraction, ewald0_min_ns = ewald0_min, both0_min_ns = both0_min,
                lj_arith_fraction = lj_fraction, ewald_arith_fraction, visit_and_overhead_fraction,
                fraction_sum = reciprocal_fraction + lj_fraction + ewald_arith_fraction + visit_and_overhead_fraction,
            )
        )
    end
end

commit = get(ENV, "PA_COMMIT") do
    try
        readchomp(`git -C $(pkgdir(PureAdsorb)) rev-parse --short HEAD`)
    catch
        "unknown"
    end
end
meta = (;
    host = get(ENV, "PA_HOST", gethostname()), julia = string(VERSION), date = string(now()),
    gpu = CUDA.name(CUDA.device()), backend = "cuda", nguests = 50, commit,
    description = "Item 2 (2026-09-26-milestone-b-nvt.md): the cis (sincos) fraction, the total " *
        "reciprocal share, and the real-space visit-vs-pair split, re-measured on the ACTUAL " *
        "mc_step! pipeline (evaluate_move_kernel!+decide_move_kernel!+apply_sk_kernel!), which " *
        "replaced the one-thread-per-move kernel bench/sincos_fraction_bench.jl and " *
        "bench/visit_vs_pair_bench.jl measured; those numbers no longer describe this code. " *
        "sincos_fraction (real vs a same-cost-but-wrong cis) isolates only evaluate_move_kernel!'s " *
        "own transcendental cost; reciprocal_fraction (real vs a kernel that skips the k-loop " *
        "entirely) is the reciprocal term's TOTAL share, which sincos_fraction alone understates. " *
        "reciprocal_fraction + lj_arith_fraction + ewald_arith_fraction + visit_and_overhead_fraction " *
        "telescope to exactly 1 by construction (each is a same-kernel difference over the same " *
        "real_min baseline); visit_and_overhead_fraction bundles the real-space atom-loop's own " *
        "visit cost with decide_move_kernel!/apply_sk_kernel!/launch overhead, which this ablation " *
        "cannot separate further without a fourth (zero-atom) baseline. apply_sk_kernel!'s own cis " *
        "(only ever touching accepted chains' k-vectors) is not isolated by any of these numbers.",
)
mkpath(joinpath(@__DIR__, "results"))
outpath = joinpath(
    @__DIR__, "results",
    "pureadsorb_mcstepdecompose_$(meta.host)_$(Dates.format(now(), "yyyymmdd"))_$(commit).json"
)
open(outpath, "w") do io
    JSON.print(io, (; meta, results), 2)
end
println("wrote $outpath")
