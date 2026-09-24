# Widom throughput benchmark results

## Protocol

`bench/widom_bench.jl` times `widom(batch, guest; ...)` end to end with `Chairmarks.@be`
(`evals = 1`, wall time only — no GPU event timers) over a grid of `(nsys, ninsert)` points,
plus a kernel-only measurement that times one `widom_kernel!` launch (+
`KernelAbstractions.synchronize`) on a single prepared chunk of `2^16` poses, isolating kernel
throughput from the per-chunk RNG fill and host<->device copies that the end-to-end call also
pays. `BLAS.set_num_threads(1)` runs before any timing, since idle OpenBLAS threads spin-wait
and contend with the benchmarked code. Every sample's wall time is written to
`bench/results/*.json`; nothing is derived by re-running a benchmark. `insertions/s = ninsert /
median(times_s)` for the end-to-end `samples`; the kernel-only `kernel_only_s` entries divide
`chunk` by the median of `phase0_times_s` (hard-core rejection) plus `phase1_times_s` (energy)
instead, since those two kernel launches plus a host-side compaction step between them are what
the end-to-end call spends its time on beyond pose generation and transfers.

The grid is smaller on CPU than on a GPU backend, because assembling an `nsys=64` batch (its
Ewald k-vector tables in particular) costs ~9 s on the CPU host and `ninsert=10^6` would run
for minutes per sample there:

| backend | nsys | ninsert | seconds | samples |
|---|---|---|---|---|
| cpu | 1 | 10^4, 10^5 | 10 | 5 |
| cuda / rocm | 1, 64 | 10^4, 10^5, 10^6 | 30 | 10 |

`PA_NINSERT_GRID` (comma-separated, e.g. `10000,100000,1000000`) overrides the `ninsert` sweep
on either backend without touching `nsys_grid` or the time/sample budget above — used for the
CPU f64 three-point rerun below, where `ninsert=10^6` takes ~4.1 s/sample and still fits the
10 s budget for 3 of the 5 requested samples.

The kernel-only measurement always uses `nsys ∈ (1, 64)` with `chunk = 2^16`. The grid actually
used, the host, the GPU name (when applicable), `Threads.nthreads()`, and the benchmark's
`seconds`/`samples` are all recorded in each JSON's `meta` block, so a file is self-describing
regardless of which host produced it. Plain `widom_bench.jl` runs (this section) need no CPU
governor step; `bench/run_headtohead.sh` (below) sets one, since it interleaves kUPS and
PureAdsorb runs on the same host.

Regenerate the plot from the committed JSON, without running anything:

```
julia --project=bench bench/plot_widom.jl
```

## Machines

| host | GPU | backend | precision | result |
|---|---|---|---|---|
| neuromancer | — | cpu | f64 | `pureadsorb_widom_neuromancer_cpu_20260917.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | f64 | `pureadsorb_widom_galen_rocm_f64_20260920_a4c86a7.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | f32 | `pureadsorb_widom_galen_rocm_f32_20260920_a4c86a7.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | f64 | `pureadsorb_widom_galen_rocm_f64_20260920_e903fac.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | f32 | `pureadsorb_widom_galen_rocm_f32_20260920_e903fac.json` |
| neuromancer | NVIDIA GeForce RTX 3050 6GB | cuda | f64 | `pureadsorb_widom_neuromancer_cuda_f64_20260920_e903fac.json` |
| neuromancer | NVIDIA GeForce RTX 3050 6GB | cuda | f32 | `pureadsorb_widom_neuromancer_cuda_f32_20260920_e903fac.json` |
| brutus | — | cpu | f64 | `pureadsorb_widom_brutus_cpu_f64_20260924_97efb80.json` |
| brutus | Apple M6 (12 GPU cores) | metal | f32 | `pureadsorb_widom_brutus_metal_f32_20260924_53f8e40.json` |

The commit-suffixed files above are each the latest for their (host, backend, precision) series;
every earlier file for the same series stays committed but is not otherwise referenced
(`plot_widom.jl` also keeps only the latest file per host/backend/precision). The galen ROCm
files carry a commit suffix because they were re-measured at several points through the
hard-core rejection work — kernel-path insertions/s (`chunk / median(phase0_times_s +
phase1_times_s)`, chunk 65,536) at each point:

| Configuration | Commit | Result file | Float64 (ins/s) | Float32 (ins/s) |
|---|---|---|---|---|
| Full reciprocal table, round-robin assignment | `91966c7` | `pureadsorb_widom_galen_rocm_20260917.json` | 155,142 | not measured (no Float32 support yet) |
| Runs of 256 | `c910867` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920.json` | 167,322 | 3,032,082 |
| Sparse reciprocal table | `584b806` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_584b806.json` | 230,728 | 3,975,541 |
| Cell-sorted atoms | `da327a6` | `pureadsorb_widom_scaling_galen_rocm_{f64,f32}_run256_20260920_da327a6.json` | 253,496 | 4,826,870 |
| Restricted-range pair term | `082c656` | RTX-3050-only measurement; not recorded on the R9700 | — | — |
| Core rejection | `3653c6f` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_3653c6f.json` | 280,754 | 3,503,406 |
| Current | `a4c86a7` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_a4c86a7.json` | 280,241 | 3,478,392 |
| Float64 host-side Boltzmann accumulation, pinned thread schedule | `e903fac` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_e903fac.json` | 280,524 | 3,520,481 |

At nsys=1, Float32, `a4c86a7`, this kernel-path rate (3,478,392 insertions/s) runs faster than
the end-to-end call at 1,000,000 insertions (`pureadsorb_widom_galen_rocm_f32_20260920_a4c86a7.json`,
2,476,918 insertions/s): pose generation, host↔device transfers, the phase-0/phase-1 host-side
compaction step, and block accumulation all run on the host and account for the remaining time.

The RTX 3050 sits behind a Thunderbolt eGPU enclosure on neuromancer, and neuromancer's CPU
clock is unpinned (see the top-level protocol note): its numbers are indicative only, never
gate-authoritative (galen and wintermute are the clock-locked, gate-authoritative hosts).
Consumer GeForce cards throttle double-precision throughput relative to a datacenter part, which
is why the f64/f32 gap on this card (~20x insertions/s) is far larger than the AMD Radeon AI
PRO R9700 numbers above.

## Apple M6 (`brutus`)

First bring-up of PureAdsorb on Apple Silicon: `Pkg.test()`'s 23,581 non-`:gpu`/`:slow` items all
pass (the `:gpu` item is filtered out by `test/runtests.jl` on every host, since it hard-requires
CUDA or AMDGPU). `bench/metal_check.jl` runs the same RUBTAK/CO2 case in Float32 once on
`KernelAbstractions.CPU()` and once on `MetalBackend()`: `mu_ex` and `q_st` match to the last
printed digit (relative difference `0.0`), `K_H` differs by a relative `6.1e-8` — four orders of
magnitude below its own statistical error (`~4.7e-2` relative) — and the phase-0 hard-core
rejection flags (`hardcore_kernel!`, boolean/integer) match exactly, 0 mismatches out of 20,000.
Apple GPUs have no double precision, so only `f32` runs on `metal`; the `cpu`/`f64` row above is
the reference run on the same machine.

Marginal insertion rate (`t = intercept + ninsert/rate`, OLS over `pureadsorb_widom_brutus_*`'s
median times, same fit as `plot_headtohead.jl`):

| backend | precision | nsys | intercept (s) | marginal rate (insertions/s) |
|---|---|---|---|---|
| cpu | f64 | 1 | 0.0052 | 243,011 |
| metal | f32 | 1 | 0.0081 | 2,118,441 |
| metal | f32 | 64 | 0.0080 | 2,112,754 |

The cpu/f64 row uses the three-point `ninsert ∈ {10^4, 10^5, 10^6}` grid
(`pureadsorb_widom_brutus_cpu_f64_20260924_97efb80.json`, `PA_NINSERT_GRID=10000,100000,1000000`);
an earlier two-point run at `10^4, 10^5` only
(`pureadsorb_widom_brutus_cpu_f64_20260924_53f8e40.json`, still committed) gave 240,328
insertions/s, within 1.1% of the three-point fit. Residuals of the three-point fit against the
median times are 0.35 ms (0.75%) at `ninsert=10^4`, -0.39 ms (-0.09%) at `10^5`, and 0.035 ms
(0.001%) at `10^6` — the fit is linear in `ninsert` across two decades, and the two-point rate
holds up.

## Accelerate GEMM feasibility (Apple M6, brutus): no-go

A measurement study of whether the all-pairs distance-matrix identity
`‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b` (a GEMM) is worth pursuing to replace part of the CPU Widom
kernel, on the RUBTAK 3x3x3 + CO2 case (LJ cutoff 12 Å, Ewald cutoff 12 Å, cellwidth 2 Å). Not an
implementation: no `src/` file changed.

**Time decomposition** (`bench/widom_decompose.jl`, chunk = 65,536, phase split from
`pureadsorb_widom_brutus_cpu_{f64,f32}_20260924_*.json`'s `kernel_only_s`, further split within
`insertion_energy` by calling the real, unmodified function four times per precision with
`cutoff`/`ewald_cutoff`/`ks` zeroed one term at a time):

| term | Float64 | Float32 |
|---|---|---|
| distance + cutoff test | 32.1% | 38.5% |
| Lennard-Jones | 32.9% | 41.7% |
| real-space screened Coulomb | 19.8% | 3.4% |
| Ewald reciprocal sum | 13.7% | 14.4% |
| phase-0 cell-list rejection (all insertions) | 1.5% | 1.9% |

The reciprocal sum is 13.7-14.4% of kernel time on this CPU, not the 1-2% measured on GPU — the
GPU split does **not** carry over. `Profile.@profile` over a real `widom()` call segfaults on
this machine (libunwind `stepWithCompactEncoding - invalid compact unwind encoding`, SIGABRT) —
an Apple Silicon libunwind issue, not code-specific; the table above uses the selective-term
fallback instead (see the script for the exact method and its caveats).

**Sparsity ratio**: the only cell list in this code is phase-0's cheap rejection test (1.5-1.9%
of kernel time above). It visits a measured 14.32 framework atoms per guest site on average,
against `natoms = 3078` total framework atoms in this batch — a 215x sparsity ratio. **Phase 1
(`insertion_energy`, the dominant 98.1-98.5% of kernel time) does not use this cell list at
all**: it already loops over all 3078 atoms per surviving pose, gated only by a scalar cutoff
test. A GEMM route does not make phase 1 sparser than it already isn't; the only work a GEMM
route avoids paying cheaply is phase-0's rejection test, which is already the smallest piece of
the kernel.

**Accelerate GEMM rate** (`bench/accelerate_probe.jl`; `AppleAccelerate` v0.7.0, LBT-forwarded;
`BLAS.set_num_threads(8)` for OpenBLAS, `AppleAccelerate.set_num_threads(8)` for Accelerate, which
reported 12 threads back — both an 8-thread request, Accelerate's own toggle picks its own
count):

| shape | OpenBLAS f64 | Accelerate f64 | OpenBLAS f32 | Accelerate f32 |
|---|---|---|---|---|
| thin-K, npose=1024, K=3, natoms=3078 | 58.8 GFLOP/s | 89.4 GFLOP/s | 126.8 GFLOP/s | 130.7 GFLOP/s |
| thin-K, npose=16384 | 20.9 GFLOP/s | 54.0 GFLOP/s | 37.3 GFLOP/s | 86.2 GFLOP/s |
| thin-K, npose=65536 | 21.4 GFLOP/s | 52.8 GFLOP/s | 40.4 GFLOP/s | 80.8 GFLOP/s |
| square, n=1024 (reference) | 241.8 GFLOP/s | 591.5 GFLOP/s | 485.1 GFLOP/s | 2157.8 GFLOP/s |

Accelerate beats OpenBLAS at every shape (2.0-2.5x at thin-K, 2.4-4.4x at square), but the thin-K
shape the distance-matrix route needs falls to 6-11% of Accelerate's own square-GEMM rate
(2.4-4% of OpenBLAS's) — thin-K is far off peak on both BLASes, as expected for K=3.

**The bound**: the distance+cutoff test plus phase-0's rejection (32.1%+1.5%=33.6% of kernel
time, f64; 38.5%+1.9%=40.4%, f32) is the only part a GEMM route can touch; Lennard-Jones,
Coulomb, and the Ewald sum (66.4% f64, 59.6% f32) stay scalar/elementwise regardless. That sets a
hard ceiling — `1/(non-replaceable fraction)` — of **1.51x (f64)** and **1.68x (f32)**, even for
an infinitely fast GEMM. Measured Accelerate thin-K already reaches most of that ceiling: at the
npose=65536 chunk (265.7 ms f64 / 208.1 ms f32 total kernel time), replacing the distance test
with Accelerate's measured thin-K GEMM (22.9 ms f64, 15.0 ms f32) gives an end-to-end speedup of
**1.33x (f64)** and **1.50x (f32)** — 88% and 89% of the theoretical ceiling — against OpenBLAS's
**1.14x (f64)** / **1.35x (f32)**. This ignores the elementwise `‖a‖²+‖b‖²-2ab` combine step and
the memory traffic of writing/reading back the full 65,536x3,078 distance matrix (1.6 GB f64,
0.8 GB f32) that a real implementation would pay on top, so the realistic number is at or below
these already-modest figures, not above them.

**Go/no-go: no-go.** The deciding number is the 66.4%/59.6% (f64/f32) of kernel time spent on
Lennard-Jones, Coulomb and the Ewald sum, which is scalar/elementwise work no GEMM touches — it
caps the best possible end-to-end speedup at 1.5-1.7x, far short of the ~10x measured on generic
matrix operations on this machine. For this to become worthwhile, the elementwise pair terms
(not just the distance test) would need to be vectorized too. `AppleAccelerate.jl`'s own
`VMATH_COVERAGE` docstring enumerates every vForce.h routine it wraps, and `erf`/`erfc` is not
among them — Accelerate has no vectorized erfc at all, so the real-space Coulomb term's
`pair_erfc_dev` (src/ewald.jl) has nothing to be measured against there; that part of item 4 is
unmeasured because the routine does not exist, not because it was skipped. Separately promising:
vForce's `exp` (a routine it does wrap) beat `Base` broadcast `exp` by 2.15x (f64) and 7.11x
(f32) on a 1e6-element array — a real, measured signal that vForce vectorization helps the
transcendentals it covers, just not the one this kernel's Coulomb term actually calls.

Result files: `bench/results/accelerate_gemm_brutus_20260924.json` (item 3-4 raw numbers);
item 1-2 numbers are read directly off `bench/widom_decompose.jl`'s stdout (not saved to JSON,
since they are derived from the already-committed `pureadsorb_widom_brutus_cpu_f64_*.json` kernel
timings plus this script's own selective-term measurements, which are exact function calls, not
samples needing a distribution).

## Batch-size scaling

`bench/widom_scaling.jl` measures kernel throughput against the number of frameworks in one
batch, up to the largest batch each precision holds on the R9700 (galen, ROCm), at two insertion
run lengths:

- Run length 256, `widom`'s own default once a framework receives at least 1024 insertions:
  `pureadsorb_widom_scaling_galen_rocm_f64_run256_20260920.json` (up to 98,304 frameworks) and
  `pureadsorb_widom_scaling_galen_rocm_f32_run256_20260920.json` (up to 196,608 frameworks), both
  at commit `c910867`, before the hard-core rejection stage existed. The commit-suffixed
  `..._584b806.json`/`..._da327a6.json`/`..._3653c6f.json`/`..._a4c86a7.json` files re-measure
  this same run length 256 sweep at 1 and 32,768 frameworks only, at each of those later commits
  — `a4c86a7` is the current one, also giving bytes/framework and the rejected fraction.
- Run length 1 (round-robin), kept as the comparison case, at commit `c910867`:
  `pureadsorb_widom_scaling_galen_rocm_f64_20260920.json` (up to 98,304 frameworks) and
  `pureadsorb_widom_scaling_galen_rocm_f32_20260920.json` (up to 196,608 frameworks).

`pureadsorb_widom_runlength_galen_rocm_20260920.json` sweeps the run length itself, at a fixed
32,768 frameworks, for both precisions: Float64 throughput plateaus by run length 16 (164,029 to
169,265 insertions/s from run length 1); Float32 keeps rising past run length 256, peaking at
run length 4,096 (3,344,249 insertions/s) before leveling off.

## Precision

`PA_PRECISION` selects the element type the benchmark builds and runs in: `f64` (`Float64`,
default) or `f32` (`Float32`). It is recorded in each JSON's `meta.precision` and appears in the
result filename (`pureadsorb_widom_<host>_<backend>_<precision>_<date>.json`). Files predating
this field (`..._20260917.json` above) are all `Float64`; `plot_widom.jl` treats a missing
`meta.precision` as `f64`.

## Running on a GPU host: `bench/gpu`, not `bench`

`bench/Project.toml` also carries `AllocCheck`/`JET`/`StrictMode`/`TrimCheck` for
`bench/audit.jl`, and `AllocCheck` 0.2.6 (the newest release) pins `GPUCompiler` to
`1.3.0-1.23.0`. AMDGPU 2.8.0 needs `GPUCompiler` up to `2.8.1` (resolves to `2.6.0`), so the two
requirements cannot share one environment — `Pkg.resolve()` under `bench/` reports an empty
intersection between `AMDGPU@2.7.0` (the newest version that fits `AllocCheck`'s constraint) and
a tightened `AMDGPU` compat. `bench/gpu/Project.toml` is a second, minimal environment (just
`PureAdsorb`, `StaticArrays`, `Chairmarks`, `JSON`, `KernelAbstractions`, `AMDGPU = "2.8.0"`,
`CUDA = "6.3.1"`, `develop`ing `PureAdsorb` from `../..`) with no audit dependencies, so it
resolves both AMDGPU 2.8.0 and CUDA 6.3.1 cleanly in one environment. `widom_bench.jl` doesn't
care which environment activated it — it writes to `bench/results/` either way — so a GPU host
runs:

```
julia --project=bench/gpu -e 'using Pkg; Pkg.instantiate()'
PA_BACKEND=rocm julia --project=bench/gpu bench/widom_bench.jl
PA_BACKEND=cuda julia --project=bench/gpu bench/widom_bench.jl
```

**AMDGPU 2.7.0 does not compile `widom_kernel!` on gfx1201**: `GPUCompiler`'s IR validator
reports `unsupported dynamic function invocation (call to convert)` once and then segfaults on
repeat attempts while re-emitting that same diagnostic (`typekeyvalue_hash` /
`jl_inst_arg_tuple_type`, from `GPUCompiler/src/validation.jl:297`). **AMDGPU 2.8.0 compiles and
runs it correctly** — confirmed both by `test/gpu_tests.jl`'s `:gpu`-tagged item (`GPU matches
CPU on the same poses`, run via `Pkg.test()` from the package root, which resolves AMDGPU 2.8.0
per `test/Project.toml`) and by the real `bench/gpu` run above. CPU-only work
(`bench/audit.jl`, plotting) keeps using plain `bench/Project.toml`.

`insertion_energy`'s host-side loops use single-array `eachindex`, not the multi-array form:
`eachindex(hpos, htype, hq)` and `eachindex(ks, kprefactor, Shost)` additionally check that every
array shares the same indices, and the mismatch branch builds an error string that GPUCompiler
cannot compile, so `widom_kernel!` failed to compile for CUDA (`InvalidIRError`) while the same
call happened to survive on the ROCm backend. `hpos`/`htype`/`hq` and `ks`/`kprefactor`/`Shost`
are index-matched by construction (slices of the same `FrameworkBatch` arrays), so the
single-array form is exact, not an approximation.

## kUPS head-to-head

`bench/run_headtohead.sh` runs the same RUBTAK/CO2 Widom case on both codes on the same GPU
(RTX 3050, `neuromancer`), alternating a kUPS block with a PureAdsorb block for each
`(nsys, ninsert)` grid point so neither code is measured entirely cold or entirely GPU-warmed
relative to the other. Both codes run float64 (kUPS forces `jax_enable_x64`), the same cutoffs
(LJ 12 Å, Ewald real-space 12 Å, precision 1e-6), the same total insertion count, and the same
force field, host CIF and adsorbate files (verified byte-identical to kUPS's own examples).
Results:

| file | nsys | ninsert |
|---|---|---|
| `kups_widom_timing_neuromancer_f64_20260919.json` | 1, 4 | 10^4, 10^5, 10^6 |
| `pureadsorb_widom_headtohead_neuromancer_f64_nsys{1,4}_ninsert{10000,100000,1000000}_20260919.json` | 1, 4 | 10^4, 10^5, 10^6 |
| `pureadsorb_widom_processcost_neuromancer_f64_20260919.json` | 1 | 10^4 (single point, whole-process cost only) |

`docs/src/benchmarks.md`'s current comparison reuses `kups_widom_timing_neuromancer_f64_20260919.json`
(kUPS does not change between the two) against a later, separately run PureAdsorb series —
`pureadsorb_widom_neuromancer_cuda_{f64,f32}_20260920_e903fac.json` — rather than the interleaved
`..._headtohead_..._20260919.json` files above. `bench/plot_headtohead.jl` draws its figure
(`widom_vs_kups.png`) from those three files using the same OLS fit described below.

### nsys = 64 does not run

kUPS's `num_widom_per_cycle` batches insertions across its systems in parallel, so a larger
`nsys` needs more GPU memory to build the batched state before any cycle runs. On the 6 GB RTX
3050, `nsys ∈ {8, 16, 32, 64}` all fail during state construction with

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate
51.26GiB.   # nsys=64
23.82GiB.   # nsys=32
10.08GiB.   # nsys=16
5.10GiB.    # nsys=8
```

(measured by hand at `ninsert=10000`; exit code 1 in every case, no cycle logged). `nsys=4`
succeeds and is the batched point used above. PureAdsorb runs `nsys=64` without difficulty (see
`pureadsorb_widom_neuromancer_cuda_f64_20260919.json`); the constraint is specific to kUPS's
batched-state memory footprint on this card.

### A `num_cycles=1` kUPS run exits 1, harmlessly

Every kUPS invocation here uses `num_cycles=1` (one compiled `while_loop`, matching
`bench/run_headtohead.sh`'s design note). After that cycle finishes and its HDF5 output is
written, `kups_mcmc_widom`'s `main()` calls `analyze_widom_file` as a convenience print;
`optimal_block_average` there needs at least 8 cycles (`min_blocks=4` needs
`n_samples // 2 >= 4`) and raises `OverflowError: cannot convert float infinity to integer` on
fewer, so the process always exits 1. This happens after the timed work completes and does not
affect the measured wall time. `run_headtohead.sh` distinguishes this exact traceback signature
from a real failure (e.g. the OOM above) and treats only this one as non-fatal; every run is
also checked for a `1/1` completed-cycle line in its log
(`bench/results/logs/`, not committed).

kUPS prints no insertion count, and its HDF5 output is zstd-compressed (filter id 32015) with no
zstd HDF5 plugin on this host, so `h5dump`/`h5ls` cannot decode `n_samples` either — there is no
independent count of insertions actually performed beyond the exit-status and completed-cycle
checks above, plus the linear time-vs-ninsert scaling below.

### Marginal throughput (like-for-like)

kUPS times the whole process (Python/JAX startup, compilation, and the insertions);
PureAdsorb's `times_s` are `Chairmarks` samples, warm in-process. Comparing `ninsert/t` between
them directly would compare a per-process cost against a per-call one, so instead: for each
`nsys`, fit `t = intercept + ninsert/rate` by ordinary least squares over the median time at
each `ninsert`, and compare the fitted `rate` (`plot_widom.jl` uses the same fit for the kUPS
series it draws). PureAdsorb's own fit intercept is near zero, consistent with "warm
in-process":

| code | nsys | intercept (s) | marginal rate (insertions/s) |
|---|---|---|---|
| kUPS (JAX) | 1 | 16.87 | 2,281 |
| kUPS (JAX) | 4 | 18.51 | 4,159 |
| PureAdsorb (CUDA f64, warm in-process) | 1 | 0.092 | 31,128 |
| PureAdsorb (CUDA f64, warm in-process) | 4 | 0.083 | 30,946 |

Ratio (PureAdsorb marginal / kUPS marginal): **13.6×** at nsys=1, **7.4×** at nsys=4 (kUPS
batches more efficiently at nsys=4; PureAdsorb was already near its per-call floor at nsys=1).

### Fixed cost per process

Two different things, both fairly called a "fixed cost": kUPS's regression intercept above is a
cost every kUPS invocation pays (Python/JAX startup and XLA compilation) before any insertion
runs. PureAdsorb's Chairmarks loop excludes that by design (warm in-process), so its own
fixed process cost is measured separately: whole-process wall time (`date +%s.%N` around the
`julia` invocation) of `PA_BACKEND=cuda PA_PRECISION=f64 PA_GRID=1:10000 PA_REPS=1 julia
--project=bench/gpu bench/widom_bench.jl` — Julia startup, package load, kernel compilation and
one warm sample, 3 repetitions:
`pureadsorb_widom_processcost_neuromancer_f64_20260919.json` — 13.44 s, 13.20 s, 13.25 s
(median 13.25 s).

| code | fixed cost per process (s) | what it includes |
|---|---|---|
| kUPS (JAX) | ≈16.9–18.5 (regression intercept) | Python/JAX startup, XLA compilation |
| PureAdsorb (CUDA) | ≈13.2–13.4 (whole-process wall time) | Julia startup, package load, kernel compile |

### Caveats

- Single card, Thunderbolt eGPU enclosure (see the Machines section above), neuromancer's CPU
  clock unpinned.
- Float64 on a GeForce card: double-precision throughput is throttled relative to a datacenter
  part (see the Precision/Machines notes above); this affects both codes equally since both run
  float64 here.
- kUPS forces float64 (`jax_enable_x64=True`) and was not run in float32 for this comparison;
  PureAdsorb's float32 numbers are `pureadsorb_widom_neuromancer_cuda_f32_20260919.json`.
- `bench/run_headtohead.sh` sets the CPU governor to `performance` when writable; on this host
  it was not (`powersave` throughout, recorded in each kUPS JSON's `meta.cpu_governor`).

`plot_widom.jl` regenerates `widom_throughput.png` from the committed JSON only (no benchmark
runs); its kUPS series is plotted as the marginal rate above (`ninsert / (t - intercept)`), not
raw `ninsert/t`, and PureAdsorb's series is labeled "warm in-process".
