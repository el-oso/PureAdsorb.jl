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
| neuromancer4070 | NVIDIA GeForce RTX 4070 12GB | cuda | f64 | `pureadsorb_widom_neuromancer4070_cuda_f64_20260926_b26cb8a.json` |
| neuromancer4070 | NVIDIA GeForce RTX 4070 12GB | cuda | f32 | `pureadsorb_widom_neuromancer4070_cuda_f32_20260926_b26cb8a.json` |

The RTX 4070 above replaced the RTX 3050 in the same Thunderbolt/USB4 eGPU enclosure slot on
`neuromancer` (see the "RTX 4070 eGPU link" section below). Its rows carry `PA_HOST=neuromancer4070`
rather than the bare `neuromancer` the 3050 rows use, so the filename's host token — which this
file's self-describing-JSON convention keeps equal to `meta.host` throughout — also names the
card, without adding a new field to the naming scheme. `CUDA.jl` needed no special handling to
pick up the 4070: `Pkg.instantiate()` under `bench/gpu` and `CUDA.versioninfo()` both ran clean
on the first attempt (driver 615.71.09, CUDA runtime 13.3.0, `sm_89`, 11.66 GiB free of 11.99 GiB
total), unlike the AMDGPU 2.7.0 compile failure noted for the R9700 below.

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

## RTX 4070 (`neuromancer4070`)

The RTX 4070 (12 GB) replaced the RTX 3050 (6 GB) in the same USB4 eGPU enclosure slot on
`neuromancer`. `bench/widom_bench.jl`'s own grid (nsys 1 and 64; ninsert 10^4, 10^5, 10^6) ran at
commit `b26cb8a`:

| file | precision |
|---|---|
| `pureadsorb_widom_neuromancer4070_cuda_f64_20260926_b26cb8a.json` | f64 |
| `pureadsorb_widom_neuromancer4070_cuda_f32_20260926_b26cb8a.json` | f32 |

Marginal insertion rate (`t = intercept + ninsert/rate`, OLS over the median time at each
`ninsert`, same fit as `plot_headtohead.jl`), with residuals against the fitted line at each
point (the three-point grid spans two decades, so a straight-line fit does not track every point
exactly — the residuals below are reported rather than hidden):

| precision | nsys | intercept (s) | marginal rate (insertions/s) |
|---|---|---|---|
| f64 | 1 | 0.0436 | 264,973 |
| f64 | 64 | 0.0448 | 263,806 |
| f32 | 1 | 0.0052 | 3,812,315 |
| f32 | 64 | 0.0067 | 3,799,434 |

| precision | nsys | ninsert | t_median (s) | residual (s, %) |
|---|---|---|---|---|
| f64 | 1 | 10^4 | 0.12455 | +0.0433 (34.7%) |
| f64 | 1 | 10^5 | 0.37339 | -0.0476 (-12.7%) |
| f64 | 1 | 10^6 | 3.82185 | +0.0043 (0.11%) |
| f64 | 64 | 10^4 | 0.12683 | +0.0441 (34.8%) |
| f64 | 64 | 10^5 | 0.37543 | -0.0485 (-12.9%) |
| f64 | 64 | 10^6 | 3.83992 | +0.0044 (0.12%) |
| f32 | 1 | 10^4 | 0.00995 | +0.0022 (21.6%) |
| f32 | 1 | 10^5 | 0.02904 | -0.0024 (-8.1%) |
| f32 | 1 | 10^6 | 0.26770 | +0.0002 (0.08%) |
| f32 | 64 | 10^4 | 0.01184 | +0.0025 (21.0%) |
| f32 | 64 | 10^5 | 0.03031 | -0.0027 (-9.0%) |
| f32 | 64 | 10^6 | 0.27017 | +0.0002 (0.09%) |

The residual pattern (large and positive at `ninsert=10^4`, negative at `10^5`, near zero at
`10^6`) is consistent across both precisions and both `nsys` values: the two-decade `ninsert`
span gives the largest point most of the leverage in the fit, so the line undershoots at the
smallest point and overshoots at the middle one even though the fit is dominated by, and most
accurate at, the largest point.

### PCIe/USB4 link and host-device bandwidth

The 4070 sits behind a USB4 (not PCIe-native) enclosure, so its effective host-device bandwidth
is set by the tunnel, not by the card's own PCIe generation. Measured, not inferred:

- **Topology** (`lspci -tv`): `...-01.2-[60-be]----00.0-[61-be]----00.0-[62-be]--+-00.0 NVIDIA
  RTX 4070 (+00.1 audio)`. Both upstream bridges (`60:00.0`, `61:00.0`) are "ASMedia Technology
  Inc. Device 2461" (an ASM2464-class USB4-to-PCIe bridge). `boltctl list` shows the enclosure as
  an "ASMedia 246x" USB4 peripheral, authorized, `rx speed: 40 Gb/s = 2 lanes * 20 Gb/s`, `tx
  speed` the same — the enclosure negotiates the 40 Gb/s USB4 mode, not 80 Gb/s. This is a
  different bridge chip (ASMedia) than the Intel JHL7440 Titan Ridge documented earlier for the
  3050's enclosure; per confirmation from the machine's owner both enclosures are USB4, so this
  is an ASMedia-vs-Intel USB4-controller difference, not a USB4-vs-Thunderbolt one. **No
  bandwidth measurement exists from the old (3050) enclosure**, so no enclosure-to-enclosure
  comparison of measured GB/s is possible here — only the current enclosure's numbers below are
  measured.
- **PCIe link state**: idle, `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
  reports `1, 4` (matches the card's idle power-saving state). `pcie.link.gen.max`/`width.max`
  report `4, 16` (the card's own native capability, not the tunnel's). Sampled every 0.1 s during
  a sustained transfer/kernel load (the bandwidth probe below, `utilization.gpu` up to 97%): the
  link rose to `2, 4` transiently and reached `4, 4` at peak load — **generation rises with load,
  width never exceeds 4 lanes** regardless of load, consistent with the USB4 tunnel provisioning
  a fixed 4-lane-equivalent path.
- **Host-device bandwidth** (`bench/gpu`'s `CUDA.jl`, `copyto!` + explicit `CUDA.synchronize()`
  inside the timed region — pinned-memory `copyto!` calls return before the transfer completes
  otherwise, which gave nonsense multi-TB/s readings before this was added; `BLAS.set_num_threads(1)`
  pinned first), at the sizes `widom`'s own per-chunk transfers actually use (chunk = 65,536
  insertions, ~40.5% rejected so ~39,007 survivors reach the phase-1 copies; computed from
  `src/widom.jl`'s `_widom` loop, not assumed):

  | transfer | size | pageable | pinned |
  |---|---|---|---|
  | H2D `dsys` (Int32×chunk) | 256 KiB | 1.43 GB/s | 1.47 GB/s |
  | H2D `drpos` (3×F64×chunk) | 1.5 MiB | 1.46 GB/s | 1.56 GB/s |
  | H2D `dquat` (4×F64×chunk) | 2 MiB | 1.47 GB/s | 3.76 GB/s |
  | H2D `dsurvivor` (Int32×nsurv) | 152.4 KiB | 1.38 GB/s | 1.43 GB/s |
  | D2H `dflags` (UInt8×chunk) | 64 KiB | 1.31 GB/s | 1.39 GB/s |
  | D2H `dΔU` (F64×nsurv) | 304.6 KiB | 1.43 GB/s | 1.59 GB/s |
  | 32 MiB reference (large-transfer asymptote) | 32 MiB | 3.77-3.78 GB/s | 3.80-3.82 GB/s |

  Pinned buys almost nothing here even at 32 MiB (3.8 GB/s either way): the ceiling is the USB4
  tunnel's own throughput (40 Gb/s ≈ 5 GB/s theoretical; ~3.8 GB/s measured is ~76% of that), not
  host-side staging overhead, which is what pinned memory normally buys back on a native PCIe
  slot.
- **Is transfer a bottleneck?** Summing `widom`'s actual per-chunk pageable transfers: Float64
  moves 4,088,136 B host-to-device + 377,488 B device-to-host per chunk (1.335 ms combined,
  measured); Float32 moves 2,253,128 B + 221,512 B (0.757 ms combined). Against the kernel-only
  phase0+phase1 time at `nsys=1` from the table above the same benchmark run also recorded
  (242.4 ms f64, 12.45 ms f32): **transfer is 0.55% of kernel time at f64 and 6.1% at f32 — not
  the bottleneck at either precision on this card.** It is a small piece of a larger per-chunk gap
  between kernel-only time and the real end-to-end `widom()` time: scaling the `ninsert=10^6`
  end-to-end median down to one chunk-equivalent gives a total non-kernel overhead of ~8.0 ms/chunk
  at f64 (3.2% of ~250 ms) and ~5.10 ms/chunk at f32 (29% of ~17.5 ms) — substantial at f32, as
  expected once the kernel itself gets this fast. But the measured transfer time above accounts
  for only ~0.76 ms of that 5.10 ms (~15%); the remaining ~4.3 ms/chunk is not PCIe/USB4 transfer
  and was not decomposed further here — the likely candidates are the host-side RNG pose
  generation, the phase-0 survivor compaction loop, and the Boltzmann-weight accumulation loop
  that `_widom` also runs per chunk, but this is unmeasured, not asserted.

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
| `kups_widom_timing_neuromancer4070_f64_20260926.json` | 1, 8 | 10^4, 10^5, 10^6 |
| `pureadsorb_widom_headtohead_neuromancer4070_f64_nsys{1,8}_ninsert{10000,100000,1000000}_20260926_{b26cb8a,7761a19}.json` | 1, 8 | 10^4, 10^5, 10^6 |
| `pureadsorb_widom_processcost_neuromancer4070_f64_20260926.json` | 1 | 10^4 (single point, whole-process cost only) |

The RTX 4070 files above use `nsys=8` in place of `nsys=4` (its own kUPS memory ceiling — see
below — is twice the 3050's) and are otherwise the same case, grid and method. One file,
`pureadsorb_widom_headtohead_neuromancer4070_f64_nsys8_ninsert1000000_20260926_7761a19.json`,
carries commit `7761a19` rather than `b26cb8a`: another agent committed an unrelated
`docs/specs/` design-note file to this branch while the (multi-hour) head-to-head run was in
progress, moving `HEAD` mid-run. That commit did not touch `src/` or `bench/`, so the code under
test did not change; the filename simply records `HEAD` accurately at the moment that one file
was written. `bench/run_headtohead.sh` was also fixed here to read the GPU name from
`nvidia-smi` and to honor a `PA_HOST` override instead of hardcoding `"NVIDIA GeForce RTX 3050"`
and `$(hostname)`, which had been silently mislabeling the `meta.gpu`/`meta.host` fields (fine
while only one GPU had ever run this script; wrong the moment a second one did).

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

### RTX 4070 (`neuromancer4070`, 12 GB): the ceiling rises to `nsys=8`

Testing powers of two by hand at `ninsert=10000` the same way as above: `nsys ∈ {1, 2, 4, 8}` all
complete their one requested cycle (confirmed by the `1/1` progress line and the harmless-only
exit-1 signature below, not just a nonzero exit code). `nsys=16` fails during state construction,
before any cycle is logged, with

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 10.08GiB.
```

— the identical allocation size the 3050 reported at the same `nsys=16` above, since the batched
state size depends only on the case and `nsys`, not the card; only the available headroom
differs. `nsys=32` and `64` were not tested, per instructions to stop at the first genuine
failure. **The 4070's larger memory doubles kUPS's usable batch from 4 to 8** — 10.08 GiB
exceeding 12 GB of *effectively usable* memory even though it is less than the card's raw 12 GB
is consistent with JAX/XLA's own allocator overhead and pool fragmentation (the log's own
suggestion, `TF_GPU_ALLOCATOR=cuda_malloc_async`, points at exactly this), but that mechanism was
not independently verified here.

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

### RTX 4070: marginal throughput

Same method, `nsys ∈ {1, 8}` (8 being kUPS's own ceiling on this card), verified real by the same
three checks as every kUPS point in this file: exit status, the harmless-only `OverflowError`
signature, and a `1/1` completed-cycle line in every one of the 36 logs this run wrote
(`bench/results/logs/kups_nsys{1,8}_ninsert{10000,100000,1000000}_{warmup,rep1..5}_20260926.log`,
not committed) — zero `RESOURCE_EXHAUSTED` occurrences across all 36:

| code | nsys | intercept (s) | marginal rate (insertions/s) | residuals (s, % of t_median) |
|---|---|---|---|---|
| kUPS (JAX) | 1 | 17.20 | 3,589 | 10^4: +0.49 (2.4%); 10^5: -0.54 (-1.2%); 10^6: +0.05 (0.02%) |
| kUPS (JAX) | 8 | 20.49 | 10,942 | 10^4: -0.04 (-0.2%); 10^5: +0.05 (0.2%); 10^6: -0.004 (0.00%) |
| PureAdsorb (CUDA f64, warm in-process) | 1 | 0.044 | 263,818 | 10^4: +0.044 (34.8%); 10^5: -0.048 (-12.9%); 10^6: +0.004 (0.11%) |
| PureAdsorb (CUDA f64, warm in-process) | 8 | 0.045 | 263,169 | 10^4: +0.042 (33.7%); 10^5: -0.047 (-12.3%); 10^6: +0.004 (0.11%) |

kUPS's residuals are tiny (≤2.4%) — its whole-process time is close to perfectly linear in
`ninsert`, as expected once its own ~17-20 s fixed cost dominates the small end and its per-
insertion rate dominates the large end evenly. PureAdsorb's larger relative residuals at
`ninsert=10^4` repeat the same non-linearity already seen in its own three-point grid above (a
two-decade span gives the largest point most of the fit's leverage); its intercept is two orders
of magnitude smaller than kUPS's, consistent with "warm in-process" against kUPS's whole-process
Python/JAX startup and compilation cost.

Ratio (PureAdsorb marginal / kUPS marginal, same precision — both float64): **73.5×** at nsys=1,
**24.1×** at kUPS's max batch (nsys=8; comparing against PureAdsorb's own nsys=8 fit above, not
its nsys=64 one, so the ratio is apples-to-apples in nsys as well as precision). Using
PureAdsorb's ordinary nsys=64 numbers from the "RTX 4070" section above against kUPS's nsys=8 (as
the original nsys=4-vs-64 table above does) gives the same ratio to within 0.2% (24.11× vs
24.05×), since PureAdsorb's own marginal rate barely depends on `nsys`.

### RTX 4070: fixed cost per process

Same method as the 3050 above:
`PA_BACKEND=cuda PA_PRECISION=f64 PA_GRID=1:10000 PA_REPS=1 PA_HOST=neuromancer4070 julia
--project=bench/gpu bench/widom_bench.jl`, 3 repetitions:
`pureadsorb_widom_processcost_neuromancer4070_f64_20260926.json` — 16.82 s, 16.18 s, 15.58 s
(median 16.18 s). This one-off wall-clock measurement reuses the same `nsys=1, ninsert=10000`
`PA_GRID` point the interleaved head-to-head run also writes to; the 5-sample head-to-head file
for that point was saved and restored around these 3 single-sample runs so neither measurement
overwrote the other's committed JSON.

| code | fixed cost per process (s) | what it includes |
|---|---|---|
| kUPS (JAX) | ≈17.2–20.5 (regression intercept) | Python/JAX startup, XLA compilation |
| PureAdsorb (CUDA) | ≈15.6–16.8 (whole-process wall time) | Julia startup, package load, kernel compile |

### Caveats

- Single card, USB4 eGPU enclosure (see the Machines section and "RTX 4070 eGPU link" above),
  neuromancer's CPU clock unpinned.
- Float64 on a GeForce card: double-precision throughput is throttled relative to a datacenter
  part (see the Precision/Machines notes above); this affects both codes equally since both run
  float64 here.
- kUPS forces float64 (`jax_enable_x64=True`) and was not run in float32 for this comparison;
  PureAdsorb's float32 numbers are `pureadsorb_widom_neuromancer_cuda_f32_20260919.json` (3050)
  and `pureadsorb_widom_neuromancer4070_cuda_f32_20260926_b26cb8a.json` (4070).
- `bench/run_headtohead.sh` sets the CPU governor to `performance` when writable; on this host
  it was not (`powersave` throughout, recorded in each kUPS JSON's `meta.cpu_governor`), for both
  the 3050 and 4070 runs.
- No host-device bandwidth or PCIe-link measurement exists from the 3050's enclosure, so the
  4070's numbers in "RTX 4070 eGPU link" above cannot be compared enclosure-to-enclosure, only
  reported on their own.

`plot_widom.jl` regenerates `widom_throughput.png` from the committed JSON only (no benchmark
runs); its kUPS series is plotted as the marginal rate above (`ninsert / (t - intercept)`), not
raw `ninsert/t`, and PureAdsorb's series is labeled "warm in-process".

## Guest-guest move throughput (Milestone B, task 3)

The throughput amendment to the Milestone B plan requires the reciprocal-versus-real-space
split to be measured before task 5 designs a move kernel, since the design's own estimate (the
reciprocal sum "expected to dominate" at 4587 k-vectors against ~360 real-space neighbors) is a
hypothesis, not a finding.

### Protocol

`bench/guest_bench.jl` builds RUBTAK 3x3x3 with 50 CO2 guests (`PA_NGUESTS`, default 50) and
`FrameworkBatch(...; fullk = true)`, generates `PA_NMOVES` (default 65536) independent move
proposals (a random guest, a Gaussian displacement — R3's translation shape), and times two
kernels with `Chairmarks.@be` (`evals = 1`, `seconds = 20`, `samples = 10`, wall time only):
`realspace_move_kernel!` (host-guest Lennard-Jones + Ewald real-space, plus guest-guest
Lennard-Jones + Ewald real-space against every other guest in the system) and
`recip_move_kernel!` (the running-structure-factor `ΔU_recip` formula, §3.3 of the design). Both
kernels process all `PA_NMOVES` proposals in a single launch; per-move cost is
`median(times_s) / PA_NMOVES`. `BLAS.set_num_threads(1)` runs before any timing.
`avg_neighbors` is the average number of host atoms within `max(cutoff, ewald_cutoff)` of a
proposal's old position, over the same proposals, computed directly (not from a benchmark) for
context on the real-space side.

Every sample is written to `bench/results/pureadsorb_guestmove_<host>_<backend>_<precision>_<date>_<commit>.json`.

### Results

RUBTAK 3x3x3 + 50 CO2: `nk = 4587`, `natoms = 3078` (host atoms in the system), measured
`avg_neighbors ≈ 367.8` — i.e. only about 12% of the real-space kernel's unconditional per-atom
loop actually falls inside the cutoff, since (matching Milestone A's own `insertion_energy`) it
is a plain linear scan over every host atom with no cell list at the energy-evaluation stage.

| host | backend | precision | real (ns/move) | recip (ns/move) | recip/real | file |
|---|---|---|---|---|---|---|
| neuromancer4070 | cuda | f64 | 11,149 | 6,806 | **0.61** | `pureadsorb_guestmove_neuromancer4070_cuda_f64_20260926_80ea256.json` |
| neuromancer4070 | cuda | f32 | 733 | 364 | **0.50** | `pureadsorb_guestmove_neuromancer4070_cuda_f32_20260926_80ea256.json` |
| neuromancer | cpu | f64 | 17,828 | 69,153 | 3.88 | `pureadsorb_guestmove_neuromancer_cpu_f64_20260926_80ea256.json` |

**The reciprocal sum does not dominate on the GPU — it is the cheaper half, by roughly 2×, at
both precisions.** This is the opposite of the design's own a priori estimate and is the more
interesting result the amendment asked to flag loudly if the expectation were wrong. The CPU row
shows the expected direction (reciprocal costlier, 3.9×), so the reversal is backend-specific,
not a sign error in either kernel — the same code, same proposals, same system, run on two
backends.

**Unverified hypothesis, not re-measured by profiling (task 3 does not optimize, only
measures):** the real-space kernel's cost is plausibly dominated by its unconditional
`O(natoms) = O(3078)` host scan (of which only ~368 iterations, ~12%, produce a nonzero
contribution) plus, for each of up to 49 other guests in the system, a full site-pair
recomputation (`guest_guest_move_delta` rebuilds every other guest's rotated sites from scratch,
twice, once against the mover's old pose and once against its new one) — both O(natoms) and
O(nguests) exceed the ~368-neighbor count the design's estimate was based on. The reciprocal
kernel's cost is a fixed `O(nk) = O(4587)` regardless of occupancy or host size. Task 5 should
treat "a real-space neighbor list for the move kernel" as at least as promising a lever as
anything on the reciprocal side, and should profile before choosing between them.

### Caveats

- Single RTX 4070 (see "RTX 4070 eGPU link" above); no ROCm/Metal numbers for this benchmark.
- The real-space kernel here is deliberately unoptimized (task 3's instruction): no neighbor
  list, no amortization of proposals sharing a launch, no exploitation of `ΔS`'s structure
  (translation/rotation-specific shortcuts) — see the plan's "Levers" list, none of which is
  applied yet.
- `avg_neighbors` uses the LJ/Ewald cutoff jointly (`max(cutoff, ewald_cutoff)`), not the two
  cutoffs separately, so it is an upper bound on either individual neighbor count.

## Rotation hoist and the three throughput-panel measurements (P3, P5)

Three measurements the throughput design panel required before task 5 designs a move kernel
(P5.1-P5.3), run after hoisting the guest-site rotation out of the reciprocal k-loop (P3,
`src/guest.jl`). The hoist itself is measured first, since P5.1 must run after it or it measures
unhoisted rotations.

### P3 — the rotation hoist was not already done by LLVM

`reciprocal_move_delta!`/`reciprocal_move_delta_energy` called `rotate(oldq/newq, guest.sites[s])`
inside the k-loop; `host_guest_realspace_energy` already hoisted the analogous computation.
Measured single-CPU-call `reciprocal_move_delta_energy` (RUBTAK 3x3x3 + 50 CO2, `Chairmarks.@be`,
`seconds=8`), before and after hoisting into a shared `guest_sites_at` call plus a new
`_reciprocal_move_delta_k` helper (same session, so before/after share the same compiled
environment):

| Precision | Before (median) | After (median) | Speedup |
|---|---|---|---|
| Float64 | 604.8 us | 265.5 us | **2.28x** |
| Float32 | 550.7 us | 213.9 us | **2.57x** |

LLVM had NOT already hoisted this — the plan's own "LLVM very likely does this already" was
wrong for this kernel, by a wide margin. Bit-identical output was verified against a literal,
un-hoisted reimplementation of the formula before trusting the benchmark. Not saved as a
`bench/results/*.json` file (a same-session A/B, not a reproducible standalone artifact); the
commit message for the hoist (`0eba49b`) records the same numbers.

### P5.1 — transcendental (sincos) fraction

`bench/sincos_fraction_bench.jl`, run with `PA_BACKEND=cuda julia --project=bench/gpu
bench/sincos_fraction_bench.jl` (GPU runs of any `bench/*.jl` script need `--project=bench/gpu`,
not `--project=bench`, per "Running on a GPU host" below). Same RUBTAK 3x3x3 + 50 CO2 setup and
65,536-shared-`Sk` proposal batch as the guest-move bench above, comparing `recip_move_kernel!`
(the real, post-hoist kernel) against a bench-local copy with `cis(x)` replaced by
`Complex(one(T) - x*x/2, x)` — deliberately wrong, same memory traffic and a comparable flop
count, isolating the cost of the transcendental evaluation itself.

| Precision | Real (ns/move) | Fake-cis (ns/move) | Sincos fraction | File |
|---|---|---|---|---|
| Float64 | 4,076 | 1,334 | **67.3%** | `pureadsorb_sincosfraction_neuromancer4070_cuda_f64_20260926_0eba49b.json` |
| Float32 | 157 | 39.6 | **74.8%** | `pureadsorb_sincosfraction_neuromancer4070_cuda_f32_20260926_0eba49b.json` |

Two-thirds to three-quarters of the (post-hoist) reciprocal kernel's own cost is the `cis`
evaluation itself, at both precisions. The post-hoist real numbers here (4,076 / 157 ns/move) are
themselves already far below the pre-hoist `recip` figures in the guest-move table above
(6,806 / 364 ns/move), matching P3's measured 2.28x/2.57x directly.

### P5.2 — visit versus in-cutoff pair arithmetic

`bench/visit_vs_pair_bench.jl`. `realspace_move_kernel!` (unmodified) run against three copies
of the same batch differing only in `cutoff`/`ewald_cutoff` (same alpha, k-table, sigma/epsilon,
atoms, cells): the real cutoffs, the Ewald cutoff collapsed to `1e-6` Å (LJ arithmetic only), and
both cutoffs collapsed to `1e-6` Å (visits only — no pair arithmetic can pass either branch).
Differencing isolates visit cost from LJ and Ewald in-cutoff arithmetic.

| Precision | Full (ns/move) | Ewald-cutoff-0 (ns/move) | Both-cutoffs-0 = visit (ns/move) | LJ arith | Ewald arith | Visit fraction | File |
|---|---|---|---|---|---|---|---|
| Float64 | 11,151 | 3,053 | 1,665 | 1,388 | 8,098 | **14.9%** | `pureadsorb_visitvspair_neuromancer4070_cuda_f64_20260926_0eba49b.json` |
| Float32 | 732 | 477 | 299 | 178 | 255 | **40.8%** | `pureadsorb_visitvspair_neuromancer4070_cuda_f32_20260926_0eba49b.json` |

At Float64 the visit (distance/min-image) cost is a small minority (15%) of the real-space
kernel's own cost; Ewald in-cutoff arithmetic dominates (73%). At Float32 the split is closer to
even (41% visit, 24% LJ, 35% Ewald) since the arithmetic itself is cheaper relative to the fixed
per-visit distance/min-image cost. Either way, removing visits alone (a neighbour list) caps the
achievable speedup on this kernel at `1/(1-0.149) = 1.18x` (Float64) or `1/(1-0.408) = 1.69x`
(Float32) even before accounting for the list's own construction and traversal overhead —
consistent with P4's own estimate (~1.18x) and with the ruling that a neighbour list is not
approved for this milestone.

### P5.3 — chain sweep: cost per move against chain count (the production shape)

`bench/chain_sweep_bench.jl`. Unlike every other measurement in this file, this builds `PA_NSYS`
**independent** systems (each RUBTAK 3x3x3 + 50 CO2, its own private slice of `Sk` — no sharing)
and launches exactly ONE move proposal per system per kernel call (`ndrange = nsys`). This is what
a real batch of chains looks like at the kernel-launch level, as opposed to the guest-move table's
single framework broadcasting 65,536 shared-`Sk` proposals into one launch (an L1 broadcast, not a
chain — see that table's own caveat). Batch construction (`FrameworkBatch` + `SystemState`) is
CPU-bound regardless of KernelAbstractions backend and dominates wall time at large chain counts
(`t_build` below); it is excluded from the reported per-move cost, which times only the kernel
launches.

RTX 4070, both precisions, real+recip per-move cost against chain count:

| nsys | t_build (s) | Float64 real (ns/move) | Float64 recip (ns/move) | Float64 total (ns/move) | Float32 total (ns/move) |
|---|---|---|---|---|---|
| 1 | 2.6 | 33,691,828 | 11,188,014 | 44,879,842 | 5,792,970 |
| 64 | 8-12 | 727,048 | 198,524 | 925,572 | 203,674 |
| 256 | 33-47 | 367,125 | 172,504 | 539,628 | 85,691 |
| 1,024 | 134-188 | 183,252 | 129,685 | 312,937 | 33,524 |
| 4,096 | 536-756 | 45,770 | 32,516 | 78,286 | 8,244 |

Files: `pureadsorb_chainsweep_neuromancer4070_cuda_f64_20260927_0eba49b.json`,
`pureadsorb_chainsweep_neuromancer4070_cuda_f32_20260927_0eba49b.json`.

**Occupancy has not saturated at 4,096 chains, at either precision.** Cost per move keeps
falling all the way to the largest chain count measured — no plateau appears in this range. At
`nsys=1` the per-move cost is dominated by CUDA kernel-launch overhead amortized over a single
work-item (tens of milliseconds); it falls by roughly three orders of magnitude by `nsys=4096`
and is still falling. Even at 4,096 chains, the measured cost (78.3 us/move Float64, 8.2 us/move
Float32) remains **4.4x (Float64) / 7.5x (Float32) above** the guest-move table's idealized
65,536-shared-`Sk` figure (18.0 us/move Float64, 1.10 us/move Float32) — confirming that
benchmark's own caveat that it measures full-occupancy kernel cost, not a realistic batch. **This
is the number that decides whether the rest of this milestone's kernel optimization work is
worth doing, and the answer is: not yet clear from this range** — a chain count large enough to
saturate this GPU's occupancy was not reached, so task 5's move-kernel design should either
target chain counts closer to (or past) 4,096, or accept that a production run at more modest
chain counts (tens to low hundreds, closer to what an actual adsorption simulation would run)
pays a per-move cost several times higher than any of the single-kernel numbers measured
elsewhere in this file.

CPU, Float64 only, reduced to `{1, 64, 256, 1024}` (`nsys=4096`'s ~750s build cost, measured on
GPU above, was not judged "cheap" to repeat on CPU as well):

| nsys | t_build (s) | real (ns/move) | recip (ns/move) | total (ns/move) |
|---|---|---|---|---|
| 1 | 2.5 | 112,186 | 254,875 | 367,061 |
| 64 | 11.7 | 119,152 | 261,711 | 380,863 |
| 256 | 46.3 | 120,625 | 261,523 | 382,148 |
| 1,024 | 185.8 | 120,775 | 261,168 | 381,943 |

File: `pureadsorb_chainsweep_neuromancer_cpu_f64_20260927_0eba49b.json`.

**CPU cost per move is flat across the whole sweep** (367-382 us/move, within noise), unlike the
GPU curve above: there is no kernel-launch overhead to amortize and no notion of occupancy on
CPU, so per-item cost is set by the work itself regardless of how many independent chains share a
launch. This is a useful cross-check, not a competing production path: CPU cost per move at any
chain count (~380 us) is already far above even the GPU's un-saturated `nsys=1` regime's
steady-state trend, let alone its `nsys=4096` figure (78.3 us).

## kUPS NVT (canonical Monte Carlo) throughput

`bench/run_kups_nvt.sh` runs kUPS's shipped `examples/nvt_co2_pressure_test.yaml` case as a
reference stopwatch: 50 CO2 in a 30 Å cubic box whose only host site is non-interacting
(`host/empty.cif`), `exchange_prob: 0`. Two modes, `timing` and `nscale`. One cycle repeats the
propagator `max(particle_count, min_cycle_length)` times, so with 50 guests and
`min_cycle_length: 1` a cycle is 50 move attempts per system; total attempts are
`nsys × num_cycles × 50`, confirmed against the HDF5 output.

RTX 4070, Float64, kUPS commit `e183c9a`:

| File | Result |
|---|---|
| `kups_nvt_timing_neuromancer4070_f64_20260927.json` | 2,580 moves/s at one system, intercept 24.3 s, fit residuals under 1% |
| `kups_nvt_nscale_neuromancer4070_f64_20260927.json` | cost per move for 1–32 systems; 64 systems fails with `RESOURCE_EXHAUSTED` at 10.49 GiB |

Batching does not help kUPS. Once the 24.3 s startup is removed, aggregate throughput at 32
systems is about 1,880 moves/s — lower than at a single system — so **2,580 moves/s is their peak
on this card**, and 32 systems is their ceiling for this case.

The runs do real work: acceptance rates are 54.6% translation, 70.7% rotation and 38.2%
reinsertion, and per-system acceptance differs at fixed seed, confirming independent chains.

Caveat: our own GPU tests ran during the `nscale` sweep, and repeat spread there is 6–20%. The
single-system fit was clean and is unaffected.

## `mc_step!` groupsize sweep (Milestone B task 7, cleanup 1)

`mc_step_bench.jl`'s own default (`groupsize = 256`) was chosen alongside a single at-one-chain
Float64 measurement (180-230 us/move at `groupsize = 64`, `be3f19e`'s commit message) without
ever comparing it against the alternatives. `bench/groupsize_sweep_bench.jl` sweeps
`groupsize ∈ {32, 64, 128, 256}` at `nsys ∈ {1, 64, 256}`, both precisions, RTX 4070, same
RUBTAK 3x3x3 + 50 CO2 case and warm-up discipline (wall-clock ramp) as `mc_step_bench.jl`:

| nsys | precision | 32 (us/move) | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 1 | f64 | 182.2 | 212.9 | 289.2 | 356.7 |
| 64 | f64 | 18.41 | 18.43 | 17.84 | 17.37 |
| 256 | f64 | 16.72 | 15.50 | 12.76 | 12.55 |
| 1 | f32 | 314.1 | 164.0 | 123.9 | 113.2 |
| 64 | f32 | 6.27 | 3.48 | 2.83 | 3.00 |
| 256 | f32 | 2.13 | 1.96 | 1.77 | 2.14 |

Files: `pureadsorb_groupsizesweep_neuromancer4070_cuda_f64_20260927_b6cc175.json`,
`pureadsorb_groupsizesweep_neuromancer4070_cuda_f32_20260927_b6cc175.json`.

No single groupsize wins everywhere: Float64 at `nsys=1` favors 32, every other Float64 cell and
every Float32 cell but `nsys=1` favors 128 or 256, and Float32 at `nsys=1` favors 256. Picking by
worst-case ratio to the best groupsize measured in each cell (minimax, since a default is chosen
without knowing in advance what `nsys` a caller will run at):

| groupsize | worst ratio to best | which cell |
|---|---|---|
| 32 | 2.77x | f32, nsys=1 |
| **64** | **1.45x** | f32, nsys=1 |
| 128 | 1.59x | f64, nsys=1 |
| 256 | 1.96x | f64, nsys=1 |

**`groupsize = 64` is the new default** (`PureAdsorb.DEFAULT_GROUPSIZE`, `src/moves.jl`): it has
the best worst-case ratio of the four (never more than 45% above the best measured groupsize in
any swept cell), it directly fixes the originally-flagged regression (256 costs 357 us/move at
`nsys=1` Float64, against 213 us/move at 64 — both comfortably under kUPS's 388 us/move
reference), and it stays within 6-23% of the best groupsize at every other measured cell. 128
edges it out on a plain sum of the six cells' ratios (6.73 vs 7.25), but by a margin smaller than
the run-to-run noise already visible between this sweep and `be3f19e`'s own at-one-chain figure
(213 us/move here against 180-230 us/move there, same nominal case) — not a large enough gap to
prefer the less robust choice. The `nsys=1` datapoint at `groupsize=64`, Float64 (213 us/move)
is this project's own record of the "187 us/move" figure `be3f19e`'s commit message cites but
never saved as JSON (the file at that name was overwritten by a later run before archival); the
number differs from 187 by ordinary run-to-run GPU measurement noise, not a regression, and is
now committed as part of the sweep JSON above rather than as a separate near-duplicate file.

## Milestone B validation ladder (task 9)

Run after fixing the energy audit's tolerance (`f6a99ca`). All PureAdsorb runs use the CPU
backend (`neuromancer`), 298.15 K, real-space/Ewald cutoff 12 A, precision 1e-6, frozen step
sizes (0.3 A translation, 0.3 rotation, R4), seed 42.

**B0 — N=0 reproduces Milestone A exactly.** `run_nvt!` at `N=0` agrees with `widom_singlephase`
to `rtol=1e-10` in μ_ex/K_H/q_st (not bit-for-bit: cycle-based blocking groups the same
per-insertion Boltzmann weights differently from Milestone A's insertion-based blocking).
`pureadsorb_nvt_b0_vs_milestone_a_neuromancer_20260927_f6a99ca.json`.

**B1 — pure CO2 fluid, kUPS's own `examples/nvt_co2_pressure_test.yaml`** (50 CO2, 30 A cubic
box, non-interacting host, `exchange_prob: 0`, 2000 warmup + 10000 production cycles, cycle
length 50 — moves match exactly, `nsys*ncycles*50`):

| | mean energy (eV) | SEM |
|---|---|---|
| PureAdsorb | -0.9632716 | 0.0101137 |
| kUPS | -0.9639866 | 0.0080932 |

Diff 0.05 combined SE. Acceptance: PureAdsorb 82.8/70.5/37.4% (translation/rotation/reinsertion,
frozen steps) vs kUPS 54.6/70.7/38.2% (adapting steps, R4) — translation differs because our step
is frozen and, at 0.3 A, happens to sit well above kUPS's adapted ~50%-target value; rotation and
reinsertion (which kUPS never actually tunes, Task 1 finding #2) land close by construction.
`pureadsorb_nvt_b1_vs_kups_neuromancer_f64_20260927_f6a99ca.json` (PureAdsorb on CPU, kUPS on the
RTX 4070).

**Float32 vs Float64 (B1).** Same case, same seed, Float32 vs Float64:

| | mean energy (eV) | SEM |
|---|---|---|
| Float64 | -0.9632716 | 0.0101137 |
| Float32 | -0.9726815 | 0.0095101 |

Diff 0.68 combined SE — no statistically significant Float32 bias detected in mean energy.
`pureadsorb_nvt_b1_float32_vs_float64_neuromancer_20260927_f6a99ca.json`.

**B2 — RUBTAK 3x3x3 + 50 CO2, `exchange_prob: 0`**, written in kUPS's own config schema (2000
warmup + 5000 production cycles). kUPS reports the FULL system energy (it computes
`U_host-host`; PureAdsorb's `total_energy` never does, by design), so its number is compared
against `energy(N=50) - energy(N=0)` from a separate kUPS run of the same host:

| | mean guest-dependent energy (eV) | SEM |
|---|---|---|
| PureAdsorb | -12.154340 | 0.025394 |
| kUPS (full − host-only) | -12.152647 | 0.024249 |

Diff 0.05 combined SE. Acceptance: PureAdsorb 55.3/35.2/1.15% vs kUPS 49/48/0% — reinsertion is
near zero in both codes (a dense host makes a fully random reinsertion almost always overlap).
`pureadsorb_nvt_b2_vs_kups_neuromancer_f64_20260927_f6a99ca.json`.

**Widom-along-the-chain vs an independent oracle (no kUPS counterpart, R5).** No kUPS example
runs N-guest NVT and Widom together, so this validates `widom_chain_kernel!`'s test particle
seeing every OTHER existing guest (not only the host) against a literal, non-incremental Ewald
sum (`ewald_energy`) computed independently before and after appending the test guest. A single
fixed pose does not match exactly — `constant_offset`'s orientation-AVERAGED reciprocal self term
carries a real per-pose error (bounded by `self_term_halfrange`) whose mean is zero by
construction — so this compares the mean Boltzmann weight over many random poses instead, the
same quantity Widom's own μ_ex accumulates:

| Case | poses | diff / combined SE |
|---|---|---|
| Empty box + 10 guests | 2000 | 0.00012 |
| RUBTAK + 10 guests | 150 | 0.0000065 |

`pureadsorb_widom_chain_vs_oracle_neuromancer_f64_20260927_f6a99ca.json`. Test items:
`test/nvt_tests.jl`'s two `"...matches an independent oracle with other guests present..."` items
(the RUBTAK one tagged `:slow`, ~1 s/pose from a full non-incremental Ewald sum over 3078+ atoms).

## Optimization wave, item 1 — framework dedup (ranked-plan item 2, `2026-09-26-milestone-b-nvt.md`)

`FrameworkBatch` now stores `Shost`/`kmin`/`bs`/atoms/cells/etc. once per DISTINCT framework
value (`framework_of` indirection), not once per system. `fill(sc, nsys)` (identical framework
across every chain — the production shape for a chain sweep) before/after commit `e38352a` ->
`efa65a5`:

| nsys | t_build before | t_build after | bytes before | bytes after |
|---|---|---|---|---|
| 1 | 0.18 s | 0.18 s | 357 KB | 357 KB |
| 64 | 10.6 s | 0.17 s | 36.1 MiB | 357 KB |
| 256 | 42.4 s | 0.17 s | 98.3 MiB | 358 KB |
| 1024 | 169.6 s | 0.19 s | 462.7 MiB | 361 KB |
| 4096 | 684.9 s | 0.26 s | 1994.9 MiB | 374 KB |

Bytes are `Base.summarysize` of the host-resident `FrameworkBatch` object, a proxy for what
`adapt` uploads to the device. `mc_step!` itself is byte-for-byte unchanged; the L2-residency
effect the design doc flagged as real-but-unquantified is exactly the gap below, since the ONLY
thing that changed is the device memory footprint the same kernel reads:

| nsys | ns/move before | ns/move after | speedup |
|---|---|---|---|
| 1024 | 1465.0 | 994.3 | 1.47x |
| 4096 | 1288.9 | 832.4 | 1.55x |

Float32, RTX 4070, warmed on wall-clock time (0.5 s of untimed calls after one compile call),
median of 200 timed calls. `pureadsorb_frameworkdedup_neuromancer_20260927_efa65a5.json`.

## Optimization wave, item 2 — the term split, re-measured on `mc_step!`

Every share published before this point (`pureadsorb_sincosfraction_*`, `pureadsorb_visitvspair_*`)
was measured on the one-thread-per-move kernel `mc_step!` replaced and no longer describes this
code. Re-measured on the real 3-kernel `mc_step!` pipeline (RUBTAK 3x3x3 + 50 CO2/chain, RTX
4070), by the same disabling-and-differencing method: a same-cost-but-numerically-wrong `cis`
isolates its own transcendental cost (`sincos_fraction`); a kernel that skips the reciprocal
k-loop entirely isolates the reciprocal term's TOTAL share (`reciprocal_fraction`, which
`sincos_fraction` alone understates); the real-space cutoff ablation, run through the
no-reciprocal kernel so it cannot conflate real space with reciprocal or launch/decide/apply
overhead, splits the remainder into LJ arithmetic, Ewald arithmetic, and a residual
"visit-and-overhead" share. All four fractions telescope to exactly 1 by construction.

| precision | nsys | sincos | reciprocal | LJ arith | Ewald arith | visit+overhead |
|---|---|---|---|---|---|---|
| Float64 | 1 | 0.049 | -0.070 | 0.029 | 0.185 | 0.856 |
| Float64 | 1024 | 0.294 | 0.408 | -0.176 | 0.199 | 0.569 |
| Float32 | 1 | 0.202 | 0.317 | 0.073 | 0.082 | 0.528 |
| Float32 | 1024 | 0.103 | 0.268 | -0.116 | 0.044 | 0.804 |

At `nsys = 1024`, `sincos_fraction` is a real, moderate share (10-29%) but a strict SUBSET of
`reciprocal_fraction` (27-41%) — roughly 40-70% of the reciprocal term's own cost is the `cis`
call itself, the rest is `kprefactor`/`Sk` loads and the complex arithmetic around it.
`visit_and_overhead_fraction` dominates the real-space share at every measured point (53-86%),
confirming P4's qualitative finding (most real-space cost is the atom-loop scan, not the pair
arithmetic once inside cutoff) survives on the new kernel even though the numbers changed.
`LJ arith` comes out **negative** at `nsys = 1024` in both precisions, a nonphysical result: this
GPU's clock is not pinned for benchmarking (only galen/wintermost are gate-authoritative) and the
`ewald0`/`both0` calls run sequentially late in the script, so a rising boost-clock state over the
run is a plausible confound that this measurement cannot rule out. `nsys = 1`'s numbers are
dominated by kernel-launch latency (per-move times of 150-250 μs against a throughput floor near
0.15 μs) and should be read as noisy. Deciding item 3 from this: `reciprocal_fraction`'s ceiling
(27-41% at 1024 chains) exceeds `sincos_fraction`'s own ceiling (10-29%), so item 4 (split the
structure factor, cutting total k-work 2.7x) has more to gain than item 5 (factorized phase
tables, which only ever touches the `cis` call) — consistent with the ranked plan's own ordering.
`pureadsorb_mcstepdecompose_neuromancer_20260927_efa65a5.json`.

## μVT exchange moves: workgroup fan-out (Milestone C job 1)

`mc_insert_kernel!`/`mc_delete_kernel!` were originally one work-item per chain, so a single GPU
thread summed the whole reciprocal-space k-vector loop serially — the same shape `mc_step!`'s own
kernels had before task 5, and the same fix: `mc_insert!`/`mc_delete!` now split into an
evaluate/decide/apply three-kernel pipeline that fans a chain's host-atom, guest-guest and
k-vector loops across `nblocks_per_chain` workgroups (`src/moves.jl`'s own comment on the μVT
exchange moves has the full mechanical description). `bench/gpu/exchange_workgroup_bench.jl`
measures `mc_insert!`/`mc_delete!`'s per-call cost at nsys ∈ {1, 64, 256}, RUBTAK 3×3×3 + CO2, 10
initial guests/chain, capacity 40, fugacity 2e4 Pa, `nblocks_per_chain` at its own
`default_nblocks_per_chain(F, nsys)` default — the same shape `mc_step_bench.jl` uses for
`mc_step!`. Occupancy is reset to its starting `nguests` after every call (warm-up and timed
samples alike): a workgroup-fanned exchange move is fast enough that thousands of calls fit in the
0.5 s wall-clock warm-up window, and this system's real (favorable) CO2-in-RUBTAK adsorption
equilibrates well above `capacity=40` at 2e4 Pa — an unconstrained warm-up walks occupancy into
`mc_insert!`'s own capacity-hit failure well before the timed samples run, a real failure mode hit
while building this benchmark.

RTX 4070, `commit f4a0503`, "before" from `git stash` of this work, "after" from the landed
workgroup fan-out:

| precision | move | nsys | before (us/call) | after (us/call) | speedup |
|---|---|---|---|---|---|
| Float64 | insert | 1 | 15,504.0 | 188.6 | 82.2x |
| Float64 | insert | 64 | 47,675.2 | 821.7 | 58.0x |
| Float64 | insert | 256 | 108,923.6 | 2,522.5 | 43.2x |
| Float64 | delete | 1 | 10,444.1 | 134.2 | 77.8x |
| Float64 | delete | 64 | 23,469.6 | 485.1 | 48.4x |
| Float64 | delete | 256 | 61,629.3 | 1,624.6 | 37.9x |
| Float32 | insert | 1 | 3,290.1 | 159.1 | 20.7x |
| Float32 | insert | 64 | 7,653.1 | 204.8 | 37.4x |
| Float32 | insert | 256 | 8,794.5 | 312.8 | 28.1x |
| Float32 | delete | 1 | 1,867.4 | 105.4 | 17.7x |
| Float32 | delete | 64 | 2,027.5 | 124.0 | 16.4x |
| Float32 | delete | 256 | 3,086.9 | 171.8 | 18.0x |

At nsys=1 both precisions land at or under `mc_step!`'s own ~187-230 us/move (Float64 target).
Cost rises with nsys the same way `mc_step!`'s does — more chains' work per launch, not fan-out
degrading — and `nblocks_per_chain` itself shrinks with nsys
(`default_nblocks_per_chain(F, nsys)`: 256/4/1 at nsys=1/64/256 for Float64, 1 throughout for
Float32), which is why the *rate* of increase from nsys=1 to nsys=256 (13-19x) is smaller than the
256x growth in chain count.

Files: `pureadsorb_exchange_workgroup_neuromancer_cuda_{f64,f32}_before_20260927_f4a0503.json`,
`pureadsorb_exchange_workgroup_neuromancer_cuda_{f64,f32}_after_20260927_f4a0503.json`.

## Units rollout: the Unitful-fugacity boundary's overhead on `mc_exchange!`

Now that the workgroup fan-out above brings `mc_exchange!` down to roughly 150-200 us/call at
nsys=1, a fixed per-call cost invisible at the pre-fan-out 18 ms scale could plausibly be a
meaningful fraction of the call. `bench/gpu/units_overhead_bench.jl` isolates the Unitful
boundary's own cost by measuring `mc_exchange!` with a bare `Vector{F}` fugacity against the same
call with a `Vector{<:Unitful.Pressure}` fugacity (`src/units.jl`'s wrapper strips it to a bare
`Vector{F}` and forwards), RUBTAK 3×3×3 + CO2, nsys=1, both precisions, RTX 4070. Each
(precision, bare-or-unitful) combination runs in its own process, warmed on wall-clock time; the
file's own header explains two measurement traps found and fixed while building it (closing over
top-level globals, and `mc_exchange!`'s own insert/delete coin flip needing BOTH kernel variants
pre-compiled before timing either one).

| precision | bare (us/call) | Unitful (us/call) | difference (us) |
|---|---|---|---|
| Float64 | 198.9 | 151.4 | -47.5 |
| Float32 | 165.8 | 154.2 | -11.6 |

Both differences are NEGATIVE — the "Unitful" call measured faster than "bare" — which is itself
the tell that this is run-to-run noise, not a real cost: four repeated bare-only Float64
measurements (same code, same call, nothing changed between runs) read 147.4, 141.3, 163.0 and
198.9 us/call, a 57.6 us spread on their own, several times the -47.5/-11.6 us "difference" above.
The Unitful boundary's own extra work per call — `ustrip.(u"Pa", fugacity)` plus one small
`Vector{F}` allocation — is not measurably distinguishable from zero against this noise floor.

Files: `pureadsorb_units_overhead_neuromancer4070_cuda_{f64,f32}_{bare,unitful}_0281172.json`.

## A real isotherm: CO2 in RUBTAK 3x3x3 (Milestone C task 6)

`bench/gpu/isotherm_bench.jl` runs `run_isotherm!` (`src/isotherm.jl`) at 298.15 K over 50
log-spaced pressure points from 100 Pa to 1e5 Pa, 4 replicas each (nsys = 200), one batch, RTX
4070, `commit 873c9ed`.

**Build time**: `FrameworkBatch`+`SystemState` for the whole 200-system batch (`ncounts = 0`
throughout, so no per-guest placement work) took 0.279 s, against 2.96 s for a single system
built cold in the same process (compilation-dominated) — the isotherm-scale build is dominated
by paying the framework's own setup once, exactly as framework deduplication predicts, not by
`nsys`. `nframeworks(batch) == 1` confirms the dedup fired.

**Run time**: 151.1 s for the full 700-cycle (200 warmup + 500 production) GCMC run over all 200
systems, no capacity failures (`mc_insert!` never threw): the batch-wide cycle length, set by the
highest-occupancy system at any moment (up to ~136 guests near the top of the pressure range),
means every system in the batch attempts that many moves per cycle regardless of its own loading
— the "wastes work on low-pressure chains" cost the design review's own "Throughput" section
flags, paid here as wall-clock rather than a failure.

**Shape**: loading rises monotonically at every one of the 50 points, from 0.40 guests at 100 Pa
to 101.7 guests at 1e5 Pa. `loading/pressure` (a local Henry's-law slope) is flat at
0.0037-0.0040 guests/Pa over the bottom 6 points (100-202 Pa) — Henry-linear, as expected well
below saturation — and falls by a factor of ~3.7 to 0.0010-0.0016 guests/Pa over the top 6 points
(4.9e4-1e5 Pa), a clear, physically sensible Type-I saturation curvature. The verdict: **physically
sensible** — monotonic, Henry-linear at the low end, saturating (sub-linear) at the high end. The
handful of points between 3.7e4 and 8.7e4 Pa (67.7, 74.9, 79.4, 86.4, 91.8, 93.4, 93.1) show a
near-plateau with a one-point dip inside statistical error (`loading_err` there is 2.7-4.3 guests)
rather than a real non-monotonicity — reported as seen, not smoothed.

**Capacity**: one value (200) for the whole batch, sized from a short pilot run at the highest
pressure rather than any a priori estimate (`run_isotherm!`'s own docstring explains why a
per-pressure capacity is not attempted). Worst-case occupancy across all 50 points' replicas
tops out at 136/200 (68%) at the highest pressure; at the lowest pressures max occupancy is
4-6 out of 200 (2-3%) — confirming the design's own prediction that a single batch-wide capacity
is wasteful at the low end (reserved memory only, since every energy loop is bounded by live
occupancy, not capacity) without ever saturating at the high end. The capacity diagnostic did
**not** fire.

**A pre-existing numerical finding, found while building this isotherm — now fixed** (below):
`audit_energy!`'s tolerance was occasionally too tight once `nmoves` reached the low thousands on
a CUDA-driven chain. The root cause was not a rounding-bound gap: `run_nvt!`/`run_gcmc!`'s
`run_audit!` never copied `state.sk_abs_accum`/`energy_abs_accum` from the device before sizing the
tolerance, so it was computed from an all-zero host copy instead of the real, device-accumulated
running sum, on every backend where `dst` is not the same allocation as `state` (CUDA, ROCm — not
`CPU()`). `isotherm_bench.jl` still sets `n_audit` past this run's end, since a mid-run audit
under any tolerance is orthogonal to what this benchmark measures.

File: `pureadsorb_isotherm_co2_rubtak_neuromancer_cuda_f64_20260927_873c9ed.json`.

## Audit tolerance false-positive rate (CUDA), before/after the sync fix

Reproduced independent of Milestone C's own work: a plain `run_nvt!` chain (no exchange moves,
unmodified NVT kernels) on CUDA trips `audit_energy!` because `run_audit!` sizes the tolerance from
an unsynced (always-zero) `energy_abs_accum`, floored at `nmoves*eps(F)*1` regardless of the real
accumulated `Σ|ΔU|`. Fixed by (1) syncing `sk_abs_accum`/`energy_abs_accum` device↔host in both
`run_nvt!`'s and `run_gcmc!`'s `run_audit!`, exactly as `Sk`/`energy` already are, and (2) scaling
the energy check's recompute-side magnitude by `total_energy`'s own term count
(`occ + occ*(occ-1)/2`) times `batch.bs[fw]` (the existing `hardcore_bound` per-guest term-magnitude
bound), matching the structure-factor check's existing `nterms_rebuild` scaling.

Measured false-positive rate over a sweep of fresh seeds (RUBTAK-3x3x3 + CO2, 298.15 K, `nsys=32`
replica chains per seed, `mc_step!` driven directly on `CUDABackend()`), at 1,000/5,000/20,000
accepted moves per system:

| | before (buggy sync) | after (both fixes) |
|---|---|---|
| Float64 (7 seeds, 672 trials) | 13.99% overall (37.1% at 1,000 moves), max ratio 11.6× tolerance | 0%, max ratio 0.0025× |
| Float32 (8 seeds, 768 trials) | 16.15% overall (39.1% at 1,000 moves), max ratio 14.8× tolerance | 0%, max ratio 0.0019× |

The false positives concentrate at low-to-moderate `nmoves` and vanish by 20,000 even without any
fix, because the buggy tolerance's other term (`nmoves*eps(F)*1`) eventually outgrows the fixed
(non-accumulating) discrepancy on its own — the discrepancy itself stays flat at order
1e-13–1e-12 eV across this whole range once `energy_abs_accum` is correctly synced, rather than
growing with move count; a direct reproduction (RUBTAK-3x3x3+CO2, state seed 77, movetype seed 5)
showed the true `energy_abs_accum` reaching 5872.9 eV over 220 accepted moves from a handful of
large early-equilibration moves, not from many small per-move terms. Every existing corruption
test (wrong sign, wrong phase, wrong guest, corrupted `ΔU` both precisions, stale host-energy
cache, occupancy-exceeds-capacity) still throws: full suite 196/196, cold `Pkg.test()` under
`--check-bounds=yes` 4,131,961 assertions, both clean.

File: `pureadsorb_audit_tolerance_falsepositive_neuromancer4070_cuda_20260927_7f1032f.json`.

## R2 merged Henry's-law/detailed-balance test (tasks 7+8)

`P(N+1)/P(N) = (fV/((N+1)kT)) * ⟨exp(-ΔU_ins/kT)⟩_N`, LHS from a GCMC chain's own occupancy
histogram and RHS from Widom insertions into the chain's live configuration at each occupancy `N`
(`widom_chain_kernel!`, Milestone B), both under RUBTAK-3x3x3 + CO2's real potential at 298.15 K,
500 Pa, 64 replica chains. Every one of 8 loadings (`N=0..7`) agrees within `z<=1.78` combined
standard errors; `N=0`'s own Widom average, expressed as a Henry coefficient, agrees with Milestone
A's independent `widom()` route (2,000,000 insertions, its own RNG stream, run on the pristine
framework) at `z=2.4` — consistent with `K_H`'s own heavy-tailed sampling noise (re-running
`widom()` at the chain's own N=0 sample count, 106,104, gives `K_H` in `[6.13e8, 6.38e8]` over 4
seeds, the same scale of spread as the chain-derived estimate).

500 Pa was chosen because `peng_robinson_fugacity` gives `phi~0.99997` there (this run does not
also exercise the equation of state — R3's own test does that, at 5e6 Pa) and because a short
isotherm check (200/500 Pa) confirms `loading/pressure` matches Milestone A's Henry slope within
5 combined SEM at both points, well inside the linear regime this file's own 50-point isotherm
(above) shows eventually breaks down many orders of magnitude higher in pressure.

Discriminating power: running the same chain with insertions/deletions accepted against 1.5x the
true fugacity (the same position the combinatorial prefactors `log_insertion_prefactor`/
`log_deletion_prefactor` take `f` in) while still comparing against the TRUE fugacity's RHS fails
every well-sampled loading (`N=0..5`) at `z=3.5-7.4`, recovering the injected 1.5x factor in the
median ratio to within 0.02 — the sparsest loadings (`N=6..8`, under 4,000 occupancy visits) do not
discriminate reliably, the sparse-loading fallback the design anticipates.

File: `pureadsorb_henry_r2_detailedbalance_neuromancer4070_cuda_20260927_72d9547.json`.

## Milestone C validation against kUPS GCMC and RASPA (task 9)

### kUPS GCMC cross-check

`bench/run_kups_gcmc.sh main` runs kUPS's shipped `examples/mcmc_rigid.yaml` UNCHANGED — its own
comment labels the pressure "10_000  # Pa (10 bar)", which is wrong (1e4 Pa is 0.1 bar), and
`translation_prob`/`rotation_prob`/`reinsertion_prob` are all 0 there with `exchange_prob` left
unset (defaulting to `RunConfig`'s 0.5, the only nonzero weight, so every cycle is an exchange
attempt). Task 9 matches both quirks literally rather than correcting them, since correcting
either would no longer be the same case. `bench/gcmc_vs_kups.jl` runs the PureAdsorb side at the
matched parameters (RUBTAK 3x3x3, CO2, 298.15 K, real-space/Ewald cutoff 12 A, Ewald precision
1e-6, `num_warmup_cycles=1000`, `num_cycles=10000`, `min_cycle_length=20`, seed 42, capacity 300,
CPU backend) and writes the combined comparison:

| Quantity | PureAdsorb | kUPS | Combined SE | Deviation |
|---|---|---|---|---|
| Loading (guests) | 31.171 ± 0.577 | 31.216 ± 0.879 | 1.052 | 0.043 σ |
| Energy (eV) | −7.4496 ± 0.1509 | −7.4346 ± 0.2410 | 0.2844 | 0.053 σ |
| q_st (eV) | 0.26291 ± 0.00208 | 0.26036 ± 0.00298 | 0.00363 | 0.703 σ |

`pureadsorb_gcmc_vs_kups_neuromancer4070_f64_20260927_db76305.json` (PureAdsorb on CPU, kUPS on
the RTX 4070); the raw kUPS output (`bench/run_kups_gcmc.sh main`) is
`kups_gcmc_main_neuromancer4070_f64_20260927.json`.

Two conventions this comparison must not get wrong, both recorded in the JSON's own `meta` notes:

- **Energy baseline.** kUPS reports the FULL system energy including `U_host-host`, a constant
  PureAdsorb's `total_energy` never computes since it cancels in every difference (exactly
  Milestone B's B2 issue). `energy_host_only_eV` (−9261.9521 eV, SEM 0 to machine precision) comes
  from a second kUPS run of the same host: `init_adsorbates: [0]`, `exchange_prob: 0`, 100 cycles.
  Subtracting it from kUPS's full-system mean (−9269.3867 eV) gives the guest-dependent energy
  PureAdsorb's own number is compared against.
- **`q_st` sign.** kUPS's GCMC analyzer (`application/mcmc/analysis.py:122-126`) computes
  `cov(U,N)/var(N) - kT`; its own Widom analyzer (`analysis.py:326-332`) computes the opposite,
  `kT - <dU*W>/<W>`. PureAdsorb's `fluctuation_qst` follows the Widom convention, so kUPS's raw
  GCMC output (−0.26036 eV) is negated before comparison, not used as reported.

At this pressure `peng_robinson_fugacity` gives `phi=0.99949`: this comparison exercises the
combinatorial insertion/deletion machinery and the energy/q_st pipeline under real interactions,
**not** the equation-of-state path beyond a 0.05% correction (`ideal_gas_tests.jl`'s CO2-at-5-MPa
case, a ~20% effect, covers that separately). kUPS's own `n_blocks` is chosen automatically by
`optimal_block_average` (4 here); PureAdsorb's is fixed at 10 (`run_gcmc!`'s own default) — the
same "different block-count rules, neither claims tighter agreement than the looser" caveat as
every other cross-code comparison in this file.

All three quantities agree within 1 combined standard error, well inside the 3σ bar this
project's other cross-code checks (`reference_tests.jl`) use.

### kUPS GCMC memory ceiling

`bench/run_kups_gcmc.sh nscale 200 64` batches N independent copies of the same `mcmc_rigid.yaml`
case (200 cycles, no warmup) doubling `nsys` until failure:

| nsys | wall time (s) |
|---|---|
| 1 | 25.4 |
| 2 | 30.5 |
| 4 | 31.4 |
| 8 | 38.0 |
| 16 | **RESOURCE_EXHAUSTED**, 13.18 GiB requested |

`kups_gcmc_nscale_neuromancer4070_f64_20260927.json`. **kUPS's GCMC ceiling on this 12 GiB card is
8 systems** — worse than Milestone B's 32-system NVT ceiling for a comparable RUBTAK+CO2 case, as
the design doc anticipated (a GCMC batch additionally reserves `max_num_adsorbates` buffer slots
per system, auto-estimated at up to `1e4` times the ideal-gas reservoir occupancy). This bounds
what a like-for-like batched throughput comparison against kUPS's GCMC could even attempt; task 9
did not attempt one (task 10 covers PureAdsorb's own throughput separately).

### RASPA IRMOF-1 methane isotherm: blocked, not attempted

PureAdsorb's `read_cif` accepts only space group P1 with a populated `_atom_site_charge` column
(`src/structure.jl`). Three independent, publicly available IRMOF-1 structure files were checked
directly (fetched unmodified, not edited):

| Source | Space group | Charge column |
|---|---|---|
| `numat/RASPA2` (`structures/mofs/cif/IRMOF-1.cif`, D. Dubbeldam, RASPA's own canonical file) | `F m -3 m` (225) | absent |
| `numat/EQeq` (`IRMOF-1.cif`) | P1 | absent |
| `SimonEnsemble/PorousMaterials.jl` (`viz/IRMOF-1.cif`) | P1 | absent |

No file combining both requirements was found. IRMOF-1's usual force fields (UFF/DREIDING LJ,
literature or DDEC partial charges) assign charge **per atom type/role** (Zn, the central oxo
O, carboxylate O, carboxylate C, substituted/unsubstituted ring C, H) rather than storing a value
per atom in the CIF; producing a P1-with-charges file would mean either symmetry-expanding the
non-P1 RASPA2 file or matching a separate literature charge table onto the P1 files' bare
element-symbol labels by geometric/topological role — both are exactly the "hand-convert a
structure without saying exactly what was done" the plan rules out, so neither was attempted. Per
the plan's own contingency, this half of task 9 is reported as blocked rather than worked around:
**no RASPA IRMOF-1 methane isotherm comparison was run.**

## Task 10: GCMC throughput against kUPS, and per-system memory

`docs/src/benchmarks.md`'s "GCMC (grand canonical) exchange moves" section is the summary; this
records the two things behind it that are not already covered elsewhere in this file.

**`bench/gpu/exchange_bench.jl`'s own committed numbers
(`pureadsorb_exchange_percall_neuromancer4070_cuda_f64_{before,after}_5918449.json`, 18.7 ms and
12.8 ms per call) are not cited in the docs.** That script calls `mc_exchange!` (the insert/delete
coin-flip wrapper) directly, and its untimed warm-up loop reseeds `rng = Xoshiro(1)` fresh on
every iteration rather than advancing one shared stream — the same fixed seed every time, so the
0.5 s wall-clock warm-up window only ever exercises whichever one of `mc_insert!`/`mc_delete!`
that fixed seed's coin flip happens to pick. `bench/gpu/units_overhead_bench.jl`'s own header
(written independently, while diagnosing a different benchmark) documents this exact trap —
"`mc_exchange!`'s own insert/delete coin flip needing BOTH kernel variants pre-compiled before
timing either one" — and measures its effect directly: with only the coin flip to rely on, every
one of ten `@be` samples read 1.9-3.2 ms uniformly (not one slow outlier), against 167 us once
both branches are explicitly warmed first. `exchange_bench.jl`'s own two numbers, both an order of
magnitude above `mc_insert!`/`mc_delete!`'s real per-call cost (below), are consistent with
carrying the same artifact (that script also predates the workgroup fan-out entirely, so its
"before"/"after" pair measures a different code change — removing a per-call
`adapt(CPU(), batch)` — not the fan-out this section otherwise reports on). Rather than re-measure
`exchange_bench.jl` itself, the throughput comparison uses `bench/gpu/exchange_workgroup_bench.jl`
instead, which was already built the correct way: its entire timed closure comes from one
function's (`run_one`) local, typed arguments, and it explicitly calls both `mc_insert!` and
`mc_delete!` once, untimed, before any warm-up loop starts.

**nsys=8 (kUPS's own GCMC ceiling, below) was added to `exchange_workgroup_bench.jl`'s sweep**
(it previously covered 1/64/256 only) via `PA_NSYS_LIST="1,8"`, RTX 4070, commit `9917bf5`, same
case as the existing workgroup-fan-out measurement (RUBTAK 3×3×3 + CO2, 10 initial guests/chain,
capacity 40, fugacity 2e4 Pa):

| precision | move | nsys | us/call | us/move (÷nsys) |
|---|---|---|---|---|
| Float64 | insert | 1 | 201.43 | 201.43 |
| Float64 | insert | 8 | 307.55 | 38.44 |
| Float64 | delete | 1 | 139.51 | 139.51 |
| Float64 | delete | 8 | 193.78 | 24.22 |
| Float32 | insert | 1 | 155.84 | 155.84 |
| Float32 | insert | 8 | 202.37 | 25.30 |
| Float32 | delete | 1 | 103.36 | 103.36 |
| Float32 | delete | 8 | 134.75 | 16.84 |

The nsys=1 rows (170.5/129.6 us mean exchange cost, Float64/Float32) sit a few percent from the
already-committed `..._after_20260927_f4a0503.json` nsys=1 rows (161.4/132.2 us mean) — run-to-run
noise of the same scale `units_overhead_bench.jl`'s own repeated-measurement note documents for
this kernel (a 57.6 us spread across four repeated Float64 measurements there), not a change in
`mc_insert!`/`mc_delete!` themselves: that file's `commit: f4a0503` records HEAD at the moment the
benchmark ran, when the workgroup fan-out existed only as the working-tree diff its own "before"
row came from via `git stash` (this file's own note on that benchmark); the fan-out landed
immediately afterward as `873c9ed`, and `git diff 873c9ed 9917bf5 -- src/moves.jl` is empty, so
`mc_insert!`/`mc_delete!`'s code at the current commit is identical to what the "after" file
measured. Files:
`pureadsorb_exchange_workgroup_neuromancer4070_cuda_{f64,f32}_task10_20260927_9917bf5.json`.

**PureAdsorb's own per-system GCMC device footprint**, `bench/gpu/gcmc_memory_bench.jl`: computed
analytically from `SystemState`'s own field lengths and element sizes (the same
length-times-`sizeof` pattern `bench/widom_scaling.jl`'s `bytes_per_system` and
`bench/gpu/cellwidth_sweep.jl`'s `bytes_per_framework` already use for `FrameworkBatch`), at the
isotherm run's own capacity (200) and RUBTAK 3×3×3's full k-vector table (`fullk = true`, nk =
4587, required once any guest is present): **122,952 B (Float64)**, **61,500 B (Float32)** —
`Sk` plus `sk_abs_accum` (both sized by the full k-vector table, not `capacity`) account for
110,088 B of the Float64 total. Against kUPS's own GCMC memory ceiling above (13.18 GiB requested
at nsys=16, treated as the whole 16-system batch per the existing Widom "Memory" section's own
"the batched-state allocation itself doubles with `nsys`" finding): roughly 13.18 GiB / 16 ≈ 885
MB/system for kUPS, against 120.1 KiB/system (Float64) for PureAdsorb — about 7,200× less. File:
`pureadsorb_gcmc_memory_neuromancer_20260927_9917bf5.json`.
