# Benchmarks

All numbers on this page are medians computed from the committed JSON files in
`bench/results/`; none are re-measured for this page. `BLAS.set_num_threads(1)` runs before
every timing in `bench/widom_bench.jl`.

## Machines

| Host | Hardware | Backend | Notes |
|---|---|---|---|
| neuromancer | CPU (16 threads) | KernelAbstractions CPU | CPU clock unpinned — indicative only |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | AMDGPU.jl (ROCm), AMDGPU 2.8.0 | clock-locked |
| neuromancer | NVIDIA GeForce RTX 3050 6 GB | CUDA.jl, driver 615.71.09 | Thunderbolt eGPU enclosure, CPU clock unpinned |

## PureAdsorb throughput

RUBTAK 3×3×3 + CO2, 12 Å cutoffs, Ewald precision 1e-6. `insertions/s = ninsert / median(times_s)`.
Each `widom` call runs two kernels per chunk: a hard-core rejection test flags every insertion
whose guest sites all stay outside a rigorous rejection radius of every host atom, then the
energy kernel computes `ΔU` for the survivors only. The host reads the flags, builds the
survivor index list, and uploads it between the two kernel launches (the "kernel path" numbers
below cover both kernels but not this host-side compaction step; "end-to-end" covers the whole
`widom` call, including it, plus pose generation, transfers and block accumulation).

### CPU (Float64, neuromancer)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.7476 | 13,376 | 5 |
| 1 | 100,000 | 6.0817 | 16,443 | 2 |

Kernel path (one launch of each kernel + synchronize, on a 2¹⁶-pose chunk):

| nsys | median (s) | insertions/s |
|---|---|---|
| 1 | 3.9427 | 16,622 |
| 64 | 4.2583 | 15,390 |

### AMD Radeon AI PRO R9700 (galen, ROCm), commit a4c86a7

`bench/results/pureadsorb_widom_galen_rocm_{f64,f32}_20260920_a4c86a7.json`.

End-to-end, Float64:

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.1290 | 77,525 | 10 |
| 1 | 100,000 | 0.3679 | 271,823 | 10 |
| 1 | 1,000,000 | 3.748 | 266,793 | 9 |
| 64 | 10,000 | 0.1292 | 77,392 | 10 |
| 64 | 100,000 | 0.3681 | 271,666 | 10 |
| 64 | 1,000,000 | 3.762 | 265,825 | 8 |

End-to-end, Float32:

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.02019 | 495,322 | 10 |
| 1 | 100,000 | 0.04668 | 2,142,125 | 10 |
| 1 | 1,000,000 | 0.4037 | 2,476,918 | 10 |
| 64 | 10,000 | 0.009633 | 1,038,059 | 10 |
| 64 | 100,000 | 0.02965 | 3,373,028 | 10 |
| 64 | 1,000,000 | 0.2742 | 3,647,215 | 10 |

Kernel path (chunk 65,536, run length 256; rejected fraction 40.5% at both precisions):

| Precision | nsys | rejection test (ms) | energy (ms) | total (ms) | insertions/s |
|---|---|---|---|---|---|
| Float64 | 1 | 1.450 | 232.407 | 233.856 | 280,241 |
| Float64 | 64 | 1.278 | 232.384 | 233.662 | 280,474 |
| Float32 | 1 | 0.351 | 18.490 | 18.841 | 3,478,392 |
| Float32 | 64 | 0.354 | 9.408 | 9.761 | 6,713,723 |

### NVIDIA RTX 3050 (neuromancer, CUDA), commit e903fac

`bench/results/pureadsorb_widom_neuromancer_cuda_{f64,f32}_20260920_e903fac.json`.

End-to-end, Float64:

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.1726 | 57,922 | 10 |
| 1 | 100,000 | 1.352 | 73,962 | 10 |
| 1 | 1,000,000 | 12.97 | 77,121 | 3 |
| 64 | 10,000 | 0.1776 | 56,306 | 10 |
| 64 | 100,000 | 1.359 | 73,574 | 10 |
| 64 | 1,000,000 | 13.09 | 76,414 | 3 |

The `ninsert = 1,000,000` points collect only 3 Chairmarks samples: each run takes about 13 s,
so the 30 s time budget for this grid point fits few repeats.

End-to-end, Float32:

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.01411 | 708,541 | 10 |
| 1 | 100,000 | 0.07301 | 1,369,644 | 10 |
| 1 | 1,000,000 | 0.6784 | 1,474,150 | 10 |
| 64 | 10,000 | 0.01770 | 564,987 | 10 |
| 64 | 100,000 | 0.08184 | 1,221,941 | 10 |
| 64 | 1,000,000 | 0.7452 | 1,341,972 | 10 |

Kernel path (chunk 65,536, run length 256; rejected fraction 40.5% at both precisions):

| Precision | nsys | rejection test (ms) | energy (ms) | total (ms) | insertions/s |
|---|---|---|---|---|---|
| Float64 | 1 | 7.295 | 838.481 | 845.775 | 77,486 |
| Float64 | 64 | 10.281 | 838.471 | 848.752 | 77,215 |
| Float32 | 1 | 1.191 | 38.599 | 39.791 | 1,647,021 |
| Float32 | 64 | 3.568 | 40.664 | 44.233 | 1,481,621 |

Consumer GeForce cards throttle double-precision throughput relative to a datacenter part: the
Float64/Float32 gap on the RTX 3050 (about 19× at nsys=1, kernel path) is far larger than the
Float64-only R9700 numbers above would suggest by themselves.

### Host-side share (Float32, R9700)

At nsys=1, Float32, the kernel path (3,478,392 insertions/s) runs faster than the end-to-end
call at 1,000,000 insertions (2,476,918 insertions/s). Pose generation, host↔device transfers,
the phase-0/phase-1 host-side compaction step, and block accumulation all run on the host and
account for the remaining time; end-to-end is not just the two kernels back to back.

### Kernel-path throughput on the R9700 by configuration

Each row is a distinct measured configuration at its own commit; only the last two rows include
Float32. The first six rows use `widom_bench.jl`'s kernel-only measurement (chunk 65,536); "Cell-
sorted atoms" instead uses `widom_scaling.jl`'s one-framework point (chunk 262,144), the only R9700
data recorded for that configuration.

| Configuration | Commit | Result file | Float64 (ins/s) | Float32 (ins/s) |
|---|---|---|---|---|
| Full reciprocal table, round-robin assignment | `91966c7` | `pureadsorb_widom_galen_rocm_20260917.json` | 155,142 | not measured (no Float32 support yet) |
| Runs of 256 | `c910867` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920.json` | 167,322 | 3,032,082 |
| Sparse reciprocal table | `584b806` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_584b806.json` | 230,728 | 3,975,541 |
| Cell-sorted atoms | `da327a6` | `pureadsorb_widom_scaling_galen_rocm_{f64,f32}_run256_20260920_da327a6.json` | 253,496 | 4,826,870 |
| Restricted-range pair term | `082c656` | RTX-3050-only measurement; not recorded on the R9700 | — | — |
| Core rejection | `3653c6f` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_3653c6f.json` | 280,754 | 3,503,406 |
| Current | `a4c86a7` | `pureadsorb_widom_galen_rocm_{f64,f32}_20260920_a4c86a7.json` | 280,241 | 3,478,392 |

![Widom throughput per backend](assets/widom_throughput.png)

## Throughput against batch size

`bench/widom_scaling.jl` measures kernel throughput — one hard-core rejection kernel launch plus
the energy kernel, on a chunk of 262,144 insertions — against the number of frameworks (`nsys`)
tiled into one batch, on the R9700 (galen, ROCm), at a given insertion run length. One
framework's host and Ewald tables are computed on the CPU and tiled onto the device; the batch
stays on the device for the whole sweep, and each batch size runs in its own process.

`widom` assigns insertion `g` of `1:ninsert` to system `mod1((g - 1) ÷ run + 1, nsys)`: `run`
consecutive insertions share a system before the assignment cycles to the next one, so device
work-items adjacent in the insertion order read the same framework's tables. The default run
length is `clamp((ninsert ÷ nsys) ÷ 4, 1, 256)`, reaching its ceiling of 256 once a system
receives at least 1,024 insertions.

### Run length 256, at commit c910867

Run length 256 is what `default_run` gives at every batch size in this sweep (`ninsert = chunk =
262,144`, so `ninsert ÷ nsys >= 1,024` for every `nsys` tested). This sweep, to the largest batch
size tested at each precision (98,304 frameworks Float64, 196,608 frameworks Float32), is
measured at commit `c910867`, before the hard-core rejection stage existed.

| frameworks | batch size (GiB) | median kernel insertions/s (Float64) |
|---|---|---|
| 1 | 0.0003083 | 169,292 |
| 64 | 0.01973 | 169,203 |
| 1,024 | 0.3157 | 169,105 |
| 8,192 | 2.525 | 168,790 |
| 32,768 | 10.1 | 168,792 |
| 65,536 | 20.2 | 168,736 |
| 98,304 | 30.3 | 168,596 |

| frameworks | batch size (GiB) | median kernel insertions/s (Float32) |
|---|---|---|
| 1 | 0.0001599 | 3,315,371 |
| 64 | 0.01023 | 3,299,559 |
| 1,024 | 0.1637 | 3,211,654 |
| 8,192 | 1.31 | 3,218,591 |
| 32,768 | 5.238 | 3,219,615 |
| 65,536 | 10.48 | 3,222,399 |
| 131,072 | 20.95 | 3,223,046 |
| 196,608 | 31.43 | 3,226,734 |

Throughput under run length 256 is flat within a few percent across the whole sweep, for both
precisions: Float64 stays within 0.4% (169,292 to 168,596 insertions/s), Float32 within 3.2%
(3,315,371 to 3,211,654 insertions/s, with a modest recovery to 3,226,734 at the largest batch).

### Run length 256, at the current commit (a4c86a7)

`bench/results/pureadsorb_widom_scaling_galen_rocm_{f64,f32}_run256_20260920_a4c86a7.json` cover
two batch sizes at the current commit, each also giving bytes/framework and the rejected
fraction (see [Memory](#memory)):

| Precision | frameworks | batch size (GiB) | rejected fraction | insertions/s |
|---|---|---|---|---|
| Float64 | 1 | 0.0001336 | 40.79% | 443,248 |
| Float64 | 32,768 | 4.377 | 40.79% | 436,939 |
| Float32 | 1 | 0.0000834 | 40.81% | 6,970,887 |
| Float32 | 32,768 | 2.733 | 40.81% | 8,911,287 |

In Float32, the kernel path runs faster with 32,768 frameworks (8,911,287 insertions/s) than
with one (6,970,887 insertions/s); this is measured, not explained here. Float64 does not show
the same effect (436,939 against 443,248 insertions/s).

### Run length sweep at 32,768 frameworks

`pureadsorb_widom_runlength_galen_rocm_20260920.json` fixes the batch at 32,768 frameworks and
varies the run length directly, for both precisions:

| run length | insertions/s (Float64) | insertions/s (Float32) |
|---|---|---|
| 1 | 164,029 | 294,865 |
| 16 | 169,213 | 3,033,118 |
| 256 | 169,265 | 3,240,284 |
| 4,096 | 169,185 | 3,344,249 |
| 262,144 | 169,208 | 3,342,436 |

Float64 throughput is close to its plateau already at run length 16 and does not move further
out to a run spanning the whole chunk. Float32 keeps rising past run length 256, reaching its
highest measured value at run length 4,096 (3,344,249 insertions/s) before leveling off.

### Run length 1 (round-robin), at commit c910867

| frameworks | batch size (GiB) | median kernel insertions/s |
|---|---|---|
| 1 | 0.0003 | 170,210 |
| 64 | 0.0197 | 169,137 |
| 1,024 | 0.316 | 165,956 |
| 8,192 | 2.525 | 165,841 |
| 32,768 | 10.10 | 162,576 |
| 65,536 | 20.20 | 161,946 |
| 81,920 | 25.25 | 161,540 |
| 90,112 | 27.78 | 161,492 |
| 94,208 | 29.04 | 161,301 |
| 98,304 | 30.30 | 161,019 |

| frameworks | batch size (GiB) | median kernel insertions/s |
|---|---|---|
| 1 | 0.00016 | 3,299,882 |
| 64 | 0.0102 | 1,786,096 |
| 1,024 | 0.164 | 471,919 |
| 8,192 | 1.310 | 321,149 |
| 32,768 | 5.238 | 295,531 |
| 65,536 | 10.48 | 294,262 |
| 131,072 | 20.95 | 293,941 |
| 163,840 | 26.19 | 293,881 |
| 180,224 | 28.81 | 293,688 |
| 188,416 | 30.12 | 293,678 |
| 196,608 | 31.43 | 293,585 |

The first table is Float64, the second Float32. Every work-item reads a different framework's
tables under round-robin, at every batch size above 1 — the opposite of run length 256, where
adjacent work-items share a framework. Float64 throughput falls by about 5% from 1 to 98,304
frameworks (170,210 to 161,019 insertions/s); Float32 falls much further, from its
one-framework value to a plateau reached by about 32,768 frameworks (295,531 insertions/s),
essentially flat out to 196,608 frameworks (293,585 insertions/s).

At the plateau, `bytes_per_system` — the per-framework device bytes the kernel reads — times
insertions/s gives an effective data rate: 330,984 B × 161,019 ins/s ≈ 53.3 GB/s (Float64, at
98,304 frameworks) and 171,648 B × 293,585 ins/s ≈ 50.4 GB/s (Float32, at 196,608 frameworks).
The two rates are close despite the precisions differing by 2× in bytes per element; this is
consistent with the kernel being limited by memory traffic at large batch sizes under
round-robin assignment, an inference from the throughput numbers above, not something confirmed
with a profiler. `bytes_per_system` at commit c910867 predates the hard-core rejection stage's
cell list and rejection tables, so it differs from the current commit's figures in
[Memory](#memory) below.

A screening run over distinct frameworks assigns insertions close to round-robin (many systems,
few insertions per system), so it operates near these figures rather than the run-length-256
figures above, unless it overrides `run` to a longer value itself.

No batch size in either sweep failed to allocate; 98,304 frameworks (Float64, 30.3 GiB) and
196,608 frameworks (Float32, 31.4 GiB) are the largest sizes tested on this card, not a
measured failure point.

![Widom kernel throughput vs batch size](assets/widom_scaling.png)

## kUPS head-to-head (RTX 3050, Float64, commit 695f083)

`bench/run_headtohead.sh` alternates kUPS and PureAdsorb blocks for each `(nsys, ninsert)`
point on the same GPU, same cutoffs (LJ 12 Å, Ewald 12 Å, precision 1e-6), same force field,
host CIF and guest files (byte-identical to kUPS's own examples). kUPS times its whole process
(interpreter startup, JIT compilation, and the insertions); PureAdsorb's numbers are
`Chairmarks` samples of `widom(...)` warm, in-process. Comparing `ninsert / t` directly would
put a per-process cost against a per-call one on the same footing, so instead: for each `nsys`,
fit `t = intercept + ninsert / rate` by ordinary least squares over the median time at each
`ninsert`, and compare the fitted `rate`. This comparison is recorded at PureAdsorb commit
`695f083`, before the hard-core rejection stage existed.

| Code | nsys | Fitted intercept (s) | Marginal rate (insertions/s) |
|---|---|---|---|
| kUPS | 1 | 16.87 | 2,281 |
| kUPS | 4 | 18.51 | 4,159 |
| PureAdsorb (warm in-process) | 1 | 0.092 | 31,128 |
| PureAdsorb (warm in-process) | 4 | 0.083 | 30,946 |

Ratio (PureAdsorb marginal rate / kUPS marginal rate): **13.6×** at nsys=1, **7.4×** at nsys=4
— kUPS batches insertions across systems more efficiently at nsys=4, while PureAdsorb was
already close to its per-call floor at nsys=1.

PureAdsorb's Float64 kernel path on the same RTX 3050, at the current commit (`e903fac`), runs
at 77,486 insertions/s (nsys=1, chunk 65,536) — see the throughput tables above for the full
current-commit measurement; no new ratio against the `695f083` kUPS timings is computed here,
since the two were measured by different methods (whole-process wall time against a warm,
in-process kernel-only launch).

### Fixed cost per process

| Code | Fixed cost per process (s) | What it includes |
|---|---|---|
| kUPS | ≈16.9–18.5 (regression intercept above) | interpreter startup, JIT compilation |
| PureAdsorb (CUDA) | 13.25 (median of 3 whole-process runs: 13.44, 13.20, 13.25) | Julia startup, package load, kernel compilation, one warm sample |

PureAdsorb's `Chairmarks` loop is warm in-process by design and so excludes this cost from the
throughput tables above; it is measured separately as whole-process wall time around
`PA_GRID=1:10000 PA_REPS=1 julia --project=bench/gpu bench/widom_bench.jl`
(`bench/results/pureadsorb_widom_processcost_neuromancer_f64_20260919.json`).

## Memory

kUPS batches insertions across all `nsys` systems into one compiled step, so its GPU memory
requirement grows with `nsys`. On the 6 GB RTX 3050, kUPS fails to build its batched state at
`nsys ∈ {8, 16, 32, 64}`:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate
51.26GiB.   # nsys=64
23.82GiB.   # nsys=32
10.08GiB.   # nsys=16
5.10GiB.    # nsys=8
```

PureAdsorb runs `nsys=64` on the same card without difficulty (see the throughput tables
above). kUPS is also Float64-only for this workload — `jax_enable_x64` is forced in
`kups.application.simulations.mcmc_widom` — so no Float32 comparison point exists for kUPS;
PureAdsorb's Float32 numbers are reported above for reference.

The kUPS figures above are on the 6 GB RTX 3050 (neuromancer); kUPS was not run on the R9700.
For comparison, in the same units, PureAdsorb's own per-framework device footprint at the
current commit is 143,436 B ≈ 0.000134 GiB (Float64) and 89,552 B ≈ 0.0000834 GiB (Float32) —
computed from a `FrameworkBatch` for RUBTAK 3×3×3 + CO2 (default `cellwidth = 2`), the same
configuration the "Throughput against batch size" section above runs at up to 98,304 (Float64)
and 196,608 (Float32) frameworks on the 32 GB R9700 (galen).

## Caveats

- Single card per measurement; the RTX 3050 sits behind a Thunderbolt eGPU enclosure, and
  neuromancer's CPU clock is unpinned. Only galen (R9700) is clock-locked.
- Double-precision throughput on a GeForce card is throttled relative to a datacenter part;
  this affects both codes equally in the head-to-head, since both run Float64 there.
- One framework type (RUBTAK) and one guest (CO2) are measured throughout.
- `bench/run_headtohead.sh` sets the CPU governor to `performance` when writable; on
  neuromancer it was not (`powersave` throughout, recorded in each kUPS result's
  `meta.cpu_governor`).
- The R9700 throughput tables above are all commit `a4c86a7`; galen is not touched to
  re-measure them at a later commit.

## Reproducing

```bash
# CPU
julia --project=bench bench/widom_bench.jl

# GPU (separate environment — see bench/results/README.md for why)
julia --project=bench/gpu -e 'using Pkg; Pkg.instantiate()'
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=cuda PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=cuda PA_PRECISION=f32 julia --project=bench/gpu bench/widom_bench.jl
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl

# A different phase-0 cell width (Å); default is 2
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f32 PA_CELLWIDTH=3 julia --project=bench/gpu bench/widom_bench.jl

# One (nsys, ninsert) grid point only, for a head-to-head run against kUPS
PA_BACKEND=cuda PA_GRID=4:1000000 PA_REPS=5 julia --project=bench/gpu bench/widom_bench.jl

# kUPS + PureAdsorb interleaved head-to-head (needs a kUPS checkout outside this repo)
bench/run_headtohead.sh

# Kernel throughput vs number of frameworks in a batch, round-robin (run length 1, the default
# of widom_scaling.jl's own PA_RUN)
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f64 julia --project=bench/gpu bench/widom_scaling.jl
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f32 julia --project=bench/gpu bench/widom_scaling.jl

# Same sweep at run length 256, the default widom itself uses at these batch sizes
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f64 PA_RUN=256 julia --project=bench/gpu bench/widom_scaling.jl
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f32 PA_RUN=256 julia --project=bench/gpu bench/widom_scaling.jl

# Near device capacity, one process per batch size: memory freed by a smaller batch is not
# returned to the device within the process, so the next large allocation would otherwise stall
PA_BACKEND=rocm PA_PRECISION=f64 PA_NSYS=81920 julia --project=bench/gpu bench/widom_scaling.jl
PA_BACKEND=rocm PA_PRECISION=f64 PA_NSYS=90112 julia --project=bench/gpu bench/widom_scaling.jl
PA_BACKEND=rocm PA_PRECISION=f64 PA_NSYS=98304 julia --project=bench/gpu bench/widom_scaling.jl

# Regenerate the plots above and the docs figures from the committed JSON only
julia --project=bench bench/plot_widom.jl
julia --project=bench bench/plot_scaling.jl
PA_PLOT_OUT=docs/src/assets/widom_throughput.png julia --project=bench bench/plot_widom.jl
PA_PLOT_OUT=docs/src/assets/widom_scaling.png julia --project=bench bench/plot_scaling.jl
```

`PA_BACKEND` selects `cpu` (default), `cuda` or `rocm`; `PA_PRECISION` selects `f64` (default)
or `f32`; `PA_COMMIT` records the commit a result file is attributed to (defaults to
`git rev-parse --short HEAD` at run time) and drives the `_<commit>.json` suffix on the output
file; `PA_CELLWIDTH` sets the phase-0 cell list's target width in Å (default 2); `PA_GRID`
restricts a `widom_bench.jl` sweep to one `nsys:ninsert` point and `PA_REPS` sets its sample
count; `PA_NSYS` restricts a `widom_scaling.jl` sweep to the given batch sizes (space-separated,
increasing); `PA_RUN` sets `widom_scaling.jl`'s insertion run length (default 1, round-robin).
No plot is ever regenerated by re-running a benchmark — `plot_widom.jl` and `plot_scaling.jl`
only read `bench/results/*.json`.
