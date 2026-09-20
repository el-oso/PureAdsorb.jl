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

### CPU (Float64, neuromancer)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.7476 | 13,376 | 5 |
| 1 | 100,000 | 6.0817 | 16,443 | 2 |

Kernel-only (one launch of `widom_kernel!` + synchronize on a 2¹⁶-pose chunk):

| nsys | median (s) | insertions/s |
|---|---|---|
| 1 | 3.9427 | 16,622 |
| 64 | 4.2583 | 15,390 |

### AMD Radeon AI PRO R9700 (Float64, galen, ROCm)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.2047 | 48,848 | 10 |
| 1 | 100,000 | 0.7877 | 126,959 | 10 |
| 1 | 1,000,000 | 6.164 | 162,242 | 5 |
| 64 | 10,000 | 0.2089 | 47,869 | 10 |
| 64 | 100,000 | 0.795 | 125,792 | 10 |
| 64 | 1,000,000 | 6.18 | 161,804 | 5 |

Kernel-only (run length 256): nsys=1, 0.3917 s (167,322 ins/s); nsys=64, 0.3919 s (167,207 ins/s).

### AMD Radeon AI PRO R9700 (Float32, galen, ROCm)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.01517 | 659,241 | 10 |
| 1 | 100,000 | 0.05317 | 1,880,730 | 10 |
| 1 | 1,000,000 | 0.4254 | 2,350,478 | 10 |
| 64 | 10,000 | 0.01554 | 643,392 | 10 |
| 64 | 100,000 | 0.05499 | 1,818,359 | 10 |
| 64 | 1,000,000 | 0.4306 | 2,322,542 | 10 |

Kernel-only (run length 256): nsys=1, 0.02161 s (3,032,082 ins/s); nsys=64, 0.02231 s
(2,937,811 ins/s).

### NVIDIA RTX 3050 (Float64, neuromancer, CUDA)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.5275 | 18,957 | 10 |
| 1 | 100,000 | 3.185 | 31,398 | 10 |
| 1 | 1,000,000 | 32.48 | 30,792 | 1 |
| 64 | 10,000 | 0.5423 | 18,441 | 10 |
| 64 | 100,000 | 3.181 | 31,437 | 10 |
| 64 | 1,000,000 | 32.24 | 31,021 | 1 |

Kernel-only (run length 256): nsys=1, 2.108 s (31,091 ins/s); nsys=64, 2.115 s (30,990 ins/s).

### NVIDIA RTX 3050 (Float32, neuromancer, CUDA)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.03157 | 316,790 | 10 |
| 1 | 100,000 | 0.1551 | 644,708 | 10 |
| 1 | 1,000,000 | 1.575 | 634,888 | 10 |
| 64 | 10,000 | 0.034 | 294,157 | 10 |
| 64 | 100,000 | 0.1669 | 599,135 | 10 |
| 64 | 1,000,000 | 1.651 | 605,675 | 10 |

Kernel-only (run length 256): nsys=1, 0.09944 s (659,081 ins/s); nsys=64, 0.1041 s
(629,587 ins/s).

Consumer GeForce cards throttle double-precision throughput relative to a datacenter part:
the Float64/Float32 gap on the RTX 3050 (roughly 20× at nsys=1) is far larger than the
Float64-only R9700 numbers above would suggest by themselves.

![Widom throughput per backend](assets/widom_throughput.png)

## Throughput against batch size

`bench/widom_scaling.jl` measures kernel throughput — one `widom_kernel!` launch plus
synchronization, on a chunk of 262,144 insertions — against the number of frameworks (`nsys`)
tiled into one batch, on the R9700 (galen, ROCm), at a given insertion run length. One
framework's host and Ewald tables are computed on the CPU and tiled onto the device; the batch
stays on the device for the whole sweep, and each batch size runs in its own process.

`widom` assigns insertion `g` of `1:ninsert` to system `mod1((g - 1) ÷ run + 1, nsys)`: `run`
consecutive insertions share a system before the assignment cycles to the next one, so device
work-items adjacent in the insertion order read the same framework's tables. The default run
length is `clamp((ninsert ÷ nsys) ÷ 4, 1, 256)`, reaching its ceiling of 256 once a system
receives at least 1,024 insertions.

### Run length 256

Run length 256 is what `default_run` gives at every batch size in this sweep (`ninsert = chunk =
262,144`, so `ninsert ÷ nsys >= 1,024` for every `nsys` tested).

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

### Run length 1 (round-robin)

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
with a profiler.

A screening run over distinct frameworks assigns insertions close to round-robin (many systems,
few insertions per system), so it operates near these figures rather than the run-length-256
figures above, unless it overrides `run` to a longer value itself.

No batch size in either sweep failed to allocate; 98,304 frameworks (Float64, 30.3 GiB) and
196,608 frameworks (Float32, 31.4 GiB) are the largest sizes tested on this card, not a
measured failure point.

![Widom kernel throughput vs batch size](assets/widom_scaling.png)

## kUPS head-to-head (RTX 3050, Float64)

`bench/run_headtohead.sh` alternates kUPS and PureAdsorb blocks for each `(nsys, ninsert)`
point on the same GPU, same cutoffs (LJ 12 Å, Ewald 12 Å, precision 1e-6), same force field,
host CIF and guest files (byte-identical to kUPS's own examples). kUPS times its whole process
(interpreter startup, JIT compilation, and the insertions); PureAdsorb's numbers are
`Chairmarks` samples of `widom(...)` warm, in-process. Comparing `ninsert / t` directly would
put a per-process cost against a per-call one on the same footing, so instead: for each `nsys`,
fit `t = intercept + ninsert / rate` by ordinary least squares over the median time at each
`ninsert`, and compare the fitted `rate`.

| Code | nsys | Fitted intercept (s) | Marginal rate (insertions/s) |
|---|---|---|---|
| kUPS | 1 | 16.87 | 2,281 |
| kUPS | 4 | 18.51 | 4,159 |
| PureAdsorb (warm in-process) | 1 | 0.092 | 31,128 |
| PureAdsorb (warm in-process) | 4 | 0.083 | 30,946 |

Ratio (PureAdsorb marginal rate / kUPS marginal rate): **13.6×** at nsys=1, **7.4×** at nsys=4
— kUPS batches insertions across systems more efficiently at nsys=4, while PureAdsorb was
already close to its per-call floor at nsys=1.

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
For comparison, in the same units, PureAdsorb's own per-framework device footprint
(`bytes_per_system` in `bench/widom_scaling.jl`) is 330,984 B ≈ 0.000308 GiB (Float64) and
171,648 B ≈ 0.000160 GiB (Float32) — measured on the 32 GB R9700 (galen), where the "Throughput
against batch size" section above runs batches up to 98,304 (Float64) and 196,608 (Float32)
frameworks.

## Caveats

- Single card per measurement; the RTX 3050 sits behind a Thunderbolt eGPU enclosure, and
  neuromancer's CPU clock is unpinned. Only galen (R9700) is clock-locked.
- Double-precision throughput on a GeForce card is throttled relative to a datacenter part;
  this affects both codes equally in the head-to-head, since both run Float64 there.
- One framework type (RUBTAK) and one guest (CO2) are measured throughout.
- `bench/run_headtohead.sh` sets the CPU governor to `performance` when writable; on
  neuromancer it was not (`powersave` throughout, recorded in each kUPS result's
  `meta.cpu_governor`).

## Reproducing

```bash
# CPU
julia --project=bench bench/widom_bench.jl

# GPU (separate environment — see bench/results/README.md for why)
julia --project=bench/gpu -e 'using Pkg; Pkg.instantiate()'
PA_BACKEND=cuda PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl
PA_BACKEND=cuda PA_PRECISION=f32 julia --project=bench/gpu bench/widom_bench.jl
PA_BACKEND=rocm PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl

# One (nsys, ninsert) grid point only, for a head-to-head run against kUPS
PA_BACKEND=cuda PA_GRID=4:1000000 PA_REPS=5 julia --project=bench/gpu bench/widom_bench.jl

# kUPS + PureAdsorb interleaved head-to-head (needs a kUPS checkout outside this repo)
bench/run_headtohead.sh

# Kernel throughput vs number of frameworks in a batch, round-robin (run length 1, the default
# of widom_scaling.jl's own PA_RUN)
PA_BACKEND=rocm PA_PRECISION=f64 julia --project=bench/gpu bench/widom_scaling.jl
PA_BACKEND=rocm PA_PRECISION=f32 julia --project=bench/gpu bench/widom_scaling.jl

# Same sweep at run length 256, the default widom itself uses at these batch sizes
PA_BACKEND=rocm PA_PRECISION=f64 PA_RUN=256 julia --project=bench/gpu bench/widom_scaling.jl
PA_BACKEND=rocm PA_PRECISION=f32 PA_RUN=256 julia --project=bench/gpu bench/widom_scaling.jl

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
or `f32`; `PA_GRID` restricts a `widom_bench.jl` sweep to one `nsys:ninsert` point and `PA_REPS`
sets its sample count; `PA_NSYS` restricts a `widom_scaling.jl` sweep to the given batch sizes
(space-separated, increasing); `PA_RUN` sets `widom_scaling.jl`'s insertion run length (default
1, round-robin). No plot is ever regenerated by re-running a benchmark — `plot_widom.jl` and
`plot_scaling.jl` only read `bench/results/*.json`.
