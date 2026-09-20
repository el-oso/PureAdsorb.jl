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
| 1 | 10,000 | 0.2215 | 45,137 | 10 |
| 1 | 100,000 | 0.8471 | 118,051 | 10 |
| 1 | 1,000,000 | 6.6292 | 150,849 | 5 |
| 64 | 10,000 | 0.2225 | 44,951 | 10 |
| 64 | 100,000 | 0.8548 | 116,981 | 10 |
| 64 | 1,000,000 | 6.6768 | 149,773 | 5 |

Kernel-only: nsys=1, 0.4224 s (155,142 ins/s); nsys=64, 0.4245 s (154,372 ins/s).

### NVIDIA RTX 3050 (Float64, neuromancer, CUDA)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.5335 | 18,743 | 10 |
| 1 | 100,000 | 3.1967 | 31,282 | 10 |
| 1 | 1,000,000 | 32.2178 | 31,039 | 1 |
| 64 | 10,000 | 0.5424 | 18,435 | 10 |
| 64 | 100,000 | 3.2482 | 30,786 | 10 |
| 64 | 1,000,000 | 33.1286 | 30,185 | 1 |

Kernel-only: nsys=1, 2.1309 s (30,756 ins/s); nsys=64, 2.1654 s (30,264 ins/s).

### NVIDIA RTX 3050 (Float32, neuromancer, CUDA)

| nsys | ninsert | median (s) | insertions/s | samples |
|---|---|---|---|---|
| 1 | 10,000 | 0.02483 | 402,714 | 10 |
| 1 | 100,000 | 0.15571 | 642,224 | 10 |
| 1 | 1,000,000 | 1.58056 | 632,686 | 10 |
| 64 | 10,000 | 0.06669 | 149,953 | 10 |
| 64 | 100,000 | 0.36208 | 276,182 | 10 |
| 64 | 1,000,000 | 3.18718 | 313,757 | 10 |

Kernel-only: nsys=1, 0.09976 s (656,932 ins/s); nsys=64, 0.18911 s (346,553 ins/s).

Consumer GeForce cards throttle double-precision throughput relative to a datacenter part:
the Float64/Float32 gap on the RTX 3050 (roughly 20× at nsys=1) is far larger than the
Float64-only R9700 numbers above would suggest by themselves.

![Widom throughput per backend](assets/widom_throughput.png)

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

# Regenerate the plot above and the docs figure from the committed JSON only
julia --project=bench bench/plot_widom.jl
PA_PLOT_OUT=docs/src/assets/widom_throughput.png julia --project=bench bench/plot_widom.jl
```

`PA_BACKEND` selects `cpu` (default), `cuda` or `rocm`; `PA_PRECISION` selects `f64` (default)
or `f32`; `PA_GRID` restricts the sweep to one `nsys:ninsert` point and `PA_REPS` sets its
sample count. No plot is ever regenerated by re-running a benchmark — `plot_widom.jl` only
reads `bench/results/*.json`.
