# Benchmarks

PureAdsorb against [kUPS](https://github.com/cusp-ai-oss/kups) on the same GPU, followed by
PureAdsorb on other hardware. Numbers below are computed from the committed JSON files in
`bench/results/`; nothing here is re-measured for this page.

## Widom insertion rate, RTX 4070 12 GB

RUBTAK 3×3×3 + CO2, LJ 12 Å / Ewald 12 Å cutoffs, Ewald precision 1e-6 — the same case, same
host CIF, force field and guest files (byte-identical to kUPS's own examples) on both codes, run
on the same card (host `neuromancer4070`, a USB4 eGPU enclosure).

| Code | Precision | Frameworks | Marginal rate (insertions/s) | Ratio to kUPS |
|---|---|---|---|---|
| kUPS | Float64 | 1 | 3,589 | 1.0× |
| kUPS | Float64 | 8 | 10,942 | — (kUPS's own best batch on this card) |
| PureAdsorb | Float64 | 1 | 263,818 | 73.5× |
| PureAdsorb | Float64 | 8 | 263,169 | 24.1× |
| PureAdsorb | Float64 | 64 | 263,806 | 24.1× (vs. kUPS's 8-fw row) |
| PureAdsorb | Float32 | 1 | 3,812,315 | 1,062.2× (reference; kUPS has no Float32 mode) |
| PureAdsorb | Float32 | 64 | 3,799,434 | 1,058.6× (reference) |

PureAdsorb's Float64 marginal rate is 73.5× kUPS's at one framework each — the literal,
same-batch-size comparison — and 24.1× kUPS's own best batch on this card (8 frameworks, its
memory ceiling; PureAdsorb's rate barely depends on the batch size, so its 8-fw and 64-fw rows
agree to within 0.3%). Both ratios come from the same two codes on the same card; the smaller,
fairer number is reported alongside the larger one, not in its place. kUPS forces
`jax_enable_x64` and has no Float32 mode for this workload, so the Float32 rows have no kUPS
counterpart; they are compared against kUPS's Float64 rate at 1 framework for reference only.

![Widom marginal insertion rate: PureAdsorb vs kUPS](assets/widom_vs_kups.png)

## Widom insertion rate, other hardware

The same case and source revision (`src/` is unchanged across the commits below, verified by
`git diff --stat <commit> b26cb8a -- src/`), on other cards. kUPS is not measured on any of
these, so the ratio column crosses machines and is not a like-for-like comparison of the two
codes.

| Code | Precision | Frameworks | Marginal rate (insertions/s) | Ratio to kUPS (RTX 4070, 1 fw) |
|---|---|---|---|---|
| PureAdsorb (R9700, ROCm) | Float64 | 1 | 270,794 | 75.4× |
| PureAdsorb (R9700, ROCm) | Float64 | 64 | 269,620 | 75.1× |
| PureAdsorb (R9700, ROCm) | Float32 | 1 | 2,700,433 | 752.4× |
| PureAdsorb (R9700, ROCm) | Float32 | 64 | 3,708,049 | 1,033.1× |
| PureAdsorb (Apple M6, Metal) | Float32 | 1 | 2,118,441 | 590.2× |
| PureAdsorb (Apple M6, Metal) | Float32 | 64 | 2,112,754 | 588.6× |
| PureAdsorb (Apple M6, CPU) | Float64 | 1 | 243,011 | 67.7× |

Apple GPUs have no double precision, so there is no Metal Float64 row; the CPU row on the same
machine is its Float64 reference instead. Comparing PureAdsorb against itself at 1 framework,
same code: the R9700 is 1.0× the RTX 4070 in Float64 and 0.7× in Float32; the Apple M6's Metal
backend is 0.6× the RTX 4070 in Float32.

## NVT (canonical Monte Carlo) throughput, RTX 4070 12 GB

RUBTAK 3×3×3 + 50 CO2, `mc_step!` (one Metropolis translation attempt per chain per kernel
launch, the workgroup-per-chain kernel run at its default `groupsize = 64` — the minimax choice
over the groupsize sweep in `pureadsorb_groupsizesweep_neuromancer4070_cuda_{f64,f32}_20260927_b6cc175.json`,
detailed in `bench/results/README.md`) against kUPS's own `examples/nvt_co2_pressure_test.yaml`
case (the same 50-CO2 system) on the same card.

| Precision | Chains | Cost per move (µs) | Ratio to kUPS (387.6 µs/move) |
|---|---|---|---|
| Float64 | 1     | 187.2 | 2.1× |
| Float64 | 64    | 18.2  | 21.3× |
| Float64 | 256   | 15.8  | 24.6× |
| Float64 | 1,024 | 10.8  | 36.0× |
| Float32 | 1     | 177.7 | 2.2× |
| Float32 | 64    | 3.55  | 109.0× |
| Float32 | 256   | 1.98  | 195.5× |
| Float32 | 1,024 | 1.39  | 278.7× |

PureAdsorb's numbers are `per_move_s` from `pureadsorb_mcstep_neuromancer4070_cuda_f64_20260927_5ff7620.json`
and the `f32` file alongside it. kUPS's reference rate is the same OLS fit as the Widom
comparison above (`t = intercept + nmoves/rate`) over its own single-system timing sweep
(`kups_nvt_timing_neuromancer4070_f64_20260927.json`: nmoves = 100,000 / 500,000 / 1,000,000;
fitted intercept 24.3 s, rate 2,580 moves/s, i.e. 387.6 µs/move, residuals under 1% at every
point). **Batching does not raise kUPS's rate: 387.6 µs/move is its peak on this card at any
chain count.** Its own chain-count sweep
(`kups_nvt_nscale_neuromancer4070_f64_20260927.json`) gives an aggregate throughput, once the
24.3 s startup is subtracted, of about 1,880 moves/s at its largest working batch (32
systems) — lower than the single-system rate — and 64 systems fails outright during state
construction with `RESOURCE_EXHAUSTED` trying to allocate 10.49 GiB. So the single-system
figure is kUPS's best case on this card at any batch size, and every ratio above is measured
against that ceiling, not against a smaller number a larger kUPS batch might have reached.

![NVT cost per move: PureAdsorb vs kUPS](assets/nvt_vs_kups.png)

## Method

Both codes run on the same RTX 4070 (host `neuromancer4070`), but kUPS times its whole process
(interpreter startup, JIT compilation, and the insertions) while PureAdsorb's samples are
`Chairmarks` measurements of `widom(...)` warm, in-process — so `ninsert / t` is not comparable
between them directly. Instead, for each `nsys`, `t = intercept + ninsert / rate` is fit by
ordinary least squares over the median time at each `ninsert`, and the fitted `rate` is
compared; the intercept absorbs kUPS's per-process startup cost, which PureAdsorb's warm
in-process timing never pays. kUPS and PureAdsorb are timed in an interleaved run on this
machine (kUPS at commit `e183c9a`, PureAdsorb at commit `b26cb8a`), alternating a kUPS block
with a PureAdsorb block at each `(nsys, ninsert)` point so neither code is measured entirely
cold or entirely GPU-warmed relative to the other. The other-hardware numbers use the same fit
and the same grid, run separately rather than interleaved (PureAdsorb commits `e903fac` for the
R9700 and `53f8e40`/`97efb80` for the Apple M6).

The three-point grid (`ninsert` = 10^4, 10^5, 10^6) spans two decades, which gives the largest
point most of an unweighted fit's leverage: the fitted rate describes the large-run limit, not
a reliable small-run estimate, and the intercept is not a reliable estimate of small-run
overhead. Residuals of the fitted line against the median times, RTX 4070, nsys=1: PureAdsorb
Float64 is +34.7% at `ninsert=10^4`, -12.7% at `10^5`, +0.11% at `10^6`; PureAdsorb Float32 is
+21.6%, -8.1%, +0.08% at the same three points. kUPS's residuals stay under 2.5% everywhere
(largest magnitude 2.4%, at `ninsert=10^4`), since its own ~17-20 s fixed cost dominates the
small end and its per-insertion rate dominates the large end evenly.

### Transfer is not the bottleneck

The RTX 4070 sits behind a USB4 (not PCIe-native) eGPU enclosure (an ASMedia ASM2464-class
bridge negotiating 40 Gb/s), so its host-device link runs PCIe gen 4 at 4 lanes under sustained
load rather than the card's native 16-lane slot; measured host-device bandwidth at the transfer
sizes `widom`'s own per-chunk copies use is 1.3-1.6 GB/s. Against the kernel-only phase0+phase1
time at `nsys=1` (242.4 ms Float64, 12.45 ms Float32, both medians from the committed
`kernel_only_s` samples), the per-chunk transfer is **0.55% of kernel time at Float64 and 6.1%
at Float32 — not the bottleneck at either precision on this card.**

Scaling the `ninsert=10^6` end-to-end median down to one chunk-equivalent and subtracting the
kernel-only time gives a total non-kernel overhead per chunk of 3.2% (Float64) and 29.0%
(Float32) of end-to-end time. Only a small part of that is the measured transfer time above;
the remainder — host-side pose generation, survivor compaction, and Boltzmann-weight
accumulation — has not been decomposed further. That 29% at Float32 is unmeasured, not
attributed to any one of those candidates.

## Fixed cost per process

| Code | Fixed cost per process (s) | What it includes |
|---|---|---|
| kUPS | 17.2-20.5 (regression intercept above) | interpreter startup, JIT compilation |
| PureAdsorb (CUDA) | 16.18 (median of 3 whole-process runs) | Julia startup, package load, kernel compilation, one warm sample |

PureAdsorb's fixed cost is `bench/results/pureadsorb_widom_processcost_neuromancer4070_f64_20260926.json`,
measured as whole-process wall time around a single-sample run, the same way kUPS's cost is a
whole-process time. The two costs are close on this card: process startup is not a large
advantage for either code here.

## Memory

kUPS batches insertions across all `nsys` systems into one compiled step, so its memory
requirement grows with `nsys` and is set by the batched-state size, not the card. On this 12 GB
card it builds `nsys ∈ {1, 2, 4, 8}` but fails at `nsys=16` with `RESOURCE_EXHAUSTED` trying to
allocate 10.08 GiB — the identical allocation size that failed on a 6 GB card at the same
`nsys=16`; doubling the card's memory (6 GB to 12 GB) bought exactly one more doubling of usable
batch size (4 to 8), not an unbounded increase, because the batched-state allocation itself
doubles with `nsys`. JAX also reserves most of the GPU's memory at process start by default,
independent of this allocation. PureAdsorb runs `nsys = 64` on the same card without difficulty.

PureAdsorb's own per-framework device footprint (RUBTAK 3×3×3 + CO2, default `cellwidth = 2`) is
143,436 B (Float64) and 89,552 B (Float32), from `bytes` in
`bench/results/pureadsorb_widom_scaling_galen_rocm_f64_run256_20260920_a4c86a7.json` and the
`f32` file alongside it.

## Reproducing

```bash
# kUPS timing (needs a kUPS checkout outside this repo, at commit e183c9a)
PA_HOST=neuromancer4070 bench/run_headtohead.sh

# PureAdsorb, at the current commit
julia --project=bench/gpu -e 'using Pkg; Pkg.instantiate()'
PA_HOST=neuromancer4070 PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=cuda PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl
PA_HOST=neuromancer4070 PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=cuda PA_PRECISION=f32 julia --project=bench/gpu bench/widom_bench.jl

# R9700, on galen: same script, ROCm backend
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f64 julia --project=bench/gpu bench/widom_bench.jl
PA_COMMIT=$(git rev-parse --short HEAD) PA_BACKEND=rocm PA_PRECISION=f32 julia --project=bench/gpu bench/widom_bench.jl

# Apple M6, on brutus: CPU reference, then the Metal backend (Float32 only)
julia --project=bench/metal -e 'using Pkg; Pkg.instantiate()'
julia --project=bench/metal bench/widom_bench.jl
PA_BACKEND=metal PA_PRECISION=f32 julia --project=bench/metal bench/widom_bench.jl

# Regenerate the figure above from the committed JSON only
julia --project=bench bench/plot_headtohead.jl
PA_PLOT_OUT=docs/src/assets/widom_vs_kups.png julia --project=bench bench/plot_headtohead.jl

# NVT (mc_step!) chain-count sweep, PureAdsorb
PA_HOST=neuromancer4070 PA_BACKEND=cuda PA_PRECISION=f64 julia --project=bench/gpu bench/mc_step_bench.jl
PA_HOST=neuromancer4070 PA_BACKEND=cuda PA_PRECISION=f32 julia --project=bench/gpu bench/mc_step_bench.jl

# NVT reference, kUPS (needs a kUPS checkout outside this repo, at commit e183c9a)
KUPS=~/src/kups PA_HOST=neuromancer4070 bench/run_kups_nvt.sh timing
KUPS=~/src/kups PA_HOST=neuromancer4070 bench/run_kups_nvt.sh nscale

# Regenerate the NVT figure above from the committed JSON only
julia --project=bench bench/plot_nvt_vs_kups.jl
PA_PLOT_OUT=docs/src/assets/nvt_vs_kups.png julia --project=bench bench/plot_nvt_vs_kups.jl
```

See `bench/results/README.md` for every other measurement recorded in this repository (CPU and
AMD Radeon AI PRO R9700 throughput, batch-size and run-length scaling, per-configuration
history, and the RTX 3050's own head-to-head numbers), and `docs/src/validation.md` for the
accuracy comparison against kUPS.
