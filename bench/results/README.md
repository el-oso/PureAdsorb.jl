# Widom throughput benchmark results

## Protocol

`bench/widom_bench.jl` times `widom(batch, guest; ...)` end to end with `Chairmarks.@be`
(`evals = 1`, wall time only — no GPU event timers) over a grid of `(nsys, ninsert)` points,
plus a kernel-only measurement that times one `widom_kernel!` launch (+
`KernelAbstractions.synchronize`) on a single prepared chunk of `2^16` poses, isolating kernel
throughput from the per-chunk RNG fill and host<->device copies that the end-to-end call also
pays. `BLAS.set_num_threads(1)` runs before any timing, since idle OpenBLAS threads spin-wait
and contend with the benchmarked code. Every sample's wall time is written to
`bench/results/*.json`; nothing is derived by re-running a benchmark.

The grid is smaller on CPU than on a GPU backend, because assembling an `nsys=64` batch (its
Ewald k-vector tables in particular) costs ~9 s on the CPU host and `ninsert=10^6` would run
for minutes per sample there:

| backend | nsys | ninsert | seconds | samples |
|---|---|---|---|---|
| cpu | 1 | 10^4, 10^5 | 10 | 5 |
| cuda / rocm | 1, 64 | 10^4, 10^5, 10^6 | 30 | 10 |

The kernel-only measurement always uses `nsys ∈ (1, 64)` with `chunk = 2^16`. The grid actually
used, the host, the GPU name (when applicable), `Threads.nthreads()`, and the benchmark's
`seconds`/`samples` are all recorded in each JSON's `meta` block, so a file is self-describing
regardless of which host produced it. There is no CPU governor step here — no CPU-vs-kUPS
interleaving is run (see below), so nothing needs the `performance` governor.

Regenerate the plot from the committed JSON, without running anything:

```
julia --project=bench bench/plot_widom.jl
```

## Machines

| host | GPU | backend | status |
|---|---|---|---|
| neuromancer | — | cpu | `pureadsorb_widom_neuromancer_cpu_20260917.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | blocked — see below |

## ROCm on galen: blocked, not run

`bench/widom_bench.jl PA_BACKEND=rocm` does not produce a result on galen as of 2026-09-17.
`widom_kernel!` reduces to the same generic `@kernel function` on every backend, and a minimal
repro (`nsys=1`, an 8-lane chunk) fails identically to the full grid, so this is not
scale-dependent:

- **Julia 1.13.0 (the juliaup default here) + AMDGPU 2.7.0**: `GPUCompiler`'s IR validator
  (`check_ir!`, reached from `AMDGPU.Compiler.hipcompile`) reports `Reason: unsupported dynamic
  function invocation (call to convert)` / `unsupported call to an unknown function (call to
  jl_f_throw_methoderror)` against the compiled kernel once, but printing that diagnostic
  segfaults on every other attempt (`typekeyvalue_hash` / `jl_inst_arg_tuple_type`, called from
  `check_ir!` at `GPUCompiler/src/validation.jl:297`) — reproduced 4/5 runs.
- **Julia 1.12.7 + the same `bench/Manifest.toml`**: fails a different way before reaching the
  kernel at all — `LLVM error: Invalid attribute group entry (Producer: 'LLVM20.0.0git' Reader:
  'LLVM 18.1.7jl')` while linking AMDGPU's bundled ROCm device-library bitcode
  (`AMDGPU.Compiler.load_and_link!`), i.e. `ROCmDeviceLibs_jll` as resolved for AMDGPU 2.7.0 was
  built against a newer LLVM than Julia 1.12 bundles.

Both failures are in the AMDGPU.jl / GPUCompiler / ROCm toolchain, not reproducible on the CPU
backend, and outside `bench/`'s scope to fix. Until one of the two combinations above actually
compiles a kernel on gfx1201, add `pureadsorb_widom_galen_rocm_<date>.json` here and rerun
`bench/plot_widom.jl` — no code change to `bench/` is needed.

## kUPS head-to-head: not present

`bench/run_headtohead.sh` (the kUPS-vs-PureAdsorb driver) and any `kups_widom_timing_*.json`
are a separate, not-yet-approved piece of work and are not in this checkout. `bench/plot_widom.jl`
already draws a kUPS series if one of those JSON files appears here later; nothing needs to
change to pick it up.
