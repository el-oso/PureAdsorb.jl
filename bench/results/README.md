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

| host | GPU | backend | precision | result |
|---|---|---|---|---|
| neuromancer | — | cpu | f64 | `pureadsorb_widom_neuromancer_cpu_20260917.json` |
| galen | AMD Radeon AI PRO R9700 (gfx1201, Navi48/RDNA4) | rocm | f64 | `pureadsorb_widom_galen_rocm_20260917.json` |
| neuromancer | NVIDIA GeForce RTX 3050 6GB | cuda | f64 | `pureadsorb_widom_neuromancer_cuda_f64_20260919.json` |
| neuromancer | NVIDIA GeForce RTX 3050 6GB | cuda | f32 | `pureadsorb_widom_neuromancer_cuda_f32_20260919.json` |

The RTX 3050 sits behind a Thunderbolt eGPU enclosure on neuromancer, and neuromancer's CPU
clock is unpinned (see the top-level protocol note): its numbers are indicative only, never
gate-authoritative (galen and wintermute are the clock-locked, gate-authoritative hosts).
Consumer GeForce cards throttle double-precision throughput relative to a datacenter part, which
is why the f64/f32 gap on this card (~20x insertions/s) is far larger than the AMD Radeon AI
PRO R9700 numbers above.

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

## kUPS head-to-head: not present

No kUPS comparison numbers are present. `plot_widom.jl` picks up any additional results file
automatically.
