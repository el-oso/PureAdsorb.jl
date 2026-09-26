# bench

`audit.jl` gates type stability and allocation-freedom on the Widom kernels' hot path
(`insertion_energy`, `minimum_image`, `rotate`, `erfc_dev`, `pair_erfc_dev`, and the phase-0
hard-core rejection kernel's own cell-list primitives `home_cell_dev`, `stencil_start_count`,
`wrap_cell`, `cell_linear`). Run `julia --project=bench
bench/audit.jl` (fast) or `STRICT_MODE=full julia --project=bench bench/audit.jl` (the
AllocCheck/JET gate); both exit non-zero on a finding.

GPU throughput benchmarking (`widom_bench.jl` with `PA_BACKEND=cuda`/`rocm`) runs from
`bench/gpu`, not this environment; see `bench/results/README.md`.

`run_kups.sh` and `run_headtohead.sh` document how the kUPS reference numbers and head-to-head
timings in `bench/results/` were produced, from a kUPS checkout outside this repo. Neither
script is called by PureAdsorb or by any test. The head-to-head batches at kUPS's own memory
ceiling, not 64 (`nsys = 4` on the RTX 3050, `nsys = 8` on the RTX 4070): see
`bench/results/README.md` for why.
