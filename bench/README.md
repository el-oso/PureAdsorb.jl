# bench

`audit.jl` gates type stability and allocation-freedom on the Widom kernel's hot path
(`insertion_energy`, `minimum_image`, `rotate`, `erfc_dev`). Run `julia --project=bench
bench/audit.jl` (fast) or `STRICT_MODE=full julia --project=bench bench/audit.jl` (the
AllocCheck/JET gate); both exit non-zero on a finding.

GPU throughput benchmarking (`widom_bench.jl` with `PA_BACKEND=cuda`/`rocm`) runs from
`bench/gpu`, not this environment; see `bench/results/README.md`.
