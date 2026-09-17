# bench

`audit.jl` gates the type stability and allocation-freedom of `insertion_energy`,
`minimum_image`, `rotate` and `erfc_dev` — the Widom kernel's hot path.

Run it from a fresh checkout with `julia --project=bench bench/audit.jl` (fast, value-free
scan), or `STRICT_MODE=full julia --project=bench bench/audit.jl` for the AllocCheck/JET-backed
gate; either exits non-zero on a finding.
