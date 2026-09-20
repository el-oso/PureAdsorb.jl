# PureAdsorb

PureAdsorb computes gas adsorption properties of porous crystals by batched Widom
test-particle insertion, on CPU or GPU through KernelAbstractions.jl, in pure Julia. Given a
batch of host frameworks, a rigid guest molecule and a Lennard-Jones + Ewald force field, it
reports the excess chemical potential, Henry coefficient and zero-loading isosteric heat of
adsorption, each with a standard error. Two [`KernelAbstractions`](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernels — a hard-core rejection test, then the energy on the survivors — run unchanged on the
CPU, on NVIDIA GPUs (CUDA.jl) and on AMD GPUs (AMDGPU.jl).

PureAdsorb reproduces the conventions of [kUPS](https://github.com/cusp-ai-oss/kups)
(commit `e183c9a`), CuspAI's Widom/Monte Carlo engine, so results from the two codes compare
directly: the same energy decomposition, the same Ewald parameter selection, and the same
input file formats. See [Validation](validation.md) for the cross-code comparison.

## Input formats

- **Host structure**: a P1 CIF file with an `_atom_site_charge` column carrying partial
  charges (required — [`read_cif`](@ref)). Other space groups and CIFs without charges are
  rejected.
- **Force field**: a kUPS-style YAML file listing per-type Lennard-Jones `[σ, ε]` (Å, eV)
  under `parameters`, plus `cutoff` and `tail_correction` ([`read_forcefield`](@ref)).
- **Guest**: a kUPS-style YAML file giving site `positions`, `symbols`, `charges` and the
  critical constants of the rigid guest molecule ([`read_guest`](@ref)).

## Installation

PureAdsorb is not registered. Add it by path:

```julia
using Pkg
Pkg.develop(path = "/path/to/PureAdsorb.jl")
```

or, in a project's `Project.toml`:

```toml
[sources]
PureAdsorb = {path = "/path/to/PureAdsorb.jl"}
```

## Usage

```julia
fw    = read_cif("RUBTAK.cif")                       # P1 CIF with partial charges
ff    = read_forcefield("trappe.yaml")
co2   = read_guest("co2.yaml", ff)
batch = FrameworkBatch([replicate(fw, (3, 3, 3))], ff, co2, EwaldParams(cutoff = 12.0, precision = 1e-6))
res   = widom(batch, co2; T = 298.15, ninsert = 1_000_000, seed = 42)
# res   = widom(batch, co2; T = 298.15, ninsert = 1_000_000, seed = 42, backend = CUDABackend())
res[1].K_H, res[1].mu_ex, res[1].q_st, res[1].K_H_err
```

## Where it runs

| Host | Hardware | Backend |
|---|---|---|
| neuromancer | CPU | KernelAbstractions CPU backend |
| galen | AMD Radeon AI PRO R9700 (ROCm) | AMDGPU.jl |
| neuromancer | NVIDIA GeForce RTX 3050 6 GB (Thunderbolt eGPU) | CUDA.jl |

See [Theory](theory.md) for the physics, [Design](design.md) for the architecture,
[Validation](validation.md) for the test oracles and the kUPS comparison, and
[Benchmarks](benchmarks.md) for throughput numbers, including kernel throughput against batch
size on the R9700.
