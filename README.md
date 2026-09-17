# PureAdsorb

PureAdsorb computes gas adsorption properties of porous crystals by batched Widom test-particle
insertion, on CPU or GPU through KernelAbstractions.jl, in pure Julia. Given a batch of host
frameworks, a rigid guest molecule and a Lennard-Jones + Ewald force field, it reports the
excess chemical potential, Henry coefficient and zero-loading isosteric heat of adsorption, each
with a standard error. Host structures are read from P1 CIF files with partial charges; force
fields and guests are read from kUPS-style YAML files, so the same inputs run on both codes.

```julia
fw   = Framework("RUBTAK.cif")                      # P1 CIF with charges
ff   = ForceField("trappe.yaml")                    # or a Dict of σ, ε per type
co2  = Guest("co2.yaml")
batch = FrameworkBatch([replicate(fw, (3,3,3))]; ff, guest = co2, ewald = (cutoff = 12.0, precision = 1e-6))
res  = widom(batch, co2, ff; T = 298.15, ninsert = 1_000_000, backend = CUDABackend(), seed = 42)
res[1].K_H, res[1].μ_ex, res[1].q_st, res[1].K_H_err
```
