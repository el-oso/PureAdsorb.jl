# PureAdsorb.jl Milestone C — grand canonical Monte Carlo

Status: draft for approval. Nothing here is built.

Milestone B samples a fixed number of guests. Milestone C lets the guest count fluctuate at fixed
chemical potential, which is what produces an **adsorption isotherm** — how much gas a material
holds at each pressure. That is the question adsorption science actually asks, and it is the
reason this package would be useful to someone rather than merely fast.

Validation targets: kUPS GCMC (`examples/mcmc_rigid.yaml`), and the published RASPA isotherm for
methane in IRMOF-1.

## 1. The ensemble

Fixed chemical potential μ, volume V, temperature T; the guest count N fluctuates. The
distribution over states is

    P(N, x) ∝ (1/N!) (V/Λ³)^N exp(βμN) exp(−U(N,x)/kT)

where Λ is the thermal de Broglie wavelength. Λ and the ideal-gas part are awkward to carry
around, so the standard move is to reparameterize in terms of **fugacity** `f`, which is the
pressure an ideal gas would need to have this chemical potential. Then the awkward constants
cancel out of every acceptance ratio and only `f`, `V`, `T` and `N` remain.

### 1.1 Acceptance ratios

With translation, rotation and reinsertion unchanged from Milestone B, two moves are new. For
insertion of one molecule, taking the system from N to N+1:

    A_ins = min(1, (f V / ((N+1) kT)) · exp(−ΔU/kT))

and for deletion, from N to N−1:

    A_del = min(1, ((N kT) / (f V)) · exp(−ΔU/kT))

These are each other's inverse in the way detailed balance requires: the `V/(N+1)` and `N/V`
factors are the ratio of configurational volumes available to the added or removed particle, and
they are what stops the chain from drifting to an arbitrary loading. **Getting these factors
wrong produces a perfectly stable simulation that converges to the wrong loading**, which is why
§5's checks are built around the ideal-gas limit rather than around the energy.

`ΔU` for an insertion is exactly what Milestone A's Widom kernel computes — the energy of one new
molecule against the host and all present guests. For a deletion it is the negative of the
removed molecule's interaction with everything else. Both are already implemented.

### 1.2 Fugacity from an equation of state

At low pressure `f ≈ P`, but real gases deviate, and adsorption isotherms are routinely measured
to tens of bar where the difference matters. The Peng–Robinson equation of state gives the
fugacity coefficient `φ = f/P` from the critical temperature, critical pressure and acentric
factor. `Guest` already carries `tc`, `pc` and `omega` for exactly this purpose; they are read
and stored today and never used.

Peng–Robinson is a cubic in the compressibility Z. Where it has three real roots — below the
critical point, where liquid and vapor coexist — the implementation must say which it takes
rather than silently picking one.

**Correction, 2026-09-27**: an earlier draft of this section said the vapor root is the largest
real root, and that is wrong. The stable phase is the one of lowest Gibbs energy, equivalently
the lowest fugacity coefficient, and which root that is depends on where the state point sits
relative to the saturation pressure: below it the vapor root (largest) wins, above it the liquid
root (smallest) does. Always taking the largest would silently return a metastable vapor above
the saturation pressure. kUPS selects by minimum fugacity coefficient, as does RASPA2, and this
was confirmed numerically — at CO2 243 K / 1.568 MPa and methane 114.3 K / 919.8 kPa, both
three-root points, kUPS takes the *smallest* root. Our implementation uses the same rule.

## 2. What is genuinely hard: a variable particle count

Everything else in this milestone is arithmetic we already have. The difficulty is that GPU
kernels need fixed shapes, and N changes every few moves.

The approach: fix a **capacity** per system at batch construction (kUPS calls this
`max_num_adsorbates`), store guests in a fixed-size array, and carry an occupancy count. Slots
beyond the count hold whatever they last held and are skipped by every loop. Insertion writes at
index N+1 and increments; deletion moves the last occupant into the removed slot and decrements,
which keeps occupancy contiguous without shifting an array.

Three consequences to get right:

- **Every energy loop is bounded by the capacity, not by N.** A loop to a runtime N is fine on a
  CPU and awkward on a GPU, where divergence between chains at different loadings costs. The
  first implementation loops to N and we measure the divergence before optimizing it away.
- **The running structure factor changes by exactly ±S_guest(k)** on insertion or deletion, which
  is the same incremental machinery Milestone B already uses and audits.
- **Exceeding capacity must be a loud failure**, not a silently rejected move. A chain that
  saturates its capacity samples a truncated distribution and still looks healthy. The audit must
  check occupancy against capacity and fail.

## 3. Isotherms map onto the batch axis

This is where the architecture pays off. An isotherm is a sweep over pressure, and each pressure
is an independent system. Since framework deduplication now stores one copy of a host no matter
how many systems reference it, **a fifty-point isotherm on one framework costs one framework's
memory** and runs as one batch.

The intended shape is a batch indexed by (framework, pressure, replica): several materials,
several pressures each, several independent chains per point for error bars, all advancing in
lockstep. For comparison, kUPS runs out of memory at 32 systems for a 50-guest NVT case.

## 4. Output

Per system: mean loading with its block-averaged error, mean energy, and the **isosteric heat of
adsorption from fluctuations**,

    q_st = kT − (⟨UN⟩ − ⟨U⟩⟨N⟩) / (⟨N²⟩ − ⟨N⟩²)

This is the finite-loading generalization of the zero-loading `q_st` Milestone A computes by the
delta method. It is a ratio of fluctuations, so it converges much more slowly than the loading
does and needs its error propagated properly rather than reported bare.

Per framework: loading against pressure, which is the isotherm.

## 5. Validation — and why the usual checks are not enough

A GCMC chain can be wrong in a way that no energy check detects, because the energies are right
and only the *acceptance ratio* is wrong. So the ladder is built around the particle number.

1. **The ideal-gas limit.** With the guest–host and guest–guest interactions switched off, the
   loading must reproduce the ideal gas law, `⟨N⟩ = fV/kT`, exactly and with the right
   fluctuations (`⟨N²⟩ − ⟨N⟩² = ⟨N⟩` for a Poisson distribution). This tests the combinatorial
   factors in isolation from the physics, and it is the single most valuable check here.
2. **Detailed balance between N and N+1.** On a small system, the ratio of time spent at each
   loading must match the analytic ratio. This catches an insertion and deletion pair that are
   individually plausible but not each other's inverse.
3. **The Milestone B limit.** With insertion and deletion disabled, the chain must reproduce
   Milestone B exactly for a fixed seed.
4. **Henry's law.** At low pressure the isotherm must be linear with slope equal to the Henry
   coefficient Milestone A computes independently. Two different methods, one number.
5. **kUPS GCMC** on the same case, comparing loading and energy in combined standard errors.
6. **Published RASPA isotherm**, methane in IRMOF-1 — an external check against neither our own
   code nor kUPS.

Checks 1 and 4 are the ones I would keep if forced to choose: they test the new physics against
closed-form answers rather than against another implementation.

## 6. Requirements checklist

- ☐ Peng–Robinson fugacity from `tc`, `pc`, `omega`; explicit, documented behavior when the cubic
      has three real roots.
- ☐ Insertion and deletion moves with the μVT acceptance ratios above, each verifiably the
      other's inverse.
- ☐ Fixed capacity per system with an occupancy count; deletion by swapping the last occupant
      into the freed slot.
- ☐ Exceeding capacity fails loudly; the audit checks occupancy against capacity.
- ☐ Structure factor updated by ±S_guest(k) on insertion and deletion, covered by the existing
      audit.
- ☐ Batch indexed by (framework, pressure, replica); an isotherm is one batch.
- ☐ Loading, energy and fluctuation `q_st`, each with a block-averaged error over cycles.
- ☐ The ideal-gas limit reproduces `⟨N⟩ = fV/kT` with Poisson fluctuations.
- ☐ Detailed balance between adjacent loadings on a small system.
- ☐ Disabling insertion and deletion reproduces Milestone B exactly for a fixed seed.
- ☐ The low-pressure slope matches Milestone A's Henry coefficient.
- ☐ Agreement with kUPS GCMC and with the published RASPA IRMOF-1 methane isotherm.
- ☐ Generic over Float32/Float64; CPU, CUDA, ROCm, Metal (Float32 only).
- ☐ Allocation-free and type-stable kernels under the StrictMode audit.
- ☐ Throughput measured against kUPS GCMC, warm, with the chain-count sweep.

## 7. Open questions

- Whether divergence between chains at different loadings costs enough to justify padding every
  loop to capacity. Measure before deciding.
- How to choose the capacity. Too small truncates the distribution; too large wastes memory and
  widens every loop. kUPS makes it a user input; a diagnostic that reports the maximum loading
  reached against the capacity would be better.
- Whether the fluctuation `q_st` converges well enough to be worth reporting by default, or
  should be opt-in with a warning about its error.
- Whether IRMOF-1 needs a CIF we do not have, and whether our P1-only reader can take it.
