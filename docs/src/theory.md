# Theory

## Widom test-particle insertion

For a rigid guest molecule inserted at a random pose (position and orientation) into a fixed
host structure at temperature ``T``, the insertion energy ``\Delta U`` gives the Boltzmann
insertion weight

```math
W = \exp(-\Delta U / k_B T).
```

Averaged over insertions at fixed ``V``, ``T``:

```math
\mu_{\mathrm{ex}} = -k_B T \, \ln \langle W \rangle
```

```math
K_H = \frac{V \langle W \rangle}{k_B T} \qquad (\mathrm{\mathring{A}}^3/\mathrm{eV})
```

```math
q_{\mathrm{st}} = k_B T - \frac{\langle \Delta U \, W \rangle}{\langle W \rangle} \qquad (\mathrm{eV})
```

``\mu_{\mathrm{ex}}`` is the excess chemical potential, ``K_H`` is Henry's constant, and
``q_{\mathrm{st}}`` is kUPS's `heat_of_adsorption` — the negative of the isosteric heat of
adsorption at zero loading.

`insertion_energy` computes ``\Delta U`` in `FrameworkBatch`'s own float type, but `widom`
accumulates ``W`` and ``\Delta U\, W`` on the host in Float64 regardless of that type, since
`exp(-\Delta U/k_B T)` overflows past a well only about 88.7 ``k_B T`` deep in Float32 — well
within the depths a real binding site reaches. `widom` converts the four block-averaged
quantities above to `FrameworkBatch`'s float type only at the end.

## Units

Energies are in eV, lengths in Å, temperature in K, and charges in units of the elementary
charge ``e``, matching kUPS so energies compare directly. The Boltzmann constant `KB` (eV/K)
and the Coulomb prefactor `KE` ``= 1/(4\pi\varepsilon_0)`` (eV·Å/e²) use the CODATA 2014 values
kUPS takes from ASE's units table.

## Insertion energy

The insertion energy of one guest pose decomposes into a Lennard-Jones term and an Ewald
Coulomb term, plus pose-independent constants collected once per framework:

```math
\Delta U = E_{\mathrm{LJ}} + E_{\mathrm{Coul}}
```

### Cell list for hard-core rejection

`insertion_energy` computes the Lennard-Jones and real-space Ewald sums below with a plain
linear loop over every host atom in the system, the fastest form measured for a framework this
size once the hard-core rejection stage (below) has already screened out most poses. The cell
list `FrameworkBatch` builds — a grid of ``n_i = \max(1, \lfloor L_i / w \rfloor)`` cells along
each of the stored cell's three perpendicular lengths ``L_i`` (target width ``w``, the
`cellwidth` keyword), atoms sorted so a cell is a contiguous array range — serves only the
rejection stage's phase-0 kernel, which needs just the few atoms within its own much shorter
reach.

One minimum image of ``\mathbf{r}`` (an insertion's reference point) is taken to each host atom,
and every guest site's own (already rotated) offset is added to that single image directly,
without a further minimum image. This is exact — every relevant pair's true separation stays
under half the cell's perpendicular length — because `FrameworkBatch` requires
`min_multiplicity(cell, r_c + r_guest) == (1,1,1)` at construction (``r_c`` the larger of the LJ
and Ewald cutoffs, ``r_{\mathrm{guest}}`` the guest's largest site distance from its reference
point).

### Lennard-Jones

Every guest site interacts with every host atom within the LJ cutoff ``r_c``, under minimum
image in the (triclinic) periodic cell, with plain truncation (no shifting):

```math
E_{\mathrm{LJ}} = \sum_{\text{guest site } s} \sum_{\substack{\text{host atom } j \\ r_{sj} < r_c}} 4\varepsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{sj}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{sj}}\right)^{6}\right]
```

with Lorentz–Berthelot mixing between LJ types ``i`` and ``j``:

```math
\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \qquad \varepsilon_{ij} = \sqrt{\varepsilon_i \varepsilon_j}.
```

**Analytic tail correction.** For a truncated LJ potential, the long-range contribution beyond
``r_c`` is added as a global correction that depends only on the particle counts per type and
the cell volume ``V``. The pairwise coefficient

```math
c_{ij} = \varepsilon_{ij}\sigma_{ij}^3\left[\frac{1}{3}\left(\frac{\sigma_{ij}}{r_c}\right)^9 - \left(\frac{\sigma_{ij}}{r_c}\right)^3\right]
```

is the standard radial integral of the LJ tail. Inserting a guest with `guest_counts` sites
per type into a system holding `counts` host particles per type changes the global correction
by

```math
\Delta E_{\mathrm{tail}} = \frac{8\pi}{3V}\sum_{i,j}\Bigl[2\,n_i\, g_j + g_i\, g_j\Bigr]\, c_{ij},
```

where ``n_i`` are the host counts and ``g_i`` the guest counts per LJ type; this is included
whenever the force field's `tail_correction` is enabled.

### Ewald summation

The Coulomb energy of the periodic system is split as

```math
E_{\mathrm{Coul}} = E_{\mathrm{real}} + E_{\mathrm{recip}} + E_{\mathrm{self}} + E_{\mathrm{excl}} + E_{\mathrm{net}}
```

**Splitting parameter and reciprocal cutoff.** Given a real-space cutoff ``r_c`` and a target
relative precision ``\varepsilon``, the splitting parameter ``\alpha`` solves

```math
\operatorname{erfc}(\alpha r_c) = \frac{r_c\,\varepsilon}{2}
```

by bisection, and the reciprocal-space cutoff follows as

```math
k_{\max} = 2\alpha\sqrt{-\ln(\varepsilon/2)}.
```

This is the same construction kUPS uses when the real-space cutoff is fixed, so both codes
select the same ``\alpha`` and ``k_{\max}`` from the same inputs.

**Real space.** For every pair of charges in different molecules, screened by
``\operatorname{erfc}``, inside the (generally different) Ewald cutoff and under minimum
image:

```math
E_{\mathrm{real}} = k_e \sum_{\substack{i<j \\ \text{different molecules} \\ r_{ij} < r_c^{\mathrm{Ew}}}} q_i q_j \frac{\operatorname{erfc}(\alpha r_{ij})}{r_{ij}}
```

``\operatorname{erfc}`` is evaluated by `erfc_dev`, a Chebyshev-series fit (28 terms, following
Numerical Recipes §6.2.2) valid for every ``z \geq 0`` and expressed without throwing branches,
so it compiles on every KernelAbstractions backend; `SpecialFunctions.erfc` is not
GPU-compilable. `insertion_energy`'s real-space pair loop instead calls `pair_erfc_dev`, a
Chebyshev series of the same construction fitted only over ``z = \alpha r \in [0, 4]`` — the
range this loop ever evaluates, since it only reaches pairs with ``r < r_c^{\mathrm{Ew}}`` and
`FrameworkBatch` rejects any batch whose ``\alpha \cdot r_c^{\mathrm{Ew}}`` would exceed that
bound. The narrower range needs far fewer terms (17 in Float64, 8 in Float32) for the same or
better accuracy: measured maximum relative error against `SpecialFunctions.erfc` over
``[0, 4]`` is 9.5e-15 in Float64 and 1.51e-6 in Float32, against `erfc_dev`'s own 3.7e-15 and
1.8e-6 on the same range. Every other use of ``\operatorname{erfc}`` (the reciprocal-space
self/exclusion terms, `ewald_energy`, and the test oracle `insertion_energy_reference`) keeps
`erfc_dev`.

**Reciprocal space.** Reciprocal vectors are enumerated over a half-space: only vectors with
``n_1 \geq 0`` (the integer coordinate along the first reciprocal lattice vector) are built,
each weighted ``w_k = 2`` to stand in for its mirror image ``-k`` (``w_k = 1`` when
``n_1 = 0``, since that plane already enumerates both signs of ``n_2, n_3`` explicitly). The
per-``k`` prefactor is

```math
p(k) = \frac{2\pi}{V}\frac{\exp(-k^2/4\alpha^2)}{k^2},
```

and the reciprocal-space energy uses the total structure factor
``S(k) = \sum_j q_j\, e^{i\,k\cdot r_j}``:

```math
E_{\mathrm{recip}} = \sum_{k} w_k\, p(k)\, |S(k)|^2.
```

The host's structure factor `Shost` is precomputed once per framework (it does not depend on
the guest pose); a guest insertion recomputes only its own, much smaller, structure factor and
combines it with `Shost` via ``2\,\mathrm{Re}(\overline{S_{\mathrm{host}}}\,S_{\mathrm{guest}}) + |S_{\mathrm{guest}}|^2``.

**Coupled k-vectors.** A supercell built by replicating a cell by `(n_1, n_2, n_3)` repeats the
host's fractional positions identically in every copy, so its structure factor is nonzero only
at k-vectors whose integer reciprocal-lattice coefficients ``(m_1, m_2, m_3)`` are each
divisible by the corresponding replication factor — every other k-vector's phase does not
repeat between copies and its host contribution cancels. `FrameworkBatch` keeps only this
coupled subset in `ks`, `kprefactor` and `Shost`: for CO2 in RUBTAK 3×3×3 that is 190 of the
4587 k-vectors the unreplicated cell's `kvectors` enumerates, cutting the reciprocal-space
table to about a third of its size.

**Guest self term.** The remaining term of ``E_{\mathrm{recip}}``, the guest's own
``|S_{\mathrm{guest}}(k)|^2`` summed over the FULL k-vector set, depends only on the guest's
orientation (not its position or the host), so `FrameworkBatch` evaluates it at 64 fixed
orientations and folds their mean into `constant_offset`, storing half the spread of those 64
samples as `self_term_halfrange`. That half-range is an estimate from a finite sample, not a
bound on the true continuous-orientation range: a continuous orientation can reach roughly
1.3 times `self_term_halfrange` away from the mean. For CO2 in RUBTAK 3×3×3 the half-range is
about 2.4e-7 eV, against a mean self term of about 0.0052 eV — the orientation dependence is
negligible next to the mean either way, which is why replacing the per-insertion sum with its
average changes `widom`'s results by only a few times `self_term_halfrange`. `FrameworkBatch`
throws instead of silently using this approximation when `2 · self_term_halfrange` exceeds
``10^{-3}\,k_B\cdot 300\,\mathrm{K}``.

**Reciprocal cutoff and k-vector bound.** Reciprocal vectors are kept while ``|k| \leq
k_{\max}``. The search range along each reciprocal-lattice direction is bounded using the
corresponding *direct* lattice vector's own length ``|a_i|``,

```math
|n_i| \leq \frac{k_{\max}\,|a_i|}{2\pi},
```

not the (shorter) perpendicular length between opposite cell faces: for a triclinic cell the
perpendicular-length bound can miss vectors that are within ``k_{\max}`` but lie off-axis
relative to the perpendicular direction.

**Self energy.**

```math
E_{\mathrm{self}} = -\frac{\alpha}{\sqrt{\pi}}\sum_i q_i^2
```

**Net-charge correction**, for a system with total charge ``Q = \sum_i q_i``:

```math
E_{\mathrm{net}} = -\frac{\pi}{2 V \alpha^2}\, Q^2
```

**Intramolecular exclusion.** The reciprocal-space sum has no notion of molecules: it always
includes the full ``\operatorname{erf}(\alpha r)/r`` part of every pair's interaction,
intramolecular pairs included, since it is built from the total structure factor. The
real-space sum already excludes same-molecule pairs (the sum above runs over pairs in
*different* molecules only), so it never contributes their ``\operatorname{erfc}(\alpha r)/r``
part. Removing the leftover ``\operatorname{erf}`` part for each intramolecular pair —
using ``1 - \operatorname{erfc}(\alpha r) = \operatorname{erf}(\alpha r)`` and the direct
(non-periodic) distance between the two sites — leaves the pair's Coulomb interaction out of
the total entirely:

```math
E_{\mathrm{excl}} = -k_e \sum_{\substack{i<j \\ \text{same molecule}}} q_i q_j\, \frac{1 - \operatorname{erfc}(\alpha r_{ij})}{r_{ij}}.
```

For a rigid guest this term is pose-independent and is folded into the per-framework constant
alongside the guest self-energy and the net-charge correction.

## Hard-core rejection

Most random insertions land too close to a host atom to matter: for CO2 in RUBTAK 3×3×3 at
298.15 K, 68% (Float64) / 78% (Float32) of insertions have a Boltzmann weight that underflows to
exactly `0.0`. `widom` decides this for a large fraction of insertions — 41.3% (Float64) / 41.4%
(Float32), measured on this system — from a rigorous lower bound on the insertion energy, before
ever evaluating `insertion_energy`.

**Underflow point.** `theta_F(T)` is the smallest value ``\theta`` for which `exp(-θ)` underflows
to exactly zero in float type ``T`` (about 745.13 for `Float64`, 103.97 for `Float32`, found by
bisection on the float grid rather than assumed). Since `exp` is monotone non-increasing,
``\Delta U / k_B T > \theta_F(T)`` guarantees `exp(-ΔU/kT) == 0`. `widom`'s rejection rule always
uses ``\theta = \theta_F(\mathrm{Float64})``, regardless of `FrameworkBatch`'s own float type,
since a flagged pose must be provably zero weight in the Float64 arithmetic `widom` actually
accumulates weights in (see above), not in a possibly narrower `theta_F(F)`.

**Lower bound.** Write the insertion energy as a sum over guest-site/host-atom pairs plus the
reciprocal-space cross term and the pose-independent constant ``c_s``:

```math
\Delta U = \sum_{a,h} u_{ah}(r_{ah}) + U_{\mathrm{recip}} + c_s, \qquad
u_{ah}(r) = \mathrm{LJ}_{ah}(r)\,[r < r_{\mathrm{lj}}] +
K_{ah}\,\frac{\operatorname{pair\_erfc\_dev}(\alpha r)}{r}\,[r < r_{\mathrm{ew}}],
\qquad K_{ah} = k_e\, q_a q_h,
```

with ``r_{\mathrm{lj}}`` and ``r_{\mathrm{ew}}`` the Lennard-Jones and Ewald cutoffs and
``[\,\cdot\,]`` the indicator (1 when the bracketed condition holds, 0 otherwise). For every
pair, the worst this term can be anywhere on its domain is bounded: ``-\varepsilon_{ah}`` when
``K_{ah} \geq 0`` (the Coulomb term is non-negative, the Lennard-Jones term is bounded below by
``-\varepsilon_{ah}``); when ``K_{ah} < 0``, at the radius ``r_0`` where the untruncated
``\mathrm{LJ}_{ah}(r) + K_{ah}\,\operatorname{pair\_erfc\_dev}(\alpha r)/r`` crosses zero below
its own minimum, clamped to ``r_{\mathrm{lj}}`` (the crossing can lie beyond ``r_{\mathrm{lj}}``,
past which ``u_{ah}`` no longer includes the Lennard-Jones term at all), the bound is
``-\varepsilon_{ah} - |K_{ah}|\operatorname{pair\_erfc\_dev}(\alpha r_0)/r_0`` (the Coulomb
term's magnitude is largest, for ``r \geq r_0``, at ``r_0`` itself). Summing these per-pair
magnitudes over every pair in a system, plus a reciprocal-space bound
``R_s = k_e \sum_a |q_a| \sum_k 2\, \mathrm{kprefactor}_k\, |S_{\mathrm{host},k}|`` (from
``|\mathrm{Re}(\overline{S_{\mathrm{host}}}\,S_g)| \leq |S_{\mathrm{host}}||S_g| \leq
|S_{\mathrm{host}}|\sum_a|q_a|``), gives ``B_s``: for CO2 in RUBTAK 3×3×3, ``B_s \approx 127{,}118
\, k_B T`` at 298.15 K, computed once per system at `FrameworkBatch` construction since it does
not depend on temperature. ``B_s = \infty`` when some pair combines an attractive Coulomb term
with no Lennard-Jones well at all (``\varepsilon_{ah} = 0``, ``K_{ah} < 0``): no finite bound
exists, and rejection is disabled for that system.

**Rejection radius.** For one guest site `a` and host type `t`, `K_min(a,t)` is the most negative
``K_{ah}`` over that system's atoms of type `t` (zero if none is negative) — temperature
independent, so it is also computed once at construction. `widom` combines it with the
temperature-dependent margin ``(\theta + 2)\,k_B T + \mathrm{safety} + B_s - c_s`` (``\theta =
\theta_F(\mathrm{Float64})``, per "Underflow point" above) into a rejection radius ``\rho_{at}``,
clamped to ``r_{\mathrm{lj}}`` for the same reason as ``r_0`` above: the first root, scanning up
from ``r \to 0``, of

```math
\mathrm{LJ}_{at}(r) - \frac{|K_{\min}(a,t)|}{r} = (\theta + 2)\,k_B T + \mathrm{safety} + B_s - c_s.
```

The safety term ``\mathrm{safety} = 2 n \,\mathrm{eps}(F) (B_s + |c_s|) + 4\times10^{-6} B_s``
(``F`` `FrameworkBatch`'s own float type, since `insertion_energy` sums on the device in ``F``)
bounds the gap between this ideal, exact-formula ``\Delta U`` and the actual floating-point value
`insertion_energy` computes. Recursive summation of ``n`` terms of magnitude at most
``B_s + |c_s|`` has rounding error at most ``(n-1)\,u\,\Sigma|x_i|`` with unit roundoff
``u = \mathrm{eps}(F)/2``, so ``2 n\,\mathrm{eps}(F)(B_s+|c_s|)`` covers it with margin;
``n = N_{\mathrm{sites}}\cdot\mathrm{natoms}_s + nk_s + 8`` counts the real-space pair terms, the
reciprocal-space terms, and a handful of pose-independent additions. The ``4\times10^{-6} B_s``
term additionally covers `pair_erfc_dev`'s own approximation error relative to the true
``\operatorname{erfc}`` (measured up to `1.51e-6` in Float32 over `[0, PAIR_ERFC_XMAX]`), applied
once per Coulomb term and so likewise proportional to the sum's magnitude. Both terms ensure that
a rejected insertion's true (bound) energy still clears ``(\theta+2)\,k_B T`` once this error is
subtracted back out. Every separation under ``\rho_{at}`` then satisfies the rejection condition
for any atom of type `t`, since the left-hand side lower bounds that atom's true pair energy at
distance `r`. For CO2 in RUBTAK 3×3×3, ``\rho_{at}`` ranges from about 0.92 to 1.20 Å across the
system's compact types — a small fraction of the Lennard-Jones ``\sigma``, consistent with these
radii marking the steep repulsive wall rather than the interaction range itself.

**Two-phase evaluation.** `widom` launches a phase-0 kernel that flags every insertion whose
guest sites all stay outside ``\rho_{at}`` of every host atom of the matching type, scanning a
short-reach cell-list stencil (typically 27 cells) rather than every atom. A rejected insertion's
weight and energy-weighted product are recorded as exactly zero without ever calling
`insertion_energy`; a phase-1 kernel computes the energy for every surviving insertion. Because
the radius is constructed to guarantee `exp(-ΔU/k_B T) == 0.0` for every insertion phase 0 flags,
this changes no result: `widom`'s output is identical to computing every insertion's energy
directly.

## Pose sampling

Each insertion samples an independent pose:

- **Position**: a fractional coordinate uniform on ``[0,1)^3``, transformed to Cartesian by
  the cell matrix — uniform over the periodic cell.
- **Orientation**: a uniformly-random unit quaternion via Shoemake's (1992) method. With
  ``u_1, u_2, u_3 \sim \mathrm{Uniform}(0,1)``, kUPS builds, in its own scalar-first
  ``(w,x,y,z)`` layout,

  ```math
  \bigl(\sqrt{1-u_1}\sin 2\pi u_2,\ \sqrt{1-u_1}\cos 2\pi u_2,\ \sqrt{u_1}\sin 2\pi u_3,\ \sqrt{u_1}\cos 2\pi u_3\bigr).
  ```

  PureAdsorb stores quaternions vector-part-first, ``(x,y,z,w)``, and builds the same
  distribution by permuting components into

  ```math
  \bigl(\sqrt{1-u_1}\cos 2\pi u_2,\ \sqrt{u_1}\sin 2\pi u_3,\ \sqrt{u_1}\cos 2\pi u_3,\ \sqrt{1-u_1}\sin 2\pi u_2\bigr).
  ```

  A guest site at ``v`` in the molecular frame is rotated by the expanded Rodrigues form
  ``v + 2\,u \times (u \times v + w\,v)``, where ``u = (q_x, q_y, q_z)`` and ``w = q_w`` —
  equivalent to the quaternion sandwich product ``q\,v\,q^{-1}`` without building a rotation
  matrix.

Insertion ``g`` of the global sequence ``1{:}ninsert`` is assigned to system
``\mathrm{mod1}((g-1) \div \mathrm{run} + 1,\ nsys)``: ``\mathrm{run}`` consecutive insertions
go to the same system before the assignment cycles to the next one. GPU work-items are indexed
by their position in one kernel-launch chunk, which follows the global sequence directly, so
work-items adjacent on the device then read the same framework's tables instead of a different
one per work-item.

## Block-averaged statistics and the delta method

Each system's ``n_s`` insertions accumulate into `nblocks` blocks of (as close to equal as
possible) size, in that system's own sample order — independent of how many insertions any
other system in the batch receives. Per block ``b``, `widom` sums ``\sum W`` and
``\sum \Delta U\,W`` and counts samples ``n_b``, giving per-block means ``\overline{W}_b`` and
``\overline{UW}_b``. The overall estimates are the
pooled ratios ``W = \sum_b \sum W_b / \sum_b n_b`` and ``UW`` likewise (not the average of the
per-block means), with block variances

```math
\operatorname{var}(W) = \frac{1}{n_b^{\mathrm{blk}}-1}\sum_b (\overline{W}_b - W)^2, \qquad
\operatorname{var}(UW),\ \operatorname{cov}(W, UW)\ \text{analogously.}
```

``\mu_{\mathrm{ex}}`` and ``K_H`` are linear (up to a log) in ``W``, so their standard errors
follow directly from ``\operatorname{sem}(W) = \sqrt{\operatorname{var}(W)/n_b^{\mathrm{blk}}}``.
``q_{\mathrm{st}}`` depends on the ratio ``UW/W`` of two correlated block-averaged quantities;
its variance is estimated by the first-order delta method for a ratio of means,

```math
\operatorname{var}\!\left(\frac{UW}{W}\right) \approx \frac{1}{n_b^{\mathrm{blk}}}\left(\frac{UW}{W}\right)^2 \left[\frac{\operatorname{var}(UW)}{UW^2} + \frac{\operatorname{var}(W)}{W^2} - \frac{2\operatorname{cov}(W,UW)}{UW\cdot W}\right],
```

clamped at zero (the first-order approximation can return a negative value when the covariance
term dominates, but the true variance is non-negative).
