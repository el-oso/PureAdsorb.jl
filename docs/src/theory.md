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
within the depths a real binding site reaches. `WidomResult`'s fields — the four block-averaged
quantities above and their standard errors — stay Float64 regardless of `FrameworkBatch`'s own
float type, since ``K_H`` scales with ``\exp(\text{well depth}/k_BT)`` and can legitimately
exceed a smaller float type's range for a strongly binding site.

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

## Canonical (NVT) Monte Carlo

Widom insertion (above) samples a single, fixed configuration: one host, no guests, no chain.
With ``N`` guests present the configuration itself must be sampled, at fixed ``N``, ``V`` and
``T``, before Widom insertion can be applied to it. This section derives the Markov chain that
does the sampling and the extra energy terms ``N`` mobile guests introduce.

### The canonical ensemble and the Metropolis criterion

The equilibrium distribution over configurations ``x`` at fixed ``N``, ``V``, ``T`` is
``P(x) \propto \exp(-U(x)/k_BT)``. A Markov chain built from a proposal density
``q(x\to x')`` and an acceptance probability ``A(x\to x')`` has ``P`` as its stationary
distribution once it satisfies detailed balance,

```math
P(x)\,q(x\to x')\,A(x\to x') = P(x')\,q(x'\to x)\,A(x'\to x).
```

For a **symmetric** proposal, ``q(x\to x') = q(x'\to x)``, the ``q`` factors cancel and detailed
balance reduces to a condition on the acceptance ratio alone,

```math
\frac{A(x\to x')}{A(x'\to x)} = \frac{P(x')}{P(x)} = \exp(-\Delta U/k_BT), \qquad \Delta U = U(x') - U(x).
```

The Metropolis choice ``A(x\to x') = \min(1, \exp(-\Delta U/k_BT))`` satisfies this for every
``\Delta U``: when ``\Delta U \leq 0``, ``A(x\to x') = 1`` and ``A(x'\to x) = \exp(\Delta U/k_BT)``,
giving ratio ``\exp(-\Delta U/k_BT)`` as required; the case ``\Delta U \geq 0`` gives the same
ratio by the same substitution with the roles exchanged. Only ``\Delta U`` ever enters this test,
so a chain never needs the total energy to decide a move — except for the periodic audit below,
which recomputes it as an independent check.

Drawing one uniform ``u \sim \mathrm{Uniform}(0,1)`` per chain and accepting when
``-\Delta U/k_BT > \ln u`` implements exactly this rule: ``P(u < \exp(-\Delta U/k_BT))`` equals
``\exp(-\Delta U/k_BT)`` when that value is below 1, and equals 1 (since ``u < 1`` always holds)
when it is at or above 1 — precisely ``\min(1, \exp(-\Delta U/k_BT))``.

### The three moves, and why each is symmetric

Each move attempt picks one guest, uniformly at random within its system, and applies one of
three proposals to it:

- **Translation.** ``\mathbf{r}' = \mathbf{r} + \delta\,\boldsymbol{\eta}``,
  ``\boldsymbol{\eta} \sim \mathcal{N}(0, I_3)`` — an isotropic Gaussian displacement of the
  guest's reference point, standard deviation ``\delta`` (Å) along each Cartesian axis. A Gaussian
  density is an even function of the displacement, ``q(\Delta\mathbf{r}) = q(-\Delta\mathbf{r})``,
  so ``q(\mathbf{r}\to\mathbf{r}') = q(\mathbf{r}'\to\mathbf{r})``: symmetric.
- **Rotation.** A fresh unit quaternion, uniform on ``SO(3)`` by the same Shoemake construction
  as the pose sampler above, is raised to a fractional power ``\rho \in [0,1]`` — same rotation
  axis, angle scaled by ``\rho`` — and composed with the guest's current orientation. On ``SO(3)``'s
  invariant measure the rotation axis is uniform on the sphere and independent of the angle, so a
  rotation by angle ``\theta`` about axis ``\mathbf{u}`` has the same density as its inverse
  (angle ``-\theta`` about ``\mathbf{u}``, equivalently angle ``\theta`` about ``-\mathbf{u}``,
  which the uniform axis distribution assigns equal density): symmetric.
- **Reinsertion.** A fresh position uniform over the cell and a fresh orientation uniform on
  ``SO(3)``, independent of the guest's current pose. The proposal density does not depend on the
  starting configuration at all, so ``q(\mathrm{old}\to\mathrm{new}) = q(\mathrm{new}\to\mathrm{old})``
  trivially: symmetric.

All three need no Hastings correction in the acceptance ratio above. Every chain in a batch
performs the *same* move type on a given step — a fixed, not randomly re-chosen per chain,
schedule, which avoids divergent branches across a batch running in lockstep. This changes
nothing about the target distribution: each move type individually satisfies detailed balance
for ``P``, and a chain built from any fixed sequence of individually detailed-balanced kernels
still has ``P`` as its stationary distribution, since each step alone preserves it. What a fixed
(rather than randomized) order forfeits is reversibility of the *composite* per-cycle transition,
not the stationary distribution itself.

### Guest–guest energy

With ``N`` guests present, the total potential energy decomposes as

```math
U = U_{\text{host-host}} + \sum_i U_{\text{host-guest}}(i) + \sum_{i<j} U_{\text{guest-guest}}(i,j).
```

``U_{\text{host-host}}`` is constant (the host never moves) and cancels in every ``\Delta U``.
``U_{\text{host-guest}}`` is exactly the insertion energy above, evaluated per guest.
``U_{\text{guest-guest}}`` is new: a Lennard-Jones sum over guest site pairs under the same
Lorentz–Berthelot mixing and plain truncation as the host term, plus a Coulomb term folded into
the same Ewald summation as the host, using the same splitting parameter ``\alpha`` and k-vector
set, since guest and host charges sit in the same periodic cell.

### The running structure factor, and why a move costs ``O(n_k)``

The structure factor of the whole system is the host's plus every guest's own,

```math
S(k) = S_{\mathrm{host}}(k) + \sum_i S_i(k), \qquad S_i(k) = \sum_{s\,\in\,\text{guest }i} q_s\,e^{i\,k\cdot r_s},
```

and the reciprocal-space energy of the whole system is ``U_{\mathrm{recip}} = \sum_k \mathrm{pref}_k\,|S(k)|^2``
(the same prefactor as the Ewald summation above, now summing the total charge distribution
rather than the host alone). Moving one guest ``i`` changes only that guest's own term,

```math
S(k) \;\to\; S(k) + \Delta S(k), \qquad \Delta S(k) = S_i^{\mathrm{new}}(k) - S_i^{\mathrm{old}}(k),
```

leaving every other guest's and the host's contribution untouched. Since
``|a+b|^2 = |a|^2 + 2\,\mathrm{Re}(\bar a b) + |b|^2`` for any complex ``a``, ``b``, the reciprocal
energy change from this one move is exactly

```math
\Delta U_{\mathrm{recip}} = \sum_k \mathrm{pref}_k \bigl(2\,\mathrm{Re}[\,\overline{S(k)}\,\Delta S(k)\,] + |\Delta S(k)|^2\bigr),
```

with ``S(k)`` the running total *before* the move — exact regardless of how many other guests or
host atoms feed into it, since they are untouched by this move. Evaluating ``\Delta S(k)`` needs
one pass over the moved guest's own sites at each ``k``, so this sum costs ``O(n_k)`` work per
move, independent of the total guest count ``N``; recomputing ``S(k)`` from scratch after every
move — summing every guest's sites at every ``k`` — would cost ``O(N\,n_k)`` instead. Carrying
``S(k)`` as mutable per-chain state, updated to ``S(k) + \Delta S(k)`` on acceptance and left
unchanged on rejection, is what makes the incremental form possible.

### Which k-vectors each term needs

"Coupled k-vectors" above shows that a supercell built by replicating a smaller cell has a host
structure factor that vanishes at every k-vector not coupled to that replication — about 190 of
4587 for RUBTAK 3×3×3. The host-guest cross term in ``U_{\mathrm{recip}}``,
``2\,\mathrm{Re}(\overline{S_{\mathrm{host}}}\,S_{\mathrm{guest}})``, carries an explicit factor
of ``S_{\mathrm{host}}(k)``, so it vanishes at exactly the same k-vectors ``S_{\mathrm{host}}``
does, whatever the guest term's own value there — restricting that term to the
replication-coupled subset changes nothing about it. The guest-guest term,
``|\sum_i S_i(k)|^2``, carries no factor of ``S_{\mathrm{host}}`` at all: a guest's position is
not tied to the host's periodicity, so its structure factor is generically nonzero at every
``k``, and the guest-guest sum needs the full k-vector table to be evaluated correctly.
PureAdsorb currently builds and uses one full k-vector table for every reciprocal-space term once
any guest is present (`fullk = true` at `FrameworkBatch` construction); restricting the cross
term back to the replication-coupled subset is a valid arithmetic identity, not yet exploited as
an optimization.

### Intramolecular exclusion: two conventions, one value

The reciprocal-space sum has no notion of molecules: built from the total structure factor, it
supplies the full ``\operatorname{erf}(\alpha r)/r`` part of every pair's Coulomb interaction,
intramolecular pairs included. What must be subtracted to remove an intramolecular pair's
contribution entirely therefore depends on a bookkeeping choice — whether that pair is also
included in the real-space sum.

PureAdsorb excludes same-guest pairs from the real-space sum (the "Real space" formula above runs
over pairs in different molecules only), so such a pair receives zero from real space and the
full ``\operatorname{erf}(\alpha r)/r`` from reciprocal space, which ``E_{\mathrm{excl}}`` (given
above) cancels exactly:

```math
\underbrace{0}_{\text{real}} + \underbrace{k_e\,q_sq_t\,\operatorname{erf}(\alpha r_{st})/r_{st}}_{\text{recip's share}} - \underbrace{k_e\,q_sq_t\,(1-\operatorname{erfc}(\alpha r_{st}))/r_{st}}_{E_{\mathrm{excl}}} = 0,
```

using ``\operatorname{erf}(x) = 1 - \operatorname{erfc}(x)``. kUPS makes the opposite bookkeeping
choice: its real-space sum includes intramolecular pairs (paying
``k_e\,q_sq_t\,\operatorname{erfc}(\alpha r_{st})/r_{st}`` for each), so real plus reciprocal
already sum to the pair's full, undamped Coulomb energy,
``\operatorname{erfc}(\alpha r)/r + \operatorname{erf}(\alpha r)/r = 1/r``; kUPS's own correction
then subtracts that full ``k_e\,q_sq_t/r_{st}`` rather than the ``\operatorname{erf}`` leftover
PureAdsorb subtracts. The two conventions cancel the same pair's contribution to the same value —
zero — not merely approximately: the identity ``\operatorname{erf} = 1 - \operatorname{erfc}``
makes them equal term by term, for every ``r`` inside the real-space cutoff (a condition
PureAdsorb asserts holds for every intramolecular distance in the guest, rather than assuming it).

### Block averaging runs over cycles, not insertions

Widom insertion along the chain evaluates several test-particle insertions into the
configuration the current cycle's moves produced, then advances the chain by another cycle before
the next batch of insertions. Two different kinds of correlation are present: insertions within
one cycle share the same background configuration but are otherwise independent draws of a test
pose, so at *fixed* configuration they are independent samples of the Boltzmann weight; the
configuration itself, however, is produced by a Markov chain, so consecutive cycles' mean
Boltzmann weights are correlated, with a correlation time set by how many *moves* it takes the
chain to decorrelate — not by how many insertions run per cycle.

A block-averaged standard error is only consistent if each block is long enough, and blocks are
separated far enough, that different block means are approximately independent. Blocking by
cycle satisfies this by construction, since each block spans whole cycles of the chain's own
correlated dynamics. Blocking by insertion instead — treating insertions from the *same* cycle as
if they were independent samples of the mean — would split highly correlated samples (all drawn
from one configuration) across different blocks and understate the true variance of the mean.
Widom insertion into a single, fixed, non-chained framework (Milestone A) has no such correlation
to worry about — there is only one configuration — which is why blocking by insertion is correct
there and not here.

### Parallelism: a batch of chains, not a batch of moves

Move ``n+1`` of a chain depends on the outcome of move ``n``, since its proposal and acceptance
test read the configuration and running structure factor move ``n`` left behind. No amount of
hardware removes that dependency: the moves of *one* chain must execute strictly in sequence. The
only axis along which independent work exists is between *different* chains — replicas of the
same framework, or different frameworks in a batch — since two chains share no state (each
carries its own poses and running ``S(k)``) and their moves may be proposed, evaluated and
accepted or rejected with no communication between them.

This is a different axis from the parallelism exploited *within* one move's own energy
evaluation — splitting a single move's host-atom, guest-guest and k-vector sums across the
work-items of a workgroup parallelizes the arithmetic of one step, which is embarrassingly
parallel regardless of the chain's sequential structure. Batching chains parallelizes across
*steps*, which the chain's own sequential dependency would otherwise forbid entirely. A
production run's throughput therefore comes from running as many independent chains as the
device can hold, each one still advancing a single move at a time.

## Grand canonical (μVT) Monte Carlo

Canonical Monte Carlo (above) samples configurations at a fixed guest count. An adsorption
isotherm asks a different question — how much gas a material holds at a given pressure — which
means letting the guest count itself fluctuate at fixed chemical potential.

### The grand canonical ensemble and why fugacity is the right variable

For ``N`` indistinguishable rigid guest molecules in volume ``V`` at temperature ``T``, the
canonical partition function factors into a translational part (the momentum integral, giving the
thermal wavelength ``\Lambda``) and a configurational part over positions and orientations:

```math
Q(N,V,T) = \frac{1}{N!\,\Lambda^{3N}}\,q_{\mathrm{rot}}^N\, Z(N,V,T), \qquad
Z(N,V,T) = \int_V d\mathbf{r}^N \int_{SO(3)} d\Omega^N\, \exp(-U(\mathbf{r},\Omega)/k_BT),
```

where ``q_{\mathrm{rot}}`` is the rotational partition function of ONE isolated molecule (the
``SO(3)`` integral with no interaction — see "the rigid-body orientational factor" below for what
this equals for a rigid body). The grand partition function at chemical potential ``\mu`` is

```math
\Xi(\mu,V,T) = \sum_N \exp(\beta\mu N)\,Q(N,V,T) = \sum_N \frac{z^N}{N!}\,Z(N,V,T), \qquad
z \equiv \exp(\beta\mu)\,q_{\mathrm{rot}}/\Lambda^3,
```

``z`` the absolute activity; the probability of exactly ``N`` guests at configuration
``(\mathbf{r},\Omega)`` is ``P(N;\mathbf{r},\Omega) \propto (z^N/N!)\exp(-U/k_BT)``.

Both ``\Lambda`` (a quantum-mechanical, momentum-space quantity) and ``q_{\mathrm{rot}}`` (awkward
to evaluate for an arbitrary rigid body) are inconvenient to carry through a simulation.
**Fugacity** sidesteps both: ``f`` is defined as the pressure an IDEAL gas of the same species
would need to have the same chemical potential ``\mu``. An ideal gas of rigid rotors has
``Q_{\mathrm{ideal}}(N,V,T) = (1/N!)(V/\Lambda^3\,q_{\mathrm{rot}})^N``, and the usual ideal-gas
thermodynamics (``PV=Nk_BT``, ``\mu_{\mathrm{ideal}} = -k_BT\,\partial \ln Q_{\mathrm{ideal}}/\partial N``)
gives ``\mu_{\mathrm{ideal}}(P,T) = k_BT\ln(P\Lambda^3/(k_BT\,q_{\mathrm{rot}}))``. Setting
``\mu = \mu_{\mathrm{ideal}}(f,T)`` — the definition of fugacity — and solving for ``z``:

```math
\exp(\beta\mu) = \frac{f\Lambda^3}{k_BT\,q_{\mathrm{rot}}} \quad\Longrightarrow\quad
z = \exp(\beta\mu)\,q_{\mathrm{rot}}/\Lambda^3 = \frac{f}{k_BT}.
```

``\Lambda`` and ``q_{\mathrm{rot}}`` have vanished: ``z`` is just ``f/k_BT``. This is why fugacity
— and only fugacity — can appear in an acceptance ratio without also carrying a thermal-wavelength
or rotational-partition-function factor alongside it: substituting ``z=f/k_BT`` into any ratio
built from ``\Xi`` replaces every occurrence of ``\Lambda`` and ``q_{\mathrm{rot}}`` by
``k_BT/f``, and `log_insertion_prefactor`/`log_deletion_prefactor` (`src/moves.jl`) take exactly
`f`, `V`, `kT` and nothing else — no such constant appears anywhere in the code because none
survives the substitution.

### Acceptance ratios and the combinatorial factors

Consider inserting one molecule at a position drawn uniformly in ``V`` (density ``1/V``) and an
orientation drawn uniformly on ``SO(3)`` (density ``1/\Omega_{\mathrm{tot}}``,
``\Omega_{\mathrm{tot}}`` the total measure of ``SO(3)``) — exactly the μVT exchange moves'
proposal. From ``P(N;\ldots)`` above, the ratio of the ``(N{+}1)``-particle density at the new
configuration to the ``N``-particle density at the old one is

```math
\frac{P(N{+}1;\mathbf r^N,\Omega^N,\mathbf r_{N+1},\Omega_{N+1})}{P(N;\mathbf r^N,\Omega^N)}
= \frac{z^{N+1}/(N{+}1)!}{z^N/N!}\,\exp(-\Delta U/k_BT) = \frac{z}{N+1}\exp(-\Delta U/k_BT).
```

Metropolis–Hastings for this move balances the forward transition — propose with density
``1/(V\Omega_{\mathrm{tot}})``, accept with probability ``A_{\mathrm{ins}}`` — against the
reverse, deletion, which proposes "remove the ``(N{+}1)``-th molecule" by picking uniformly among
the ``N{+}1`` molecules present (probability ``1/(N{+}1)``, no continuous density at all, since
deletion picks a discrete existing guest rather than drawing a pose). Detailed balance,

```math
P(N)\cdot\frac{1}{V\Omega_{\mathrm{tot}}}\cdot A_{\mathrm{ins}} = P(N{+}1)\cdot\frac{1}{N+1}\cdot A_{\mathrm{del}},
```

with the density ratio above and the Metropolis choice ``A(\cdot)=\min(1,\text{ratio})``, gives

```math
A_{\mathrm{ins}} = \min\!\left(1,\ \frac{zV}{N+1}\exp(-\Delta U/k_BT)\right)
= \min\!\left(1,\ \frac{fV}{(N+1)k_BT}\exp(-\Delta U/k_BT)\right)
```

after substituting ``z=f/k_BT`` — `log_insertion_prefactor(f,V,kT,N) = log(fV) - log(N+1) -
log(kT)` is exactly the log of this prefactor, `N` the occupancy BEFORE the insertion. The
identical argument for deletion (propose removing one of ``N`` present molecules uniformly, at
density ``1/N``; accept against re-inserting it at the same pose, density
``1/(V\Omega_{\mathrm{tot}})``) gives ``A_{\mathrm{del}} = \min(1, (Nk_BT)/(fV)\exp(-\Delta U/k_BT))``.

**These are exact inverses of each other in a precise sense: the SAME step, ``N \leftrightarrow
N{+}1``, has an insertion prefactor ``fV/((N{+}1)k_BT)`` and a deletion prefactor for undoing it —
deleting the guest that insertion just added — of ``(N{+}1)k_BT/(fV)``, the reciprocal.**
`log_deletion_prefactor` is defined in the source as `-log_insertion_prefactor(f,V,kT,N-1)`
rather than as a second, independently typed formula, which makes this identity a matter of
definition rather than something two separately hand-derived expressions might disagree about.
Physically, ``V/(N{+}1)`` is the fraction of the system's configurational volume that becomes
newly available to the ``(N{+}1)``-th molecule once it exists — equivalently, ``1/(N{+}1)`` is the
probability that the ``(N{+}1)!``-fold labeling ambiguity among indistinguishable molecules
happens to assign the new one to a particular slot — and ``N/V`` is the same ratio read backward,
from the perspective of the ``N`` molecules remaining after one is removed.

### Insertion and deletion are not individually detailed-balanced

Translation, rotation and reinsertion are each symmetric moves whose own forward and reverse
transitions are the SAME kind of move — a translation is undone by another translation. Insertion
has no such partner: **no insertion move takes ``N{+}1`` guests back to ``N``.** Formally, the
transition kernel ``K_{\mathrm{ins}}`` built from insertion attempts alone satisfies
``K_{\mathrm{ins}}(x'\to x) = 0`` whenever ``x'`` has one more guest than ``x`` (nothing in
"attempt an insertion" can remove a guest), while ``K_{\mathrm{ins}}(x\to x')`` is generically
nonzero. Detailed balance for ``K_{\mathrm{ins}}`` alone would require
``P(x)K_{\mathrm{ins}}(x,x') = P(x')K_{\mathrm{ins}}(x',x) = 0`` for every such pair — every
insertion attempt would have to be rejected. The same argument, insertion and deletion exchanged,
rules out ``K_{\mathrm{del}}`` alone. This is the opposite of the three canonical moves, each its
own reverse — and the reason a fixed sequence of canonical moves preserves the target distribution
while a fixed sequence of insertion/deletion attempts does not.

What DOES hold is a weaker, pairwise statement: by the reciprocal-prefactor construction above,

```math
P(x)\,K_{\mathrm{ins}}(x,x') = P(x')\,K_{\mathrm{del}}(x',x) \qquad \text{for every pair } (x,x')
\text{ differing by one guest,}
```

**provided insertion and deletion are attempted with equal probability.** The 50/50 mixture
``K = \tfrac12 K_{\mathrm{ins}} + \tfrac12 K_{\mathrm{del}}`` then satisfies ordinary detailed
balance term by term: for an insertion-type pair, ``K_{\mathrm{del}}(x,x')=0`` so
``P(x)K(x,x') = \tfrac12 P(x)K_{\mathrm{ins}}(x,x')``, and ``K_{\mathrm{ins}}(x',x)=0`` so
``P(x')K(x',x) = \tfrac12 P(x')K_{\mathrm{del}}(x',x)`` — equal, by the identity above. A biased or
scheduled alternation breaks this: attempting insertion and deletion in a fixed, deterministic
sequence converges to the WRONG stationary distribution even though each move's own acceptance
formula is individually correct.

A minimal worked example makes this concrete. Take the ideal-gas limit (``\Delta U \equiv 0``)
with ``x \equiv fV/k_BT = 2`` and a toy system capped at ``N \in \{0,1\}``, so
``A_{\mathrm{ins}}(0\to1) = \min(1,x) = 1`` and ``A_{\mathrm{del}}(1\to0) = \min(1,1/x) = 0.5``.
The correct (fair-mixture) stationary ratio is the ``N{=}0..1`` truncation of the ideal-gas law,
``P(1)/P(0) = x = 2``. Alternating insertion then deletion DETERMINISTICALLY instead gives a
period-2 chain: starting from any distribution ``(p_0,p_1)``, an insertion step sends everything
at ``N{=}0`` to ``N{=}1`` (since ``A_{\mathrm{ins}}=1``), leaving ``(0,1)``; the following deletion
step then sends half of that back to ``N{=}0`` (since ``A_{\mathrm{del}}=0.5``), giving
``(0.5,0.5)`` — already the chain's own stationary point, ratio ``P(1)/P(0)=1``, exactly HALF the
correct value. Every individual insertion and deletion attempt used the textbook-correct
acceptance formula; only the SCHEDULE was wrong.

### The rigid-body orientational factor

``\Omega_{\mathrm{tot}}``, the total measure of ``SO(3)``, never appeared in the acceptance ratio
above, even though both insertion's proposal (uniform orientation, density
``1/\Omega_{\mathrm{tot}}``) and the implicit ``q_{\mathrm{rot}}`` folded into ``z=f/k_BT``
individually depend on it. Two conditions make it cancel:

1. **The guest is rigid, with no internal orientational potential of its own.**
   ``q_{\mathrm{rot}} = \int_{SO(3)} d\Omega\,\exp(-U_{\mathrm{int}}(\Omega)/k_BT)``; for a rigid
   body ``U_{\mathrm{int}} \equiv 0`` identically (only the molecule's interaction with the REST of
   the system, already counted in ``\Delta U``, depends on orientation), so
   ``q_{\mathrm{rot}} = \int_{SO(3)} d\Omega = \Omega_{\mathrm{tot}}`` exactly — not
   approximately, and not something that needs measuring.
2. **The trial orientation is drawn exactly uniformly on ``SO(3)``** (`shoemake_quaternion`),
   matching the SAME measure ``q_{\mathrm{rot}}``'s integral runs over. Substituting
   ``z = f/k_BT`` (which used ``q_{\mathrm{rot}} = \Omega_{\mathrm{tot}}``) into the ratio's
   numerator and the proposal density ``1/(V\Omega_{\mathrm{tot}})`` into its denominator leaves
   ``\Omega_{\mathrm{tot}}/\Omega_{\mathrm{tot}} = 1``.

**If the hard-core rejection stage is ever used to *resample* an insertion pose** — redrawing the
orientation (or position) until some clearance condition holds, rather than drawing once uniformly
and letting the ordinary Metropolis test reject a bad pose — condition 2 breaks: the EFFECTIVE
proposal density is then uniform *conditioned on clearing the host*, not uniform over all of
``SO(3)``, and the acceptance ratio needs an explicit correction for that conditioning. Using
hard-core rejection only to assign a provably-zero Boltzmann weight to a pose that is then
rejected as an ordinary Metropolis attempt (what `widom` and the μVT moves both do) changes
nothing, since the pose was still drawn uniformly to begin with and simply carries zero weight.

### Units: two systems meeting at one boundary

Exactly one place in this milestone has two unit systems meeting: `peng_robinson_fugacity` takes
and returns pressure in pascals, the SI unit its equation of state is conventionally quoted in,
while every energy in this package — `kT`, every term of ``\Delta U``, ``V`` in ų — is in eV and
Å. An acceptance ratio needs ``fV/k_BT`` to be a pure number, so ``f`` must be converted from
pascals to eV·Å⁻³ before it reaches `log_insertion_prefactor`/`log_deletion_prefactor`:

```math
1\ \mathrm{Pa} = 1\ \mathrm{J/m^3} = \frac{1}{1.6021766208\times10^{-19}}\ \frac{\mathrm{eV}}{10^{30}\ \mathrm{\mathring A}^3}
\quad\Longrightarrow\quad
\texttt{PASCAL} \equiv \frac{1}{10^{30}\times 1.6021766208\times10^{-19}}\ \mathrm{eV\cdot\mathring A^{-3}\cdot Pa^{-1}} \approx 6.2415\times10^{-12}.
```

Peng–Robinson itself is scale-invariant in pressure (only the reduced pressure ``P/p_c`` enters
its cubic), so this conversion cannot be caught by anything INSIDE `peng_robinson_fugacity` — it
has to happen exactly once, at the boundary where a pascal-valued `f` crosses into an
eV·Å⁻³-valued acceptance ratio (`mc_insert!`/`mc_delete!`, `PASCAL` in `src/constants.jl`).

**A missing factor of `PASCAL` (1.6e11 too large) is invisible to every energy check.**
``\Delta U`` is computed identically whether or not the fugacity conversion is right — the error
lives entirely in the combinatorial PREFACTOR, which never touches `insertion_energy`,
`total_energy`, or the running structure factor `Sk`. `audit_energy!` recomputes `Sk` and the
total energy from the current poses and compares them to the running, incrementally-updated
values; both sides of that comparison are built from ``\Delta U``, so a chain that accepts far too
many insertions (its prefactor 1.6e11 too generous) still has an internally CONSISTENT energy at
every checkpoint, right up until its loading saturates at capacity. **A validation test that
computes its own expected value from the code's own `f`, `V` and `kT` cannot catch this either**,
for the same reason a self-referential test never catches a scale error: ``\langle N\rangle =
fV/k_BT`` holds whether `f` is 1.6e11 too large or exactly right, as long as the SAME `f` the
acceptance ratio consumes is also the one the assertion's target is computed from. What DOES catch
it is a physical anchor computed independently of the code under test — a textbook constant
(Loschmidt's number, one molecule per 37,219 ų at 1 atm and 273.15 K) or a literal,
hand-transcribed pressure/volume/temperature triple, never obtained by calling
`peng_robinson_fugacity` or reading `PASCAL` back out of the package.

### Peng–Robinson and the stable-phase root

Below the critical temperature, the Peng–Robinson cubic in the compressibility factor ``Z`` has
three real roots over a range of pressures — the liquid–vapor coexistence region. Which root
describes the actual, stable phase at a given ``(P,T)`` is a question the equation of state alone
does not answer without an extra physical principle: **the stable phase is whichever one has the
lower Gibbs free energy.**

At fixed ``T`` and ``P``, a phase's molar Gibbs energy is ``g(T,P) = g_{\mathrm{ideal}}(T,P) +
RT\ln\varphi(T,P)``, where ``g_{\mathrm{ideal}}(T,P)`` is the ideal-gas reference at the same
``T,P`` — identical for every root, since it depends only on ``T`` and ``P``, not on which root of
the cubic is used — and ``\varphi = f/P`` is the fugacity coefficient at that root's ``Z``.
Comparing ``g`` across roots is therefore exactly comparing ``\ln\varphi`` across roots: the root
with the smallest ``\varphi`` has the smallest ``g``, and is the thermodynamically stable one.
`peng_robinson_fugacity` implements exactly this — among the roots with ``Z > B`` (the only ones
for which ``\log(Z-B)``, hence ``\varphi``, is finite), it selects the one with the smallest
``\varphi``.

This is **not always the largest root.** Below the critical temperature, at a pressure BELOW the
liquid–vapor coexistence pressure, the vapor branch (largest ``Z``) has the lower ``\varphi`` and
is selected; ABOVE the coexistence pressure, the liquid branch (smallest ``Z``) has the lower
``\varphi`` instead, and the minimum-``\varphi`` rule silently switches to it. Always taking the
largest root would return a metastable vapor state above the coexistence pressure with no
indication that anything had changed.

### The fluctuation heat of adsorption

Both ``\langle N\rangle`` and ``\langle U\rangle`` are functions of ``(\mu,V,T)`` through the same
``\Xi`` above. Differentiating ``\ln\Xi = \ln\sum_N (z^N/N!)\,Z(N,V,T)`` with respect to
``\beta\mu`` (``z`` depends on ``\beta\mu`` through ``z = \exp(\beta\mu)q_{\mathrm{rot}}/\Lambda^3``,
so ``\partial/\partial(\beta\mu) = z\,\partial/\partial z``) gives the standard grand-canonical
fluctuation identities, at fixed ``V,T``:

```math
\frac{\partial \langle N\rangle}{\partial(\beta\mu)} = \langle N^2\rangle - \langle N\rangle^2,
\qquad
\frac{\partial \langle U\rangle}{\partial(\beta\mu)} = \langle UN\rangle - \langle U\rangle\langle N\rangle.
```

The ratio of these two derivatives is ``(\partial U/\partial N)`` along the equilibrium curve — the
differential internal energy of the adsorbed phase per additional guest. Subtracting ``k_BT`` —
the ideal-gas translational enthalpy (``pV=Nk_BT``) a gas-phase molecule loses on adsorbing, since
it is no longer free to explore the gas phase — gives the isosteric heat of adsorption:

```math
q_{\mathrm{st}} = k_BT - \frac{\langle UN\rangle - \langle U\rangle\langle N\rangle}{\langle N^2\rangle - \langle N\rangle^2}
= k_BT - \frac{\operatorname{cov}(U,N)}{\operatorname{var}(N)}.
```

This is the SAME sign convention as Milestone A's zero-loading ``q_{\mathrm{st}} = k_BT - \langle
\Delta U\,W\rangle/\langle W\rangle``: the finite-loading fluctuation formula is the general form,
and the zero-loading Widom expression is its own dilute limit. kUPS's own GCMC analyzer computes
the OPPOSITE sign, ``\operatorname{cov}(U,N)/\operatorname{var}(N) - k_BT``, from its own Widom
analyzer's convention, so comparing against kUPS's GCMC output means negating it, not this
package's result.

``\operatorname{cov}(U,N)/\operatorname{var}(N)`` is exactly the ordinary least-squares regression
slope of ``U`` on ``N``: writing ``\rho = \operatorname{corr}(U,N)`` and ``\sigma_U,\sigma_N`` for
the two standard deviations, the slope is ``b = \rho\,\sigma_U/\sigma_N``, and the standard result
for a regression slope's sampling variance, ``\operatorname{Var}(\hat b) =
\sigma_{\mathrm{resid}}^2/(n\sigma_N^2)`` with residual variance ``\sigma_{\mathrm{resid}}^2 =
\sigma_U^2(1-\rho^2)``, gives its RELATIVE variance directly:

```math
\frac{\operatorname{Var}(\hat b)}{b^2} = \frac{\sigma_U^2(1-\rho^2)/(n\sigma_N^2)}{\rho^2\sigma_U^2/\sigma_N^2}
= \frac{1-\rho^2}{n\,\rho^2}.
```

A correlation ``\rho`` close to ``\pm1`` therefore converges almost as fast as a direct mean
(``n`` cycles, relative error ``\propto 1/\sqrt n``) despite ``q_{\mathrm{st}}`` being a ratio of
two small differences of large sums; weakly correlated ``U`` and ``N`` need far more cycles for
the same relative precision. This is a SCALING relation, not a substitute for the jackknife error
`fluctuation_qst` actually reports — the formula above assumes independent samples, while
successive GCMC cycles are correlated exactly as "block averaging" above discusses, which is why
the real error comes from a delete-one-block jackknife rather than this asymptotic formula
evaluated at ``n=``the raw cycle count. Measured on RUBTAK 3×3×3 + CO2 at 298.15 K and 2e4 Pa
fugacity, 5000 production cycles (`bench/qst_fluctuation_bench.jl`,
`bench/results/pureadsorb_qst_fluctuation_neuromancer_cpu_f64_20260927_e691d0d.json`):
``\operatorname{corr}(U,N) = -0.9945`` (``\rho^2 = 0.989``), and the resulting relative errors are
0.54% for ``q_{\mathrm{st}}`` against 5.3% for the loading itself — an order of magnitude tighter,
consistent with ``\rho^2`` close enough to 1 that ``q_{\mathrm{st}}``'s effective sample size is
barely reduced relative to a direct mean, while the loading's own error is set by ordinary
occupancy variance with no such correlation boost.

### Capacity and truncation

`SystemState`'s fixed per-system capacity biases nothing while it is never reached: every energy
and acceptance calculation reads only the `occupancy[n]` LIVE guests (`guest_range`), never the
reserved-but-unused slots beyond it, so a capacity of 200 and a capacity of 40 give IDENTICAL
physics for a chain whose true equilibrium loading stays under 40 — the extra slots are pure
unused memory. The bias appears only at the moment an insertion the Metropolis test WOULD have
accepted has nowhere to go: `mc_insert!` throws immediately when this happens (`capacity_hits`)
rather than silently rejecting the move, because a silent rejection would sample a distribution
truncated at `capacity` — every configuration with more guests than `capacity` allows would have
probability zero instead of its true, nonzero Boltzmann weight — and nothing in the energy audit
would notice, since the audit checks that the energy of the CURRENT (truncated) configuration is
self-consistent, not that the configuration's own occupancy distribution is untruncated.

That capacity was never reached must therefore be DEMONSTRATED for a given run, not assumed from
the absence of a thrown error (a run that ends the cycle before it would have thrown looks
identical to one that never came close). `GCMCResult`/`IsothermResult` report `max_occupancy` (the
highest occupancy reached over the WHOLE run, warmup included) alongside `capacity` for exactly
this reason: a committed 50-point isotherm run
(`bench/results/pureadsorb_isotherm_co2_rubtak_neuromancer_cuda_f64_20260927_873c9ed.json`) fixes
`capacity = 200` at every pressure point and reaches a highest `max_occupancy` of 136 across all
fifty points — capacity was never approached, and this is a property of the recorded run that can
be checked without rerunning it, not an assumption about the underlying physics.
