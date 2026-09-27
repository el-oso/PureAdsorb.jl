# Why this package exists

This page is background, not API: why PureAdsorb computes what it computes, and where an
adsorption simulation like this one sits in the larger problem of finding new porous materials.
[Theory](theory.md) has the full derivations; this page gives the physical picture those
derivations serve.

## Adsorption as a phase equilibrium

A porous crystal in contact with a gas is, thermodynamically, two phases in contact: the bulk
gas outside the material, and the gas molecules adsorbed inside its pores. Like any two phases
allowed to exchange particles — liquid and vapor, solute and solvent — they reach equilibrium
when a particle moving from one phase to the other changes the system's free energy by zero net
amount. That condition is equality of chemical potential,

```math
\mu_{\text{gas}}(P, T) = \mu_{\text{ads}}(n, T),
```

where ``n`` is the loading (guests adsorbed per unit of framework) and ``P``, ``T`` are the bulk
gas's pressure and temperature. Everything a simulation does — sampling guest positions,
accepting or rejecting insertions and deletions, accumulating an energy — is machinery for
finding the loading ``n`` that satisfies this one equation at a given ``P`` and ``T``. Nothing
about the pore geometry, the force field, or the sampling algorithm changes what equilibrium
*means*; they only change how expensive it is to compute.

The difficulty is the left-hand side. For an ideal gas, ``\mu_{\text{gas}}`` is elementary:
``\mu_{\text{ideal}}(P,T) = \mu^\circ(T) + k_BT\ln(P/P^\circ)``, a single logarithm. A real gas
has none of that simplicity — its molecules attract and repel each other, so its chemical
potential is a complicated function of density that in general has no closed form. Simulating
that complexity directly, molecule by molecule, inside the equilibrium condition above, is
possible but wasteful: the bulk gas phase is not what this package is trying to understand, and
its equation of state is well studied and cheap to evaluate on its own.

**Fugacity** is the standard way to keep the ideal-gas formula and hide the real gas's
complexity in a single correction factor. Define ``f``, the fugacity, as the pressure an
*ideal* gas of the same substance would need to reproduce the real gas's actual chemical
potential:

```math
\mu_{\text{gas}}(P, T) \equiv \mu^\circ(T) + k_BT \ln\!\left(\frac{f}{P^\circ}\right).
```

This is not an approximation — it is a definition, valid at any pressure, that shifts every bit
of real-gas non-ideality into the single number ``f(P,T)``. As ``P \to 0`` any gas becomes
ideal, so ``f \to P`` in that limit; the ratio ``\varphi \equiv f/P``, the fugacity coefficient,
is how far a real gas at a given ``P`` and ``T`` departs from ideal behavior. Once ``f`` is in
hand, every formula that would apply to an ideal gas at pressure ``f`` applies to the real gas at
pressure ``P`` — including, as [Theory](theory.md) derives in detail, the acceptance ratios a
grand canonical Monte Carlo simulation needs. This is why fugacity, not pressure, is the
variable that reaches the acceptance ratio: it is the one number that already encodes the bulk
phase's equation of state, so the simulation never has to reason about intermolecular forces in
the gas phase directly.

**Peng–Robinson** is the equation of state this package uses to compute ``f`` from a measurable
``P`` and ``T``. It relates pressure, molar volume and temperature through a cubic equation in
the compressibility factor ``Z = PV_m/RT``, parameterized only by the substance's critical
temperature, critical pressure and acentric factor — quantities tabulated for essentially any
gas of interest. Fugacity follows from any equation of state through the general thermodynamic
relation

```math
\ln\varphi(T,P) = \int_0^P \frac{Z(T,P') - 1}{P'}\,dP',
```

which measures the accumulated departure of ``Z`` from its ideal-gas value of 1 as pressure
rises from zero. Peng–Robinson supplies a closed-form ``Z(T,P)`` (by solving its cubic) and a
closed-form evaluation of this integral, so ``f`` comes out of a few algebraic steps rather than
a numerical integration. Peng–Robinson is the bridge, in other words, from "pressure and
temperature, which an experimentalist sets and measures" to "fugacity, the one number a
simulation of the adsorbed phase actually needs."

#### Figure: phase equilibrium

```@raw html
<svg viewBox="0 0 720 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bulk gas at pressure P and temperature T maps through the Peng-Robinson equation of state to a fugacity f, which sets the chemical potential shared with the adsorbed phase.">
  <defs>
    <marker id="ctx-arrow1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="10" y="55" width="160" height="60" rx="8"/>
    <rect x="280" y="55" width="140" height="60" rx="8"/>
    <rect x="550" y="55" width="160" height="60" rx="8"/>
    <line x1="170" y1="85" x2="272" y2="85" marker-end="url(#ctx-arrow1)"/>
    <line x1="420" y1="85" x2="542" y2="85" marker-end="url(#ctx-arrow1)"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle" font-family="sans-serif">
    <text x="90" y="80">Bulk gas</text>
    <text x="90" y="96">(P, T)</text>
    <text x="221" y="72">Peng&#8211;Robinson</text>
    <text x="221" y="102">equation of state</text>
    <text x="350" y="90">Fugacity f</text>
    <text x="481" y="72">&#956;(f,T) = &#956;(n,T)</text>
    <text x="481" y="102">equal chemical</text>
    <text x="481" y="116">potential</text>
    <text x="630" y="80">Adsorbed phase</text>
    <text x="630" y="96">(loading n)</text>
  </g>
</svg>
```

## The isotherm and its two regimes

An adsorption **isotherm** is the loading ``n`` as a function of pressure at fixed temperature —
the curve an experimentalist measures by dosing a sample with gas and weighing it at each
pressure. Its shape has two regimes with different physics, and this package computes each with
a different method.

**Low pressure: Henry's law.** When the loading is small, adsorbed guests are so dilute that
they essentially never encounter each other; each one interacts only with the host framework, as
if it were the only guest present. Under this approximation, the grand partition function
``\Xi = \sum_N (z^N/N!) Z(N,V,T)`` (``z = f/k_BT``, the absolute activity — see
[Theory](theory.md)) can be truncated after its first two terms, since every term beyond ``N=1``
requires two or more guests present simultaneously. Differentiating the truncated sum gives the
mean loading to leading order in ``z``:

```math
\langle N \rangle \approx z \int_V d\mathbf{r}\int_{SO(3)} d\Omega\,\exp(-\Delta U_{\text{ins}}/k_BT)
= \frac{f}{k_BT}\, V \langle W \rangle = f\, K_H,
```

where ``\langle W \rangle`` is exactly the Boltzmann-weighted average one test-particle insertion
into the *empty* framework would sample, and ``K_H \equiv V\langle W\rangle / k_BT`` is Henry's
constant. Loading is therefore proportional to pressure at low pressure — a straight line
through the origin — and its slope is set entirely by how a single guest interacts with the bare
host, with no many-body guest–guest physics involved at all.

**High pressure: saturation.** As pressure rises, the pore fills, guests begin to compete for
the same space and interact with each other, and eventually the material approaches its maximum
capacity; the loading curve bends over and flattens (a Type-I isotherm, in the standard
classification). None of the simplifications above hold here: computing the loading now requires
sampling the full many-body configuration of however many guests the pore happens to hold at
that pressure, which fluctuates from one moment to the next.

**Two methods, one for each regime — but only one that can compute the whole curve.** Widom
test-particle insertion computes exactly the dilute-limit average ``\langle W \rangle`` above: it
inserts trial guests into a single, fixed configuration (empty, for the Henry-law limit) and
never has to simulate a fluctuating population. It is cheap and exact in that limit, but it has
nothing to say about saturation, since the whole point of the calculation is that only one guest
is ever present at a time. Grand canonical Monte Carlo (GCMC), by contrast, lets the guest count
itself fluctuate at fixed fugacity, sampling the true equilibrium population at *any* pressure —
dilute or saturated — at the cost of a much more expensive simulation (a Markov chain over
insertions, deletions and moves, rather than a single pass of independent trial insertions).
GCMC computes the whole isotherm; Widom insertion computes only its zero-loading slope, fast.

Because both methods are, in the dilute limit, computing the *same* physical quantity — the
Henry coefficient — by different routes (one an analytic zero-loading limit, the other the
low-pressure end of a full stochastic simulation), they must agree there. This package checks
exactly that agreement as a validation test: a GCMC isotherm's own low-pressure slope is compared
against Widom insertion's independently computed ``K_H``, run separately, with its own random
stream, on the pristine framework (`docs/src/validation.md`'s Henry/detailed-balance crosscheck).
Agreement is not assumed — it is the kind of internal consistency check that catches an error a
single method's own internal bookkeeping (an energy audit, say) would never see, since both
routes would have to be wrong in exactly the same way to agree by accident.

#### Figure: two isotherm regimes

```@raw html
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="An isotherm is linear at low pressure, matching the Henry coefficient from Widom insertion, and saturates at high pressure, requiring grand canonical Monte Carlo for the whole curve.">
  <defs>
    <marker id="ctx-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g stroke="currentColor" stroke-width="1.5" fill="none">
    <line x1="60" y1="210" x2="600" y2="210" marker-end="url(#ctx-arrow2)"/>
    <line x1="60" y1="210" x2="60" y2="20" marker-end="url(#ctx-arrow2)"/>
    <path d="M 60 210 C 160 160, 260 60, 320 40 S 480 20, 590 15" stroke-width="2.5"/>
    <path d="M 60 210 L 220 95" stroke-dasharray="4 4"/>
  </g>
  <g fill="currentColor" font-size="13" font-family="sans-serif">
    <text x="300" y="234" text-anchor="middle">pressure</text>
    <text x="30" y="115" text-anchor="middle" transform="rotate(-90 30 115)">loading</text>
    <text x="120" y="170">Widom: zero-loading</text>
    <text x="120" y="186">slope K_H (Henry's law)</text>
    <text x="430" y="55">GCMC: the whole curve</text>
    <circle cx="90" cy="185" r="5" fill="currentColor"/>
    <text x="105" y="150">must agree here</text>
    <text x="105" y="164">(validated crosscheck)</text>
  </g>
</svg>
```

## Where this sits in materials discovery

Porous materials — metal-organic frameworks, zeolites, activated carbons — are candidates for
gas storage, separation, and carbon capture, and the space of chemically possible structures is
vast: choices of metal node, organic linker, pore topology and functional group combine
combinatorially into far more candidate materials than could ever be synthesized and measured
one at a time in a lab. A generative model — trained to propose plausible crystal structures,
whether by evolving known frameworks, sampling a learned distribution over structures, or some
other route — can produce candidates far faster than any experimental or simulation pipeline can
evaluate them.

That mismatch is exactly the role Monte Carlo simulation plays in this pipeline: it is the
*label*. A generative model's candidate structure is just a set of atomic coordinates until
something computes what it would actually do — how much gas it adsorbs, at what pressure, at
what temperature. Grand canonical Monte Carlo, run computationally rather than in a lab, supplies
that label for any candidate structure the generative model proposes, without needing to
synthesize anything. The labels then feed back into training or filtering the generative model
itself, closing a loop between "propose a structure" and "know whether it's any good."

That loop's throughput is set by whichever step is slowest, and for a candidate pool in the
thousands to millions, the simulation step is that bottleneck: proposing a structure is cheap
computationally, but computing its isotherm is not. This is precisely why *batching* — running
many independent frameworks' worth of simulation at once, sharing the framework construction
cost across pressure points and replicas, keeping a GPU's many parallel lanes occupied with many
chains rather than one — is the design axis this package optimizes, rather than only making one
single simulation faster. A discovery loop that can only score materials one at a time, however
fast that one calculation is, still gates the whole pipeline's throughput on however many
candidates arrive per unit time; a loop that can score a batch of candidates together removes
that gate.

#### Figure: the discovery loop

```@raw html
<svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A generative model proposes candidate structures, Monte Carlo simulation computes their adsorption labels, and those labels feed back to the generative model, with the simulation step as the throughput bottleneck.">
  <defs>
    <marker id="ctx-arrow3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="230" y="15" width="180" height="55" rx="8"/>
    <rect x="450" y="130" width="180" height="55" rx="8"/>
    <rect x="230" y="245" width="180" height="55" rx="8"/>
    <rect x="10" y="130" width="180" height="55" rx="8" stroke-width="3"/>
    <path d="M 400 70 C 440 90, 445 105, 460 128" marker-end="url(#ctx-arrow3)"/>
    <path d="M 500 185 C 460 215, 420 230, 400 245" marker-end="url(#ctx-arrow3)"/>
    <path d="M 260 245 C 200 215, 160 200, 130 188" marker-end="url(#ctx-arrow3)"/>
    <path d="M 100 128 C 120 105, 170 85, 240 68" marker-end="url(#ctx-arrow3)"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle" font-family="sans-serif">
    <text x="320" y="47">Generative model</text>
    <text x="540" y="153">Candidate structures</text>
    <text x="320" y="277">Property labels</text>
    <text x="100" y="153">Monte Carlo simulation</text>
    <text x="100" y="169">(throughput bottleneck)</text>
  </g>
</svg>
```

## The division of labor between the energy model and the sampling

Every calculation in this package factors into two separable pieces that never need to know
about each other's internals. One piece is an **energy model**: given a proposed configuration
of atoms, return its interaction energy, or the energy change ``\Delta U`` of a proposed move.
This package's own energy model is a classical force field — Lennard-Jones plus Ewald-summed
Coulomb, the same functional form textbooks use for molecular simulation — but nothing about the
rest of the calculation depends on that choice. A machine-learned interatomic potential, trained
to reproduce quantum-mechanical energies more faithfully than a fixed Lennard-Jones/Coulomb
form can, would slot into exactly the same place: something that consumes a configuration and
returns ``\Delta U``.

The other piece is the **sampler**: Widom insertion or grand canonical Monte Carlo, which knows
nothing about *how* ``\Delta U`` was computed and only uses the resulting numbers — through the
Boltzmann weight ``\exp(-\Delta U/k_BT)`` and the acceptance ratios [Theory](theory.md) derives —
to turn a stream of energies into a thermodynamic property: a Henry coefficient, an isotherm, a
heat of adsorption. The sampler is what converts "the energy of this one configuration" into "the
average behavior of this material at this temperature and pressure," which is the quantity an
engineer actually needs and no single-configuration energy calculation can supply on its own.

Neither half is useful alone. An energy model with no sampler produces one number for one
snapshot — informative about that snapshot, silent about the material's behavior in equilibrium.
A sampler with no energy model has no ``\Delta U`` to accept or reject moves with; the acceptance
ratio's Boltzmann factor is undefined without one. The two halves are interchangeable exactly
because each one only ever hands the other a well-defined quantity (energies in one direction,
accept/reject decisions and configuration updates in the other) rather than sharing internal
state. Improving the energy model — a better force field, a machine-learned potential trained on
more data — makes the *energies* more faithful to the real material's physics without touching a
line of the Monte Carlo machinery; improving the sampler — the batching, block-averaging and
detailed-balance work this package's own design and validation documents describe — makes the
same energies converge to a thermodynamic answer faster and with a trustworthy error bar, without
knowing or caring where those energies came from.

#### Figure: energy model and sampler

```@raw html
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A classical force field or a machine-learned potential each supply the same interface, an energy for a configuration, which Monte Carlo consumes to produce thermodynamic properties.">
  <defs>
    <marker id="ctx-arrow4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="10" y="15" width="200" height="55" rx="8"/>
    <rect x="10" y="145" width="200" height="55" rx="8"/>
    <rect x="360" y="80" width="150" height="60" rx="8"/>
    <path d="M 210 42 C 280 42, 300 70, 358 100" marker-end="url(#ctx-arrow4)"/>
    <path d="M 210 172 C 280 172, 300 140, 358 115" marker-end="url(#ctx-arrow4)"/>
    <line x1="510" y1="110" x2="600" y2="110" marker-end="url(#ctx-arrow4)"/>
  </g>
  <g fill="currentColor" font-size="13" text-anchor="middle" font-family="sans-serif">
    <text x="110" y="38">Classical force field</text>
    <text x="110" y="54">(Lennard-Jones + Ewald)</text>
    <text x="110" y="168">Machine-learned</text>
    <text x="110" y="184">potential</text>
    <text x="280" y="66">&#916;U</text>
    <text x="280" y="200">&#916;U</text>
    <text x="435" y="105">Monte Carlo</text>
    <text x="435" y="121">sampler</text>
    <text x="555" y="100" text-anchor="start">loading,</text>
    <text x="555" y="116" text-anchor="start">q_st,</text>
    <text x="555" y="132" text-anchor="start">isotherm</text>
  </g>
</svg>
```
