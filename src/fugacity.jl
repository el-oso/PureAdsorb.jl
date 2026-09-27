# Peng-Robinson fugacity from a guest's critical temperature, critical pressure and acentric
# factor (`Guest.tc`/`.pc`/`.omega`, `src/forcefield.jl`, read from a guest YAML file and stored
# but otherwise unused until this file). A μVT move's acceptance ratio (design doc §1.2) needs the
# fugacity `f` a reservoir at pressure `P` and temperature `T` would have if it behaved ideally at
# that chemical potential (the fugacity section of
# `docs/superpowers/specs/2026-09-27-milestone-c-gcmc.md`); this runs once per system per run, at
# the host, so a plainly readable cubic solve is preferred over a device-kernel one.

# Real and complex roots of the monic cubic `Z^3 + c2*Z^2 + c1*Z + c0 = 0`, always as three complex
# values (a real root has zero imaginary part to rounding) so the return type does not depend on
# how many of them are real: Cardano's formula when one root is real and two are complex
# conjugates, the trigonometric form (the "casus irreducibilis", avoiding complex arithmetic
# entirely) when all three are real. Standard depressed-cubic derivation: substituting
# `Z = t - c2/3` gives `t^3 + p*t + q = 0` with `p = c1 - c2^2/3`, `q = 2*c2^3/27 - c2*c1/3 + c0`,
# discriminant `Δ = (q/2)^2 + (p/3)^3`.
function cubic_roots(c2::T, c1::T, c0::T) where {T <: AbstractFloat}
    p = c1 - c2^2 / 3
    q = 2 * c2^3 / 27 - c2 * c1 / 3 + c0
    shift = c2 / 3
    Δ = (q / 2)^2 + (p / 3)^3
    if Δ >= zero(T)
        s = sqrt(Δ)
        u = cbrt(-q / 2 + s)
        v = cbrt(-q / 2 - s)
        z1 = u + v - shift
        re23 = -(u + v) / 2 - shift
        im23 = (u - v) * sqrt(T(3)) / 2
        return SVector(Complex(z1), Complex(re23, im23), Complex(re23, -im23))
    else
        r = sqrt(-p^3 / 27)
        θ = acos(clamp(-q / (2r), -one(T), one(T)))
        m = 2 * sqrt(-p / 3)
        z1 = m * cos(θ / 3) - shift
        z2 = m * cos((θ + 2 * T(π)) / 3) - shift
        z3 = m * cos((θ + 4 * T(π)) / 3) - shift
        return SVector(Complex(z1), Complex(z2), Complex(z3))
    end
end

# Peng-Robinson's single-component fugacity coefficient at compressibility `Z` (Peng & Robinson
# 1976, eq. 22 with mole fraction 1 and no binary interaction — the same reduction kUPS's own test
# suite checks its multicomponent formula against, `test/mcmc/test_fugacity.py::_pr_log_phi`).
# Finite only for `Z > B`; callers evaluate this only at roots already filtered on that condition.
log_fugacity_coefficient(Z::T, A::T, B::T) where {T} =
    (Z - 1) - log(Z - B) - A / (2 * sqrt(T(2)) * B) * log((Z + (1 + sqrt(T(2))) * B) / (Z + (1 - sqrt(T(2))) * B))

"""
    peng_robinson_fugacity(P, Tgas, tc, pc, omega) -> (; f, phi, Z)
    peng_robinson_fugacity(P, Tgas, guest::Guest) -> (; f, phi, Z)

Peng-Robinson fugacity `f` (same pressure units as `P`), fugacity coefficient `phi = f/P`, and
compressibility factor `Z` for a pure gas at pressure `P` and temperature `Tgas`, from its critical
temperature `tc`, critical pressure `pc` and acentric factor `omega` (`Guest`'s fields). Unlike
kUPS, which returns `log(f)`, this returns `f`/`phi` directly (what a μVT move's acceptance ratio
needs); `phi = exp(ln_phi)` can overflow `Inf` for `P`/`Tgas`/`omega` combinations far outside any
real gas's, where kUPS's log-domain result would stay finite.

The Peng-Robinson equation of state is a cubic in `Z`. Above the critical temperature it has one
real root; below it, over a range of pressures where liquid and vapor coexist, it has three. Of the
roots with `Z > B` (the only ones for which `log(Z - B)`, hence the fugacity coefficient, is
finite), this picks the one with the LOWEST fugacity coefficient — equivalently the lowest Gibbs
free energy, the thermodynamically stable phase at `(P, Tgas)` — matching kUPS's own selection
(`mcmc/fugacity.py`, itself following RASPA2's `equations_of_state.c`) and Peng & Robinson's
original phase-stability criterion.

This is NOT always the largest root: at a pressure above the liquid-vapor coexistence pressure for
a subcritical `Tgas`, the SMALLEST valid root (the liquid-like density) has the lower fugacity
coefficient and is selected instead — pinned directly against kUPS at exactly such a point in
`fugacity_tests.jl` ("three real roots, above the coexistence pressure, selects the liquid
branch"). A caller supplying only genuinely vapor-phase state points (any pressure at or above the
critical temperature, or a pressure below the coexistence pressure otherwise) always gets the
largest (vapor) root back, since that is then the one with the lowest fugacity coefficient — the
regime `fugacity_tests.jl`'s other three-real-root pins exercise.

Throws `ArgumentError` for a non-positive `P`, `Tgas`, `tc` or `pc`, and if no root has `Z > B`.
That second case cannot happen for positive `P`/`Tgas`/`tc`/`pc` (any acentric factor): the cubic's
value at `Z = B` is identically `-2*B^2` regardless of `A`, and it is a monic cubic (so it goes to
`+Inf` as `Z -> Inf`), so a real root strictly above `B` always exists by the intermediate value
theorem. This check is a defensive fail-fast against non-finite coefficients (overflow from an
extreme input) rather than a reachable branch for physically sensible arguments.
"""
function peng_robinson_fugacity(P::T, Tgas::T, tc::T, pc::T, omega::T) where {T <: AbstractFloat}
    P > zero(T) || throw(ArgumentError("peng_robinson_fugacity: P=$P must be positive"))
    Tgas > zero(T) || throw(ArgumentError("peng_robinson_fugacity: Tgas=$Tgas must be positive"))
    tc > zero(T) || throw(ArgumentError("peng_robinson_fugacity: tc=$tc must be positive"))
    pc > zero(T) || throw(ArgumentError("peng_robinson_fugacity: pc=$pc must be positive"))
    Pr = P / pc
    Tr = Tgas / tc
    m = T(0.37464) + T(1.54226) * omega - T(0.26992) * omega^2
    sqrt_alpha = 1 + m * (1 - sqrt(Tr))
    A = T(0.45724) * Pr / Tr^2 * sqrt_alpha^2
    B = T(0.0778) * Pr / Tr

    roots = cubic_roots(B - 1, A - 3 * B^2 - 2 * B, -(A * B - B^2 - B^3))
    # `eps(T)^(1/2)` bounds the rounding noise on a real root's (nominally zero) imaginary part;
    # `eps(T)^(3/4)` is the strictly-positive margin `Z` needs over `B` for `log(Z - B)` to stay
    # finite and well-scaled — the same two tolerances kUPS's own root filter uses.
    tol_im = sqrt(eps(T))
    tol_re = eps(T)^T(3 / 4)

    bestZ = T(NaN)
    bestlnphi = T(Inf)
    for r in roots
        Zc = real(r)
        (abs(imag(r)) < tol_im && Zc > B + tol_re) || continue
        lnphi = log_fugacity_coefficient(Zc, A, B)
        if lnphi < bestlnphi
            bestlnphi = lnphi
            bestZ = Zc
        end
    end
    isfinite(bestlnphi) || throw(
        ArgumentError(
            "peng_robinson_fugacity: no root has Z > B (an unphysical state point) at " *
                "P=$P, Tgas=$Tgas, tc=$tc, pc=$pc, omega=$omega"
        )
    )
    phi = exp(bestlnphi)
    return (f = phi * P, phi = phi, Z = bestZ)
end

peng_robinson_fugacity(P, Tgas, guest::Guest{T}) where {T} =
    peng_robinson_fugacity(T(P), T(Tgas), guest.tc, guest.pc, guest.omega)
