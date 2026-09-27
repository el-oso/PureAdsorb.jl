# Boltzmann constant in eV/K and the Coulomb prefactor 1/(4πε₀) in eV·Å/e².
# CODATA 2014 values as used by kUPS (via ASE's units table), so energies agree with it.
const KB = 1.38064852e-23 / 1.6021766208e-19
const KE = 14.399645351950548

# Pa -> eV/Å³: 1 Pa = 1 J/m³ = (1/1.6021766208e-19) eV/(1e30 Å³). A μVT acceptance ratio
# (`log_insertion_prefactor`/`log_deletion_prefactor`) needs `f*V` in the same energy units as
# `kT`, and `f` from `peng_robinson_fugacity` is in Pa (the same units as its `P` argument), so
# every fugacity crossing into a μVT move must be scaled by this factor exactly once.
const PASCAL = 1 / (1.0e30 * 1.6021766208e-19)
