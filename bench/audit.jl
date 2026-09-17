# StrictMode gate for PureAdsorb's hot kernels.
#
# STRICT_MODE=fast (default): StrictMode's value-free scan (no analysis backend) — cheap,
# reports structural allocation/type-stability guesses.
# STRICT_MODE=full: StrictModeTest's proof tier, backed by AllocCheck and JET — the rigorous
# gate. TrimCheck is loaded alongside so a `:trimsafe` guarantee would also be provable, though
# it is not requested below.
#
# Run:
#   julia --project=bench bench/audit.jl                    # fast, from any checkout
#   STRICT_MODE=full julia --project=bench bench/audit.jl    # full, the pre-merge gate
using PureAdsorb, StrictMode, StaticArrays
using StrictModeTest, JET, AllocCheck, TrimCheck   # declared bench deps; assert_enabled requires them loaded

StrictMode.assert_enabled()

const MODE = get(ENV, "STRICT_MODE", "fast") == "full" ? :full : :fast

signature_findings(f, types; guarantees) = MODE === :full ?
    StrictModeTest.proof_findings(f, types; guarantees) : StrictMode.findings(f, types; guarantees)

const G = PureAdsorb.Guest{Float64, 3}
const V3 = Vector{SVector{3, Float64}}
const M3 = SMatrix{3, 3, Float64, 9}

results = vcat(
    signature_findings(
        PureAdsorb.insertion_energy,
        (
            SVector{3, Float64}, SVector{4, Float64}, G, Matrix{Float64}, Matrix{Float64}, Float64,
            V3, Vector{Int32}, Vector{Float64}, M3, M3, Float64, V3, Vector{Float64}, Vector{ComplexF64}, Float64,
        );
        guarantees = (:typestable, :noalloc)
    ),
    signature_findings(PureAdsorb.minimum_image, (M3, M3, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rotate, (SVector{4, Float64}, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.erfc_dev, (Float64,); guarantees = (:typestable, :noalloc)),
)
StrictMode.format_findings(stdout, results; format = :text)
exit(StrictMode.nfailures(results))
