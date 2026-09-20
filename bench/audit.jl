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

insertion_energy_types(::Type{T}) where {T} = (
    SVector{3, T}, SVector{4, T}, PureAdsorb.Guest{T, 3}, Matrix{T}, Matrix{T}, T, T,
    Vector{SVector{3, T}}, Vector{Int32}, Vector{T}, Int, SVector{3, Int32}, SVector{3, Int32}, Vector{Int32},
    SMatrix{3, 3, T, 9}, SMatrix{3, 3, T, 9}, T, Vector{SVector{3, T}}, Vector{T}, Vector{Complex{T}},
)
const M3 = SMatrix{3, 3, Float64, 9}

results = vcat(
    signature_findings(PureAdsorb.insertion_energy, insertion_energy_types(Float64); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.insertion_energy, insertion_energy_types(Float32); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.minimum_image, (M3, M3, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.rotate, (SVector{4, Float64}, SVector{3, Float64}); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.erfc_dev, (Float64,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.pair_erfc_dev, (Float64,); guarantees = (:typestable, :noalloc)),
    signature_findings(PureAdsorb.pair_erfc_dev, (Float32,); guarantees = (:typestable, :noalloc)),
)
StrictMode.format_findings(stdout, results; format = :text)
exit(StrictMode.nfailures(results))
