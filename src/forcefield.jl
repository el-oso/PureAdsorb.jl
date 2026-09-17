"""
    ForceField{T}

Lennard-Jones parameters for a set of atom types: `names` indexes into the `sigma` (Å) and
`epsilon` (eV) pairwise tables, `cutoff` (Å) truncates the LJ sum, and `tail` selects whether
an analytic long-range tail correction is applied.
"""
struct ForceField{T}
    names::Vector{String}
    sigma::Matrix{T}
    epsilon::Matrix{T}
    cutoff::T
    tail::Bool
end

"""
    ForceField(names, σ, ε; cutoff, tail = true) -> ForceField

Build per-pair `sigma`/`epsilon` tables from per-type LJ parameters (Å, eV) using
Lorentz-Berthelot combining rules: `σ_ij = (σ_i + σ_j)/2`, `ε_ij = √(ε_i ε_j)`.
"""
function ForceField(names::AbstractVector{<:AbstractString}, σ::AbstractVector, ε::AbstractVector; cutoff, tail = true)
    T = float(promote_type(eltype(σ), eltype(ε), typeof(cutoff)))
    idxs = eachindex(names, σ, ε)
    n = length(idxs)
    S = Matrix{T}(undef, n, n)
    E = Matrix{T}(undef, n, n)
    for (a, i) in enumerate(idxs), (b, j) in enumerate(idxs)
        S[a, b] = (σ[i] + σ[j]) / 2
        E[a, b] = sqrt(ε[i] * ε[j])
    end
    # A comprehension over `idxs` would inherit its (possibly offset) axes; `names` and the
    # sigma/epsilon tables must land in the same 1-based positions that `typeindex` searches.
    table = Vector{String}(undef, n)
    for (a, i) in enumerate(idxs)
        table[a] = String(names[i])
    end
    return ForceField{T}(table, S, E, T(cutoff), tail)
end

function typeindex(ff::ForceField, name::AbstractString)
    i = findfirst(==(name), ff.names)
    isnothing(i) && throw(ArgumentError("unknown LJ type $name; force field defines $(ff.names)"))
    return i
end

"""
    read_forcefield(path; T = Float64) -> ForceField{T}

Read a kUPS-style YAML force field file: `parameters` lists `[σ, ε]` (Å, eV) per type name
(a missing σ defaults to 1, a missing ε to 0), `cutoff` (Å) is required, and `tail_correction`
defaults to `true`.
"""
function read_forcefield(path::AbstractString; T = Float64)
    d = YAML.load_file(path)
    params = d["parameters"]
    names = collect(String, keys(params))
    σ = [T(something(params[n][1], 1.0)) for n in names]
    ε = [T(something(params[n][2], 0.0)) for n in names]
    return ForceField(names, σ, ε; cutoff = T(d["cutoff"]), tail = get(d, "tail_correction", true))
end

# A kUPS guest file's critical_pressure sometimes has digit-group underscores (e.g.
# "7_840_000"); YAML then parses it as a String instead of a number.
parse_number(::Type{T}, x::AbstractString) where {T} = parse(T, replace(x, "_" => ""))
parse_number(::Type{T}, x) where {T} = T(x)

"""
    Guest{T, N}

A rigid guest molecule with `N` sites: `sites` gives each site's position (Å) relative to the
molecule's reference frame, `types` indexes the force field's LJ types, `charges` are partial
charges (e), and `tc`/`pc`/`omega` are the critical temperature (K), critical pressure (Pa) and
acentric factor.
"""
struct Guest{T, N}
    sites::SVector{N, SVector{3, T}}
    types::SVector{N, Int}
    charges::SVector{N, T}
    tc::T
    pc::T
    omega::T
end

"""
    read_guest(path, ff::ForceField; T = Float64) -> Guest{T}

Read a kUPS-style YAML guest file: site `positions` (Å), `symbols` (looked up in `ff`),
`charges` (e), `critical_temperature` (K), `critical_pressure` (Pa) and `acentric_factor`.
"""
function read_guest(path::AbstractString, ff::ForceField; T = Float64)
    d = YAML.load_file(path)
    N = length(d["positions"])
    sites = SVector{N}(SVector{3, T}(p...) for p in d["positions"])
    types = SVector{N}(typeindex(ff, s) for s in d["symbols"])
    charges = SVector{N, T}(d["charges"]...)
    return Guest{T, N}(
        sites, types, charges,
        T(d["critical_temperature"]), parse_number(T, d["critical_pressure"]), T(d["acentric_factor"]),
    )
end

function tail_coefficient(ff::ForceField{T}, i, j) where {T}
    ε = ff.epsilon[i, j]
    iszero(ε) && return zero(T)
    s = ff.sigma[i, j]
    x = s / ff.cutoff
    return ε * s^3 * (x^9 / 3 - x^3)
end

# Change of the global analytic tail correction when a guest with `guest_counts` sites per
# species is added to a system holding `counts` particles per species.
function tail_delta(ff::ForceField{T}, counts::AbstractVector{<:Integer}, guest_counts::AbstractVector{<:Integer}, V) where {T}
    ff.tail || return zero(T)
    idxs = eachindex(counts, guest_counts)
    length(idxs) == size(ff.sigma, 1) ||
        throw(DimensionMismatch("counts must have one entry per LJ type ($(size(ff.sigma, 1))), got $(length(idxs))"))
    acc = zero(T)
    for (a, i) in enumerate(idxs), (b, j) in enumerate(idxs)
        c = tail_coefficient(ff, a, b)
        acc += 2 * counts[i] * guest_counts[j] * c + guest_counts[i] * guest_counts[j] * c
    end
    return T(8π / 3) / T(V) * acc
end
