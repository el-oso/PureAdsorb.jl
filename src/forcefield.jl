struct ForceField{T}
    names::Vector{String}
    sigma::Matrix{T}
    epsilon::Matrix{T}
    cutoff::T
    tail::Bool
end

function ForceField(names::AbstractVector{<:AbstractString}, σ::AbstractVector, ε::AbstractVector; cutoff, tail = true)
    axes(names, 1) == axes(σ, 1) == axes(ε, 1) || throw(DimensionMismatch("names, σ and ε must share axes: $(axes(names)) $(axes(σ)) $(axes(ε))"))
    T = float(promote_type(eltype(σ), eltype(ε), typeof(cutoff)))
    n = length(names)
    S = Matrix{T}(undef, n, n)
    E = Matrix{T}(undef, n, n)
    for (a, i) in enumerate(eachindex(σ)), (b, j) in enumerate(eachindex(σ))
        S[a, b] = (σ[i] + σ[j]) / 2
        E[a, b] = sqrt(ε[i] * ε[j])
    end
    return ForceField{T}(collect(String, names), S, E, T(cutoff), tail)
end

function typeindex(ff::ForceField, name::AbstractString)
    i = findfirst(==(name), ff.names)
    isnothing(i) && throw(ArgumentError("unknown LJ type $name; force field defines $(ff.names)"))
    return i
end

# kUPS parameter files list [σ, ε] per type name; a missing σ defaults to 1 and a missing ε to 0.
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

struct Guest{T, N}
    sites::SVector{N, SVector{3, T}}
    types::SVector{N, Int}
    charges::SVector{N, T}
    tc::T
    pc::T
    omega::T
end

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
    n = length(ff.names)
    length(counts) == length(guest_counts) == n || throw(DimensionMismatch("counts must have one entry per LJ type ($n)"))
    acc = zero(T)
    for (a, i) in enumerate(eachindex(counts)), (b, j) in enumerate(eachindex(counts))
        c = tail_coefficient(ff, a, b)
        acc += 2 * counts[i] * guest_counts[j] * c + guest_counts[i] * guest_counts[j] * c
    end
    return T(8π / 3) / T(V) * acc
end
