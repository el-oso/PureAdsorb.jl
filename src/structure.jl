"""
    Framework{T}

A periodic host structure: unit cell `cell` (Å, lattice vectors as columns), atom fractional
coordinates `frac`, per-atom `labels` and element `symbols`, partial `charges` (e), and
`replication`, the `(nx, ny, nz)` supercell factors already folded into `cell`/`frac`
relative to the CIF-read unit cell (`(1,1,1)` for an unreplicated framework). `replication`
asserts that `frac`/`labels`/`symbols`/`charges` are exact translational copies of a smaller
cell repeated `nx × ny × nz` times — `FrameworkBatch` checks this claim on a sample of up to 32
k-vectors rather than trusting it blindly, which gives high probability, not certainty, of
catching a false claim; a caller constructing a `Framework` directly (not via `replicate`) is
responsible for making it true.
"""
struct Framework{T}
    cell::SMatrix{3, 3, T, 9}
    frac::Vector{SVector{3, T}}
    labels::Vector{String}
    symbols::Vector{String}
    charges::Vector{T}
    replication::NTuple{3, Int}
end

Framework{T}(cell, frac, labels, symbols, charges) where {T} =
    Framework{T}(cell, frac, labels, symbols, charges, (1, 1, 1))

natoms(fw::Framework) = length(fw.frac)
total_charge(fw::Framework) = sum(fw.charges)
cartesian(fw::Framework) = [fw.cell * f for f in fw.frac]

# Parses one CIF symmetry-operation string ("x,y,z", "-x+1/2,y,-z+1/2", "1/2-y,1/2+x,z", ...)
# into a function `frac::SVector{3} -> SVector{3}` applying it, wrapped into [0, 1). Each of the
# three comma-separated terms is a signed sum of `a*var` (var one of x, y, z; `a` an optional
# leading fraction or integer, default 1) and constant fraction/integer pieces; CIF never nests
# parentheses or uses any other operator here, so a fixed-width regex over signed tokens is exact.
function parse_symop(expr::AbstractString)
    coeff = zeros(Rational{Int}, 3, 3)
    const_term = zeros(Rational{Int}, 3)
    varidx = Dict('x' => 1, 'y' => 2, 'z' => 3)
    for (row, term) in enumerate(split(replace(expr, " " => ""), ','))
        for m in eachmatch(r"([+-]?\d*/?\d*)([xyz])|([+-]?\d+/\d+|[+-]?\d+)(?![xyz])", term)
            if !isnothing(m.captures[2])
                coefstr, var = m.captures[1], m.captures[2][1]
                a = isempty(coefstr) || coefstr == "+" ? one(Rational{Int}) :
                    coefstr == "-" ? -one(Rational{Int}) : parse_frac(coefstr)
                coeff[row, varidx[var]] += a
            else
                const_term[row] += parse_frac(m.captures[3])
            end
        end
    end
    A = SMatrix{3, 3}(coeff)
    b = SVector{3}(const_term)
    return frac -> wrap_frac.(A * frac + b)
end
parse_frac(s::AbstractString) = occursin('/', s) ? (parts = split(s, '/'); parse(Int, parts[1]) // parse(Int, parts[2])) : Rational{Int}(parse(Int, s))

# Expands the asymmetric unit `frac`/`labels`/`symbols`/`charges` under every operation in `ops`
# to P1, dropping images that coincide (within `tol` fractional units, periodic) with one already
# kept — a symmetry-equivalent position generated more than once (an atom on a special Wyckoff
# position, or the identity operation itself) contributes exactly one atom, not one per operation.
function expand_symmetry(ops, frac::Vector{SVector{3, T}}, labels, symbols, charges; tol = T(1.0e-3)) where {T}
    efrac = SVector{3, T}[]; elabels = String[]; esymbols = String[]; echarges = T[]
    periodic_close(a, b) = all(d -> (d = abs(d - round(d)); d < tol), a - b)
    for a in eachindex(frac)
        for op in ops
            p = op(frac[a])
            any(q -> periodic_close(q, p), efrac) && continue
            push!(efrac, p); push!(elabels, labels[a]); push!(esymbols, symbols[a]); push!(echarges, charges[a])
        end
    end
    return efrac, elabels, esymbols, echarges
end

"""
    read_cif(path; T = Float64, charges = nothing) -> Framework{T}

Read a CIF: one data block, one `_atom_site` loop with fractional coordinates. Symmetry
operations (`_symmetry_equiv_pos_as_xyz` or `_space_group_symop_operation_xyz`, whichever loop
the file has) expand the asymmetric unit to P1, deduplicating positions that coincide within
1e-3 fractional units — a file with neither loop is accepted only when its H-M space-group name
is literally `P1` (the implicit, one-operation case), and refused (naming the space group) for
any other name, rather than guessing a symmetry table from it.

Partial charges (e) come from the file's own `_atom_site_charge` column when present and
`charges` is not given; `charges`, when given, is a `label => charge` mapping (this class of
material typically assigns charges per crystallographic role via a force field, not per atom in
the CIF) that supplies every expanded atom's charge by its original asymmetric-unit label,
overriding any in-file column. Throws if neither source gives every atom a charge, naming
whichever labels `charges` is missing.
"""
function read_cif(path::AbstractString; T = Float64, charges::Union{Nothing, AbstractDict{<:AbstractString, <:Real}} = nothing)
    lines = strip.(readlines(path))
    getval(key) = begin
        i = findfirst(l -> startswith(l, key * " ") || startswith(l, key * "\t"), lines)
        isnothing(i) && throw(ArgumentError("CIF is missing $key"))
        strip(lines[i][(length(key) + 1):end])
    end
    loop_strings(header) = begin
        i = findfirst(==(header), lines)
        isnothing(i) && return nothing
        i += 1
        out = String[]
        while i <= length(lines) && !isempty(lines[i]) && !startswith(lines[i], "_") && !startswith(lines[i], "loop_")
            push!(out, strip(lines[i], ['\'', '"'])); i += 1
        end
        return out
    end
    opstrings = loop_strings("_symmetry_equiv_pos_as_xyz")
    isnothing(opstrings) && (opstrings = loop_strings("_space_group_symop_operation_xyz"))
    if isnothing(opstrings)
        sg = strip(getval("_symmetry_space_group_name_H-M"), ['\'', '"'])
        replace(sg, " " => "") == "P1" || throw(
            ArgumentError(
                "space group $sg has no _symmetry_equiv_pos_as_xyz or _space_group_symop_operation_xyz loop to " *
                    "expand to P1, and its name is not literally P1; refusing rather than guessing a symmetry table"
            )
        )
        opstrings = ["x,y,z"]
    end
    ops = parse_symop.(opstrings)
    keys6 = ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
    cell = cell_matrix(ntuple(i -> parse(T, getval(keys6[i])), 6)...)
    hstart = findfirst(==("_atom_site_label"), lines)
    isnothing(hstart) && throw(ArgumentError("CIF has no _atom_site_label loop"))
    cols = String[]
    i = hstart
    while i <= length(lines) && startswith(lines[i], "_atom_site_")
        push!(cols, lines[i]); i += 1
    end
    has_charge_column = "_atom_site_charge" in cols
    isnothing(charges) && !has_charge_column &&
        throw(ArgumentError("CIF has no _atom_site_charge column; partial charges are required (pass `charges` to supply them instead)"))
    col(name) = findfirst(==(name), cols)
    cx, cy, cz, cq = col("_atom_site_fract_x"), col("_atom_site_fract_y"), col("_atom_site_fract_z"), col("_atom_site_charge")
    cl, cs = col("_atom_site_label"), col("_atom_site_type_symbol")
    any(isnothing, (cx, cy, cz, cl, cs)) && throw(ArgumentError("CIF atom_site loop lacks label, type_symbol or fractional coordinates"))
    frac = SVector{3, T}[]; labels = String[]; symbols = String[]; asu_charges = T[]
    while i <= length(lines) && !isempty(lines[i]) && !startswith(lines[i], "_") && !startswith(lines[i], "loop_")
        f = split(lines[i])
        length(f) == length(cols) || throw(ArgumentError("CIF atom row has $(length(f)) fields, header has $(length(cols))"))
        push!(frac, SVector(parse(T, f[cx]), parse(T, f[cy]), parse(T, f[cz])))
        label = String(f[cl])
        push!(labels, label); push!(symbols, String(f[cs]))
        if !isnothing(charges)
            haskey(charges, label) || throw(ArgumentError("read_cif: `charges` has no entry for atom label \"$label\""))
            push!(asu_charges, T(charges[label]))
        else
            push!(asu_charges, parse(T, f[cq]))
        end
        i += 1
    end
    efrac, elabels, esymbols, echarges = expand_symmetry(ops, frac, labels, symbols, asu_charges)
    return Framework{T}(cell, efrac, elabels, esymbols, echarges)
end

"""
    replicate(fw::Framework, n::NTuple{3, Int}) -> Framework

Build the `n[1] × n[2] × n[3]` supercell of `fw`, replicating the unit cell along each lattice
vector and repeating labels, symbols and charges accordingly, and setting `replication` to
`fw.replication .* n`. The result's atoms are, by construction, exact translational copies
under that replication — the property `replication` asserts.
"""
function replicate(fw::Framework{T}, n::NTuple{3, Int}) where {T}
    all(>=(1), n) || throw(ArgumentError("replication factors must be ≥ 1, got $n"))
    N = prod(n)
    frac = Vector{SVector{3, T}}(undef, N * natoms(fw))
    idx = 0
    for k in 0:(n[3] - 1), j in 0:(n[2] - 1), i in 0:(n[1] - 1), f in fw.frac
        idx += 1
        frac[idx] = (f + SVector(i, j, k)) ./ SVector(n)
    end
    return Framework{T}(
        fw.cell * Diagonal(SVector(n)), frac, repeat(fw.labels, N), repeat(fw.symbols, N), repeat(fw.charges, N),
        fw.replication .* n
    )
end
