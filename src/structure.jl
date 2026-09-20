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

"""
    read_cif(path; T = Float64) -> Framework{T}

Read a minimal CIF: one data block, space group P1 only, one `_atom_site` loop with fractional
coordinates and an `_atom_site_charge` column (partial charges, in e, are required). Anything
else in the file is rejected rather than guessed.
"""
function read_cif(path::AbstractString; T = Float64)
    lines = strip.(readlines(path))
    getval(key) = begin
        i = findfirst(l -> startswith(l, key * " ") || startswith(l, key * "\t"), lines)
        isnothing(i) && throw(ArgumentError("CIF is missing $key"))
        strip(lines[i][(length(key) + 1):end])
    end
    sg = strip(getval("_symmetry_space_group_name_H-M"), ['\'', '"'])
    replace(sg, " " => "") == "P1" || throw(ArgumentError("only P1 CIF files are supported, got space group $sg"))
    keys6 = ("_cell_length_a", "_cell_length_b", "_cell_length_c", "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
    cell = cell_matrix(ntuple(i -> parse(T, getval(keys6[i])), 6)...)
    hstart = findfirst(==("_atom_site_label"), lines)
    isnothing(hstart) && throw(ArgumentError("CIF has no _atom_site_label loop"))
    cols = String[]
    i = hstart
    while i <= length(lines) && startswith(lines[i], "_atom_site_")
        push!(cols, lines[i]); i += 1
    end
    "_atom_site_charge" in cols || throw(ArgumentError("CIF has no _atom_site_charge column; partial charges are required"))
    col(name) = findfirst(==(name), cols)
    cx, cy, cz, cq = col("_atom_site_fract_x"), col("_atom_site_fract_y"), col("_atom_site_fract_z"), col("_atom_site_charge")
    cl, cs = col("_atom_site_label"), col("_atom_site_type_symbol")
    any(isnothing, (cx, cy, cz, cl, cs)) && throw(ArgumentError("CIF atom_site loop lacks label, type_symbol or fractional coordinates"))
    frac = SVector{3, T}[]; labels = String[]; symbols = String[]; charges = T[]
    while i <= length(lines) && !isempty(lines[i]) && !startswith(lines[i], "_") && !startswith(lines[i], "loop_")
        f = split(lines[i])
        length(f) == length(cols) || throw(ArgumentError("CIF atom row has $(length(f)) fields, header has $(length(cols))"))
        push!(frac, SVector(parse(T, f[cx]), parse(T, f[cy]), parse(T, f[cz])))
        push!(labels, String(f[cl])); push!(symbols, String(f[cs])); push!(charges, parse(T, f[cq]))
        i += 1
    end
    return Framework{T}(cell, frac, labels, symbols, charges)
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
