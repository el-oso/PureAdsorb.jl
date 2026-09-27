# data/raspa2_IRMOF-1.cif is RASPA2's own canonical IRMOF-1 structure (space group #225, F m
# -3 m), symmetry-reduced to 7 asymmetric-unit atoms plus 192 equivalent-position operators.
# PureAdsorb's read_cif accepts only P1 CIFs (src/structure.jl), so this one-off script expands
# the file's own listed operators into the full 424-atom unit cell and writes
# data/IRMOF-1_P1.cif, a plain P1 CIF read_cif can load directly. Every source charge is already
# 0 (RASPA2 ships this file with an all-zero charge column), so the expansion carries that
# column through unchanged rather than assigning charges.
#
# Not part of the package; run once with `julia --project=. bench/irmof1_expand_to_p1.jl` to
# regenerate data/IRMOF-1_P1.cif from data/raspa2_IRMOF-1.cif.

const SRC = joinpath(@__DIR__, "..", "data", "raspa2_IRMOF-1.cif")
const DST = joinpath(@__DIR__, "..", "data", "IRMOF-1_P1.cif")

# Parses one component of a `_symmetry_equiv_pos_as_xyz` operator (e.g. "-x+1/2", "z", "y-1/2")
# into (coefficient, translation): the value is coefficient * v + translation for v in {x, y, z}.
function parse_component(s::AbstractString)
    m = match(r"^([+-]?)([xyz])(?:([+-])(\d+)/(\d+))?$", s)
    isnothing(m) && throw(ArgumentError("unrecognized symmetry-operator component: $s"))
    coeff = m.captures[1] == "-" ? -1.0 : 1.0
    var = only(m.captures[2])
    trans = isnothing(m.captures[3]) ? 0.0 : (m.captures[3] == "-" ? -1.0 : 1.0) * parse(Int, m.captures[4]) / parse(Int, m.captures[5])
    return var, coeff, trans
end

function parse_symop(s::AbstractString)
    parts = split(replace(s, " " => ""), ",")
    length(parts) == 3 || throw(ArgumentError("symmetry operator must have 3 comma-separated components: $s"))
    coeffs = zeros(3, 3)  # coeffs[i, j]: contribution of input coordinate j to output coordinate i
    trans = zeros(3)
    varindex = Dict('x' => 1, 'y' => 2, 'z' => 3)
    for (i, p) in enumerate(parts)
        var, coeff, t = parse_component(p)
        coeffs[i, varindex[var]] = coeff
        trans[i] = t
    end
    return coeffs, trans
end

wrap01(x) = mod(x, 1.0)

function main()
    lines = readlines(SRC)
    getval(key) = split(strip(only(filter(l -> startswith(l, key), lines))))[2]
    a = parse(Float64, getval("_cell_length_a"))
    b = parse(Float64, getval("_cell_length_b"))
    c = parse(Float64, getval("_cell_length_c"))
    a == b == c || throw(ArgumentError("expected a cubic cell, got a=$a b=$b c=$c"))

    opstart = findfirst(==("_symmetry_equiv_pos_as_xyz"), lines)
    opend = findnext(l -> !startswith(l, " '") && !startswith(l, "'"), lines, opstart + 1) - 1
    symops = [parse_symop(strip(l, [' ', '\''])) for l in lines[(opstart + 1):opend]]

    hstart = findfirst(==("_atom_site_label"), lines)
    hend = hstart
    while startswith(lines[hend + 1], "_atom_site_")
        hend += 1
    end
    cols = lines[hstart:hend]
    label_c, sym_c, x_c, y_c, z_c, q_c = (
        findfirst(==(k), cols) for k in
            ("_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_charge")
    )
    asym_end = findnext(isempty, lines, hend + 1) - 1
    asym_atoms = [split(strip(l)) for l in lines[(hend + 1):asym_end] if !isempty(strip(l))]

    tol = 1.0e-3
    kept_frac = NTuple{3, Float64}[]
    kept_symbol = String[]
    kept_charge = Float64[]
    for atom in asym_atoms
        sym = atom[sym_c]
        frac0 = (parse(Float64, atom[x_c]), parse(Float64, atom[y_c]), parse(Float64, atom[z_c]))
        q = parse(Float64, atom[q_c])
        for (coeffs, trans) in symops
            v = coeffs * collect(frac0) .+ trans
            fw = (wrap01(v[1]), wrap01(v[2]), wrap01(v[3]))
            if !any(kept_frac) do g
                    all(k -> min(abs(fw[k] - g[k]), 1 - abs(fw[k] - g[k])) < tol, 1:3)
                end
                push!(kept_frac, fw)
                push!(kept_symbol, sym)
                push!(kept_charge, q)
            end
        end
    end

    counts = Dict{String, Int}()
    open(DST, "w") do io
        println(io, "data_IRMOF-1_P1")
        println(io, "# Expanded to P1 from data/raspa2_IRMOF-1.cif (RASPA2, MIT license) by")
        println(io, "# bench/irmof1_expand_to_p1.jl, using that file's own 192 F m -3 m (#225)")
        println(io, "# symmetry operators. Original structure: Eddaoudi et al., Science 2002,")
        println(io, "# 295, 469 (DOI: 10.1126/science.1067208).")
        println(io, "_symmetry_space_group_name_H-M 'P1'")
        println(io, "_cell_length_a $a")
        println(io, "_cell_length_b $a")
        println(io, "_cell_length_c $a")
        println(io, "_cell_angle_alpha 90")
        println(io, "_cell_angle_beta 90")
        println(io, "_cell_angle_gamma 90")
        println(io)
        println(io, "loop_")
        println(io, "_symmetry_equiv_pos_as_xyz")
        println(io, " 'x,y,z'")
        println(io)
        println(io, "loop_")
        println(io, "_atom_site_label")
        println(io, "_atom_site_type_symbol")
        println(io, "_atom_site_fract_x")
        println(io, "_atom_site_fract_y")
        println(io, "_atom_site_fract_z")
        println(io, "_atom_site_charge")
        for i in eachindex(kept_frac)
            sym = kept_symbol[i]
            counts[sym] = get(counts, sym, 0) + 1
            label = sym * string(counts[sym])
            x, y, z = kept_frac[i]
            println(io, "$label $sym $x $y $z $(kept_charge[i])")
        end
    end
    println("wrote $DST: ", length(kept_frac), " atoms, composition ", counts)
    return counts
end

main()
