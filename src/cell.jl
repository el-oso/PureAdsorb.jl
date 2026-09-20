# Lattice vectors are the columns of the cell matrix; a along x, b in the xy plane.
function cell_matrix(a, b, c, α, β, γ)
    all(>(0), (a, b, c)) || throw(ArgumentError("cell lengths must be positive, got a=$a, b=$b, c=$c"))
    T = float(promote_type(typeof(a), typeof(b), typeof(c), typeof(α), typeof(β), typeof(γ)))
    ca, cb, cg, sg = cosd(T(α)), cosd(T(β)), cosd(T(γ)), sind(T(γ))
    iszero(sg) && throw(ArgumentError("cell angle γ=$γ is degenerate: sin(γ) is zero"))
    cy = (ca - cb * cg) / sg
    radicand = one(T) - cb^2 - cy^2
    radicand > 0 || throw(ArgumentError("cell angles α=$α, β=$β, γ=$γ describe a degenerate cell"))
    cz = sqrt(radicand)
    return SMatrix{3, 3, T}(a, zero(T), zero(T), b * cg, b * sg, zero(T), c * cb, c * cy, c * cz)
end

volume(A::SMatrix{3, 3}) = abs(det(A))

function perpendicular_lengths(A::SMatrix{3, 3})
    V = volume(A)
    a, b, c = A[:, 1], A[:, 2], A[:, 3]
    return SVector(V / norm(cross(b, c)), V / norm(cross(a, c)), V / norm(cross(a, b)))
end

# Smallest supercell in which every pair closer than `cutoff` has a unique nearest image:
# 2·cutoff must fit between opposite faces.
function min_multiplicity(A::SMatrix{3, 3}, cutoff)
    L = perpendicular_lengths(A)
    return ntuple(i -> ceil(Int, 2 * cutoff / L[i]), 3)
end

reciprocal_basis(A::SMatrix{3, 3}) = 2π * inv(A)'

# Rounding in fractional coordinates finds the true nearest image only for separations
# shorter than half the smallest perpendicular length; min_multiplicity guarantees that for
# every separation inside the cutoff, and longer ones are masked out by the caller.
function minimum_image(A::SMatrix{3, 3}, invA::SMatrix{3, 3}, Δ::SVector{3})
    f = invA * Δ
    return Δ - A * round.(f)
end

# Number of cells along one axis for a cell-list grid: at least one, so an axis shorter than
# the target width `w` still gets a single cell spanning it.
grid_dim(L, w) = Int32(max(1, floor(Int, L / w)))
grid_dims(L::SVector{3}, w) = SVector{3, Int32}(ntuple(i -> grid_dim(L[i], w), 3))

# Stencil half-width (in cells) along one axis needed to cover `reach` (Å) from any point in a
# cell, given `n` cells spanning perpendicular length `L`.
stencil_reach(L, n::Int32, reach) = Int32(ceil(Int, reach * n / L))
stencil_reaches(L::SVector{3}, n::SVector{3, Int32}, reach) = SVector{3, Int32}(ntuple(i -> stencil_reach(L[i], n[i], reach), 3))

# 0-based cell coordinate of a fractional coordinate along one axis, wrapped into [0, 1) first:
# `f*n` is truncated toward zero and is always < n by construction (f < 1), so clamping only
# guards f == 1 exactly at the wrap boundary. Shared by `FrameworkBatch`'s cell sort (host,
# `floor`-based) and `insertion_energy`'s stencil (kernel, `unsafe_trunc`-based, no throwing
# `InexactError` branch and no negative intermediate since `f` is pre-wrapped there too).
home_cell(f::T, n::Int32) where {T} = min(Int32(floor(f * n)), n - one(Int32))
home_cell_dev(f::T, n::Int32) where {T} = min(unsafe_trunc(Int32, max(f * n, zero(T))), n - one(Int32))

# 0-based cell coordinate `i, j, k` of a point given its fractional coordinate (wrapped into
# [0,1) first) in a grid of `n` cells per axis.
function cell_coord(f::SVector{3, T}, n::SVector{3, Int32}) where {T}
    fw = mod.(f, one(T))
    return SVector(home_cell(fw[1], n[1]), home_cell(fw[2], n[2]), home_cell(fw[3], n[3]))
end

# Linear cell index (0-based), matching the sort order atoms are stored in: `i + n1*(j + n2*k)`.
cell_linear(i::Int32, j::Int32, k::Int32, n1::Int32, n2::Int32) = i + n1 * (j + n2 * k)
cell_linear(ijk::SVector{3, Int32}, n::SVector{3, Int32}) = cell_linear(ijk[1], ijk[2], ijk[3], n[1], n[2])

# Wraps a 0-based cell coordinate into [0, n), branch-free: valid whenever -n <= x < 2n, which
# holds for `x = start + t` in the stencil loop since `start >= h - m >= -m > -n` there.
wrap_cell(x::Int32, n::Int32) = x + n * Int32(x < 0) - n * Int32(x >= n)

# Start and count (both 0-based) of the stencil's cell range along one axis: `m` cells either
# side of the home cell `h`, or, once `2m+1 >= n`, the whole axis (every cell visited once,
# starting at cell 0 so `wrap_cell` never needs to wrap).
function stencil_start_count(h::Int32, m::Int32, n::Int32)
    span = m + m + one(Int32)
    count = min(span, n)
    start = span >= n ? zero(Int32) : h - m
    return start, count
end

# Stable counting sort of `frac` (per-atom fractional coordinates) into a grid of `n` cells:
# returns the permutation that groups atoms by cell index (original relative order preserved
# within a cell) and the local `prod(n) + 1` cell offsets, `offsets[c+1]` the 0-based start of
# cell `c`'s atoms in the permuted order and `offsets[end]` the atom count. Host-side
# construction only — `floor`/`mod` here are not the throwing-branch concern that applies to
# `insertion_energy`'s kernel code.
function cell_sort(frac::AbstractVector{SVector{3, T}}, n::SVector{3, Int32}) where {T}
    natoms = length(frac)
    ncell = Int(n[1]) * Int(n[2]) * Int(n[3])
    cellidx = Vector{Int32}(undef, natoms)
    counts = zeros(Int32, ncell)
    for a in eachindex(frac)
        c = cell_linear(cell_coord(frac[a], n), n)
        cellidx[a] = c
        counts[c + 1] += one(Int32)
    end
    offsets = zeros(Int32, ncell + 1)
    for c in 1:ncell
        offsets[c + 1] = offsets[c] + counts[c]
    end
    cursor = copy(offsets)
    perm = Vector{Int}(undef, natoms)
    for a in eachindex(cellidx)
        c = cellidx[a]
        cursor[c + 1] += one(Int32)
        perm[cursor[c + 1]] = a
    end
    return perm, offsets
end
