# Lattice vectors are the columns of the cell matrix; a along x, b in the xy plane.
function cell_matrix(a, b, c, α, β, γ)
    T = float(promote_type(typeof(a), typeof(b), typeof(c), typeof(α), typeof(β), typeof(γ)))
    ca, cb, cg, sg = cosd(T(α)), cosd(T(β)), cosd(T(γ)), sind(T(γ))
    cy = (ca - cb * cg) / sg
    cz = sqrt(one(T) - cb^2 - cy^2)
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
