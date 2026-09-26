# Counter-based per-move random numbers, safe to call from inside a KernelAbstractions kernel on
# any backend. `ChainRNG` draws depend only on its own construction arguments, never on execution
# order, so the same `(chain_seed, cycle, move)` triple draws identical numbers no matter how many
# chains share a batch or how a run of moves is split across kernel launches.

"""
    mulhi64(a::UInt64, b::UInt64) -> UInt64

The high 64 bits of the full 128-bit product `a * b`, from four 32-bit partial products rather
than a widening `UInt128` multiply: `UInt128` is not supported on every backend this generator
targets (Metal has no native 128-bit integer type), while 64-bit multiplication and addition are
universal.
"""
function mulhi64(a::UInt64, b::UInt64)
    a_lo, a_hi = a & 0xffffffff, a >> 32
    b_lo, b_hi = b & 0xffffffff, b >> 32
    lo_lo = a_lo * b_lo
    hi_lo = a_hi * b_lo
    lo_hi = a_lo * b_hi
    hi_hi = a_hi * b_hi
    cross = (lo_lo >> 32) + (hi_lo & 0xffffffff) + (lo_hi & 0xffffffff)
    return hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (cross >> 32)
end

"""
    ChainRNG(chain_seed::Integer, cycle::Integer, move::Integer)

Counter-based generator for one Monte Carlo chain's one move. The whole stream is keyed by
`(chain_seed, cycle, move)`, folded through `splitmix64` (`src/state.jl`) into a single 64-bit
key, so two calls with identical arguments draw identical numbers regardless of which chain,
cycle, or move ran before them, how many chains share a batch, or how the work was split across
kernel launches — reproducibility follows from the key alone, never from a shared running state.

`chain_seed` is `SystemState`'s own `rng_seed[n]` (`src/state.jl`), itself `splitmix64` of a
global seed and the chain index. `ChainRNG` never sees the global seed or the chain index
directly, only that already-mixed combination, so the four-way key `(seed, chain, cycle, move)`
is exactly `splitmix64(splitmix64(chain_seed, cycle), move)`.

Successive draws from one `ChainRNG` (`next_bits`) advance an internal word counter starting at
`0`; the counter is not carried between moves, since each `(chain_seed, cycle, move)` triple
starts its own fresh stream rather than continuing a single per-chain sequence.
"""
struct ChainRNG
    key::UInt64
    draw::UInt64
end

ChainRNG(chain_seed::Integer, cycle::Integer, move::Integer) =
    ChainRNG(splitmix64(splitmix64(UInt64(chain_seed), UInt64(cycle)), UInt64(move)), zero(UInt64))

"""
    next_bits(rng::ChainRNG) -> (UInt64, ChainRNG)

The next 64 pseudorandom bits from `rng`'s stream, and the state advanced by one word. Every
higher-level draw below (`rand_uniform`, `rand_range`, `rand_normal`, `rand_quaternion`) is built
from this.
"""
function next_bits(rng::ChainRNG)
    raw = splitmix64(rng.key, rng.draw)
    return raw, ChainRNG(rng.key, rng.draw + one(UInt64))
end

"""
    unit_from_bits(raw::UInt64, ::Type{F}) -> F

`raw`'s top `p` bits, `p` the significand width of `F` (24 for `Float32`, 53 for `Float64`),
scaled into `[0, 1)`. Every integer in `0:2^p - 1` is exactly representable in `F` and equally
likely, so the result is uniform to `F`'s own precision.
"""
unit_from_bits(raw::UInt64, ::Type{Float64}) = Float64(raw >> 11) * 1.1102230246251565e-16
unit_from_bits(raw::UInt64, ::Type{Float32}) = Float32(raw >> 40) * 5.9604645f-8

"""
    rand_uniform(rng::ChainRNG, ::Type{F}) -> (F, ChainRNG)

Uniform `F` in `[0, 1)`, drawn from the next word of `rng`'s stream.
"""
function rand_uniform(rng::ChainRNG, ::Type{F}) where {F}
    raw, rng2 = next_bits(rng)
    return unit_from_bits(raw, F), rng2
end

"""
    rand_range(rng::ChainRNG, n::T) where {T<:Integer} -> (T, ChainRNG)

Uniform integer in `1:n`, `n >= 1`, drawn from the next word of `rng`'s stream. Computed by
Lemire's multiply-shift map (`mulhi64`) rather than `raw % n`: `%`/`div`/`mod` by a runtime value
lowers to an integer division, which every GPU backend this generator targets executes far more
slowly than a multiply, and which some do not implement at all for 64-bit operands. The map's
bias toward the low end of the range is at most `n / 2^64`, unmeasurable for any `n` this project
draws (a guest count, or a handful of move types).
"""
function rand_range(rng::ChainRNG, n::T) where {T <: Integer}
    raw, rng2 = next_bits(rng)
    idx = (mulhi64(raw, UInt64(n)) % T) + one(T)
    return idx, rng2
end

"""
    rand_normal(rng::ChainRNG, ::Type{F}) -> (F, ChainRNG)

Standard normal `F` (mean 0, variance 1), via the Box–Muller transform on two uniform draws. `u1`
is taken as `1 - rand_uniform` rather than `rand_uniform` directly so it lands in `(0, 1]`:
`rand_uniform` can return exactly `0`, and `log(0) = -Inf` would make `sqrt(-2·log(u1))` diverge
to `Inf` instead of yielding a merely-improbable-but-finite sample. Only the cosine branch is
returned; the companion, equally valid `sin` branch is discarded, keeping every draw in this file
a single value plus the advanced state.
"""
function rand_normal(rng::ChainRNG, ::Type{F}) where {F}
    u1, rng1 = rand_uniform(rng, F)
    u2, rng2 = rand_uniform(rng1, F)
    r = sqrt(-2 * log(one(F) - u1))
    θ = F(2π) * u2
    return r * cos(θ), rng2
end

"""
    shoemake_quaternion(u1::F, u2::F, u3::F) -> SVector{4, F}

The Shoemake (1992) uniform-on-SO(3) unit quaternion, `(x, y, z, w)` order, from three explicit
uniforms in `[0, 1)`. Same construction and convention as `shoemake_quaternion(rng::AbstractRNG,
::Type{T})` (`src/widom.jl`), but computed entirely in `F` rather than through an `AbstractRNG`
call (which draws `Float64`s regardless of `T`): a kernel must stay in one float type throughout,
since Metal has no double-precision unit at all.
"""
function shoemake_quaternion(u1::F, u2::F, u3::F) where {F}
    a, b = sqrt(one(F) - u1), sqrt(u1)
    return SVector{4, F}(a * cospi(2u2), b * sinpi(2u3), b * cospi(2u3), a * sinpi(2u2))
end

"""
    rand_quaternion(rng::ChainRNG, ::Type{F}) -> (SVector{4, F}, ChainRNG)

Uniform random unit quaternion on SO(3), via `shoemake_quaternion` on three uniform draws.
"""
function rand_quaternion(rng::ChainRNG, ::Type{F}) where {F}
    u1, rng1 = rand_uniform(rng, F)
    u2, rng2 = rand_uniform(rng1, F)
    u3, rng3 = rand_uniform(rng2, F)
    return shoemake_quaternion(u1, u2, u3), rng3
end

"""
    rng_draw_kernel!(u, ranged, z, quat, rng_seed, cycles, moves, nrange)

One work-item per entry: draws a uniform `u[i]`, a ranged integer `ranged[i] in 1:nrange[i]`, a
normal `z[i]`, and a quaternion `quat[i]`, all from `ChainRNG(rng_seed[i], cycles[i], moves[i])`.
Exercises every draw this file provides from inside a real `KernelAbstractions` kernel, so the
same code path runs unchanged on CPU and on every GPU backend.
"""
@kernel function rng_draw_kernel!(u, ranged, z, quat, @Const(rng_seed), @Const(cycles), @Const(moves), @Const(nrange))
    i = @index(Global)
    rng = ChainRNG(rng_seed[i], cycles[i], moves[i])
    uval, rng = rand_uniform(rng, eltype(u))
    rval, rng = rand_range(rng, nrange[i])
    zval, rng = rand_normal(rng, eltype(z))
    qval, rng = rand_quaternion(rng, eltype(u))
    u[i] = uval
    ranged[i] = rval
    z[i] = zval
    quat[i] = qval
end
