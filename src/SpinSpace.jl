# ============ Helicity space ============
# This file is included by NPHFforFVE.jl; its functions belong directly to the NPHFforFVE module.

"""
    helicity_representatives(representative::NTuple{N, Momentum};
                             species::Vector{Int}, particle_types::Vector{Symbol},
                             spins::Vector{<:Real}, d=Momentum(0,0,0),
                             isospins=nothing, subsystem_isospins=nothing,
                             total_isospin=nothing) -> Vector

Return all independent helicity configurations for a representative momentum.

First validate the zero-momentum/nonzero-spin constraint, then compute the Per({n⃗}) little group,
construct helicity equivalence relations, and return lexicographically ordered class representatives.

# Arguments
- `representative`: N-particle representative momentum state
- `species`: particle counts per species, for example `[1,2]`
- `particle_types`: particle type of each species (`:distinguishable`, `:boson`, `:fermion`)
- `spins`: spin quantum number of each species; supports integers and half-integers (`Rational{Int}` or `.5`).
- `d`: total momentum
- `isospins`: single-particle isospins per species (optional), with the same length as species, for example `[1//2, 1//2]`
- `subsystem_isospins`: coupled total isospin of each subsystem (optional), with the same length as species, for example `[0, 1]`
- `total_isospin`: total isospin of the full system (optional)

The three isospin arguments must either all be `nothing` or all be supplied.

# Returns
list of helicity tuples (expanded to N tuples per particle), with λᵢ ∈ {-s, -s+1, ..., s}
"""
function helicity_representatives(representative::NTuple{N, Momentum};
                                  species::Vector{Int}, particle_types::Vector{Symbol},
                                  spins::Vector{<:Real}, d=Momentum(0,0,0),
                                  isospins=nothing, subsystem_isospins=nothing,
                                  total_isospin=nothing) where N
    _validate_helicity_inputs(representative, species, particle_types, spins)
    _validate_isospin_inputs(species, isospins, subsystem_isospins, total_isospin)

    _, group_name = group_for_momentum(d)
    group = group_elements(group_name)

    per_generators = _compute_per(representative, group, species, particle_types)
    all_configs = _all_helicity_configs(spins, species)

    if isempty(per_generators) || isempty(all_configs)
        return all_configs
    end

    return _find_equivalence_representatives(all_configs, per_generators)
end

# ============ Input validation ============

function _validate_helicity_inputs(representative, species, particle_types, spins)
    n_particles = length(representative)
    if sum(species) != n_particles
        throw(ArgumentError("sum(species) ($(sum(species))) must equal the number of particles in the representative momentum ($n_particles)"))
    end
    if length(species) != length(particle_types)
        throw(ArgumentError("particle_types length ($(length(particle_types))) must match species length ($(length(species)))"))
    end
    if length(spins) != length(species)
        throw(ArgumentError("spins length ($(length(spins))) must match species length ($(length(species)))"))
    end

    all_spins = _expand_spins(spins, species)
    for i in 1:n_particles
        s = all_spins[i]
        n = representative[i]
        if s != 0 && n == Momentum(0, 0, 0)
            throw(ArgumentError("particle $i has spin s=$s ≠ 0 but zero momentum, where helicity is undefined"))
        end
    end
end

# ============ Isospin-argument validation ============

function _validate_isospin_inputs(species::Vector{Int}, isospins, subsystem_isospins, total_isospin)
    any_provided = isospins !== nothing || subsystem_isospins !== nothing || total_isospin !== nothing
    if !any_provided
        return nothing
    end

    if isospins === nothing || subsystem_isospins === nothing || total_isospin === nothing
        throw(ArgumentError("isospins, subsystem_isospins, and total_isospin must either all be provided or all be omitted"))
    end

    if length(isospins) != length(species)
        throw(ArgumentError("isospins length ($(length(isospins))) must match species length ($(length(species)))"))
    end
    if length(subsystem_isospins) != length(species)
        throw(ArgumentError("subsystem_isospins length ($(length(subsystem_isospins))) must match species length ($(length(species)))"))
    end

    for (idx, val) in enumerate(isospins)
        _validate_isospin_value(val, "isospins[$idx]")
    end
    for (idx, val) in enumerate(subsystem_isospins)
        _validate_isospin_value(val, "subsystem_isospins[$idx]")
    end
    _validate_isospin_value(total_isospin, "total_isospin")
    return nothing
end

function _validate_isospin_value(val, label::String)
    if val < 0
        throw(ArgumentError("$label = $val < 0; isospin must be nonnegative"))
    end
    twice = 2 * val
    if !isinteger(twice)
        throw(ArgumentError("$label = $val is not an integer or half-integer"))
    end
    return nothing
end

# ============ Per({n⃗}) calculation ============

"""
    _compute_per(representative, group, spec_sizes, spec_types)

Return the generators of the equivalence relation. Each generator is (parity::Int, inv_perm::Vector{Int})，
parity = 1(proper) or -1 (improper); inv_perm is the global inverse permutation (1-indexed).

For each group element g, check whether the momentum multiset of every species subsystem is invariant under g.
If so, enumerate all allowed permutations and collect them as generators.
"""
function _compute_per(representative::NTuple{N, Momentum}, group,
                      spec_sizes::Vector{Int}, spec_types) where N
    generators = Tuple{Int, Vector{Int}}[]

    for (g_idx, g) in enumerate(group)
        parity = _parity_of(g_idx, length(group))
        transformed = [apply_transform(g, representative[i]) for i in 1:N]

        all_perms_per_species = Vector{Vector{Vector{Int}}}()
        offsets = Int[]
        ok = true
        offset = 1
        for sz in spec_sizes
            orig_block = Momentum[representative[i] for i in offset:offset+sz-1]
            trans_block = Momentum[transformed[i] for i in offset:offset+sz-1]
            if !_multiset_equal(orig_block, trans_block)
                ok = false
                break
            end
            perms = _find_all_permutations(orig_block, trans_block)
            if isempty(perms)
                ok = false
                break
            end
            push!(all_perms_per_species, perms)
            push!(offsets, offset)
            offset += sz
        end

        ok || continue

        # Enumerate all species-permutation combinations and construct global permutations.
        for combo in _cartesian_product(all_perms_per_species)
            # combo is the flattened concatenation of local permutations for all species subsystems.
            # Add each species offset to obtain global indices.
            global_perm = Vector{Int}(undef, N)
            pos = 1
            for (s, sz) in enumerate(spec_sizes)
                base = offsets[s] - 1
                for local_idx in 1:sz
                    # combo[pos] is the local target index (1..sz); convert it to a global index.
                    global_perm[base + local_idx] = base + combo[pos]
                    pos += 1
                end
            end
            inv_perm = _inverse_permutation(global_perm)
            push!(generators, (parity, inv_perm))
        end
    end

    return _deduplicate_generators(generators)
end

# ============ Permutation utilities ============

# Backtrack over all bijective permutations orig[i] → trans (every returned p satisfies trans[i] == orig[p[i]]).
function _find_all_permutations(orig::Vector{Momentum}, trans::Vector{Momentum})
    k = length(orig)
    used = falses(k)
    current = Vector{Int}(undef, k)
    results = Vector{Int}[]

    function backtrack(i::Int)
        if i > k
            push!(results, copy(current))
            return
        end
        target = trans[i]
        for j in 1:k
            if !used[j] && orig[j] == target
                used[j] = true
                current[i] = j
                backtrack(i + 1)
                used[j] = false
            end
        end
    end

    backtrack(1)
    return results
end

function _multiset_equal(a::Vector{Momentum}, b::Vector{Momentum})
    return sort(a) == sort(b)
end

function _inverse_permutation(p::Vector{Int})
    inv_p = Vector{Int}(undef, length(p))
    for i in 1:length(p)
        inv_p[p[i]] = i
    end
    return inv_p
end

function _cartesian_product(vectors::Vector{Vector{Vector{Int}}})
    if isempty(vectors)
        return [Vector{Int}[]]
    end
    result = [Int[]]
    for vec in vectors
        new_result = Vector{Int}[]
        for r in result, v in vec
            push!(new_result, vcat(r, v))
        end
        result = new_result
    end
    return result
end

function _deduplicate_generators(generators)
    seen = Set{Tuple{Int, Vector{Int}}}()
    unique_gens = Tuple{Int, Vector{Int}}[]
    for gen in generators
        gen in seen && continue
        push!(seen, gen)
        push!(unique_gens, gen)
    end
    return unique_gens
end

# ============ Helicity-configuration generation ============

function _expand_spins(spins::Vector{<:Real}, spec_sizes::Vector{Int})
    result = Real[]
    for (s, sz) in zip(spins, spec_sizes)
        append!(result, fill(s, sz))
    end
    return result
end

function _all_helicity_configs(spins::Vector{<:Real}, spec_sizes::Vector{Int})
    all_spins = _expand_spins(spins, spec_sizes)
    ranges = [_helicity_values(s) for s in all_spins]
    return _helicity_cartesian_product(ranges)
end

# Possible helicity values for s: (-s, -s+1, ..., s)，stored in the most specific type.
function _helicity_values(s::Real)
    if isinteger(s)
        s_int = Int(s)
        return collect(Int, -s_int:s_int)
    else
        s_rat = Rational{Int}(s)
        den = denominator(s_rat)
        if den != 2
            throw(ArgumentError("spin $s is not an integer or half-integer"))
        end
        num = numerator(s_rat)
        return Rational{Int}[Rational{Int}(λ_num, 2) for λ_num in -num:2:num]
    end
end

function _helicity_cartesian_product(ranges::Vector{<:Vector})
    if isempty(ranges)
        return [()]
    end
    result = [()]
    for r in ranges
        new_result = []
        for prev in result, val in r
            push!(new_result, (prev..., val))
        end
        result = new_result
    end
    return result
end

# ============ Equivalence relations and union-find ============

struct _UnionFind
    parent::Vector{Int}
    rank::Vector{Int}
end

function _UnionFind(n::Int)
    return _UnionFind(collect(1:n), zeros(Int, n))
end

function _find(uf::_UnionFind, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]
        x = uf.parent[x]
    end
    return x
end

function _union!(uf::_UnionFind, x::Int, y::Int)
    rx = _find(uf, x)
    ry = _find(uf, y)
    rx == ry && return
    if uf.rank[rx] < uf.rank[ry]
        uf.parent[rx] = ry
    elseif uf.rank[rx] > uf.rank[ry]
        uf.parent[ry] = rx
    else
        uf.parent[ry] = rx
        uf.rank[rx] += 1
    end
end

# λ_new[i] = parity * λ_old[inv_perm[i]]
function _apply_helicity_equivalence(config, gen::Tuple{Int, Vector{Int}})
    parity, inv_perm = gen
    N = length(config)
    result = similar(collect(config))
    for i in 1:N
        result[i] = parity * config[inv_perm[i]]
    end
    return Tuple(result)
end

function _find_equivalence_representatives(all_configs::Vector, generators)
    n = length(all_configs)
    uf = _UnionFind(n)
    config_to_idx = Dict(config => idx for (idx, config) in enumerate(all_configs))

    for gen in generators
        for (idx, config) in enumerate(all_configs)
            target = _apply_helicity_equivalence(config, gen)
            target_idx = config_to_idx[target]
            _union!(uf, idx, target_idx)
        end
    end

    # Take the lexicographically smallest member of each equivalence class.
    class_best = Dict{Int, Any}()  # root → minimal config
    for (idx, config) in enumerate(all_configs)
        root = _find(uf, idx)
        if !haskey(class_best, root) || _lex_less(config, class_best[root])
            class_best[root] = config
        end
    end

    reps = collect(values(class_best))
    sort!(reps, lt=_lex_less)
    return reps
end

function _lex_less(a, b)
    for i in 1:length(a)
        if a[i] < b[i]
            return true
        elseif a[i] > b[i]
            return false
        end
    end
    return false
end

# ============ Group-element parity ============

"""
    _parity_of(g_idx::Int, n_bosonic::Int) -> Int

Determine parity from the group-element index. Convention: group elements are ordered as [proper rotations ..., improper rotations ...],
the first n_bosonic/2 are proper rotations (parity=+1) and the last n_bosonic/2 are improper rotations (parity=-1).
For double-cover groups, this pattern extends periodically with n_bosonic.
"""
function _parity_of(g_idx::Int, n_bosonic::Int)
    return ((g_idx - 1) % n_bosonic) < div(n_bosonic, 2) ? 1 : -1
end

# ============ Wigner angles and helicity phases ============

"""
    _momentum_to_euler(n::Momentum) -> (θ::Float64, φ::Float64)

Compute Euler angles of the standard rotation taking the z axis to momentum direction n R_st(n) = e^{-iφ J_z} e^{-iθ J_y}  .

Convention: 0 ≤ θ ≤ π, -π ≤ φ < π。
When n ∥ the z axis: n_z > 0 → φ = 0; n_z < 0 → φ = -π。
"""
function _momentum_to_euler(n::Momentum)
    nx, ny, nz = n[1], n[2], n[3]
    r2 = nx*nx + ny*ny + nz*nz
    if r2 == 0
        return 0.0, 0.0
    end
    r = sqrt(Float64(r2))
    theta = acos(nz / r)
    if nx == 0 && ny == 0
        phi = nz > 0 ? 0.0 : -pi
    else
        phi = atan(ny, nx)
        # atan Returns (-π, π]，normalize to [-π, π)
        if phi >= pi - 1e-15
            phi = -pi
        end
    end
    return theta, phi
end

"""
    _su2_standard_rotation(theta, phi) -> Matrix{ComplexF64}

Return D^{1/2}(R_st(n)) = exp(-i φ J_z) exp(-i θ J_y), a 2×2 matrix.
"""
function _su2_standard_rotation(theta::Float64, phi::Float64)
    ct = cos(theta / 2)
    st = sin(theta / 2)
    cp = cos(phi / 2)
    sp = sin(phi / 2)
    e_neg = ComplexF64(cp, -sp)   # e^{-iφ/2}
    e_pos = ComplexF64(cp,  sp)   # e^{+iφ/2}
    return ComplexF64[
        e_neg * ct   -e_neg * st
        e_pos * st    e_pos * ct
    ]
end

"""
    _su2_standard_rotation_inv(theta, phi) -> Matrix{ComplexF64}

Returns D^{1/2}(R_st(n))^{-1}，namely the Hermitian conjugate above。
"""
function _su2_standard_rotation_inv(theta::Float64, phi::Float64)
    # D^{-1} = D† for SU(2)
    return _su2_standard_rotation(theta, phi)'
end

"""
    _so3_to_su2(R::SMatrix{3,3,Int}, sign::Int=1) -> Matrix{ComplexF64}

Lift an SO(3) rotation matrix to its spin-1/2 SU(2) representation.
Use the axis-angle parameters (m, ω) in _OH_ROTATION_PARAMS:
  D^{1/2}(g) = cos(ω/2) I - i sin(ω/2) (m·σ)

sign = ±1 distinguishes the two SU(2) lifts of a double-cover group.
"""
function _so3_to_su2(R::SMatrix{3,3,Int}, sign::Int=1)
    return sign * SymmetryGroup._OH_PROPER_SU2[R]
end

struct _WignerAngleKey
    momentum::Momentum
    rotation_code::UInt16
    sign::Int
end

@inline function _rotation_cache_code(g::SMatrix{3,3,Int})
    code = 0
    factor = 1
    for value in g
        code += (value + 1) * factor
        factor *= 3
    end
    return UInt16(code)
end

const _WIGNER_ANGLE_CACHE = Dict{_WignerAngleKey, Float64}()
const _WIGNER_ANGLE_CACHE_LOCK = ReentrantLock()

"""Clear the internal Wigner-angle cache. Primarily useful for tests/profiling."""
function _clear_wigner_angle_cache!()
    lock(_WIGNER_ANGLE_CACHE_LOCK) do
        empty!(_WIGNER_ANGLE_CACHE)
    end
    return nothing
end

function _wigner_angle_cache_size()
    return lock(_WIGNER_ANGLE_CACHE_LOCK) do
        length(_WIGNER_ANGLE_CACHE)
    end
end

"""
    _compute_wigner_angle(n, g, sign) -> Float64

Uncached reference implementation used by `_wigner_angle` and correctness
tests. `g` must be a proper rotation.
"""
function _compute_wigner_angle(n::Momentum, g::SMatrix{3,3,Int}, sign::Int)
    gn = apply_transform(g, n)

    θ_n,  φ_n  = _momentum_to_euler(n)
    θ_gn, φ_gn = _momentum_to_euler(gn)

    D_n = _su2_standard_rotation(θ_n, φ_n)
    D_gn_inv = _su2_standard_rotation_inv(θ_gn, φ_gn)
    D_g = _so3_to_su2(g, sign)

    U = D_gn_inv * D_g * D_n

    # U should diagonalize to [[e^{-iφ_w/2}, 0], [0, e^{+iφ_w/2}]]
    # atan(imag, real) ∈ (-π, π]，retain the 4π periodicity required by SU(2).
    return -2 * atan(imag(U[1,1]), real(U[1,1]))
end

"""
    _wigner_angle(n::Momentum, g::SMatrix{3,3,Int}, sign::Int=1) -> Float64

Compute the Wigner rotation angle φ_w(n, g), defined by
D^{1/2}(e^{-iφ_w J_z}) = D^{1/2}(R_st^{-1}(gn) g R_st(n))  .

The return value lies in (-π, π].

sign = ±1 distinguishes the two SU(2) lifts of a double-cover group.
Applicable only to proper rotations（det(g) = +1）。
"""
function _wigner_angle(n::Momentum, g::SMatrix{3,3,Int}, sign::Int=1)
    key = _WignerAngleKey(n, _rotation_cache_code(g), sign)
    cached = lock(_WIGNER_ANGLE_CACHE_LOCK) do
        get(_WIGNER_ANGLE_CACHE, key, nothing)
    end
    cached === nothing || return cached

    # Compute outside the lock. Concurrent first misses may duplicate this
    # small calculation, but cache access remains safe and threads are not
    # serialized around the expensive trigonometry/matrix products.
    angle = _compute_wigner_angle(n, g, sign)
    return lock(_WIGNER_ANGLE_CACHE_LOCK) do
        get!(_WIGNER_ANGLE_CACHE, key, angle)
    end
end

"""
    helicity_phase(n::Momentum, g::SMatrix{3,3,Int}, lambda::Real, sign::Int=1) -> ComplexF64

Compute the phase e^{-iλ φ_w(n, g)} of a proper rotation g acting on a helicity state. e^{-iλ φ_w(n, g)}。

sign = ±1 distinguishes the two SU(2) lifts of a double-cover group.
Applicable only to proper rotations。
"""
function helicity_phase(n::Momentum, g::SMatrix{3,3,Int}, lambda::Real, sign::Int=1)
    φ_w = _wigner_angle(n, g, sign)
    return exp(-im * Float64(lambda) * φ_w)
end

"""
    _parity_helicity_phase(n::Momentum, lambda::Real, s::Real, eta::Real) -> ComplexF64

Compute the phase of spatial reflection P acting on a helicity state:
P |n, λ⟩ = η e^{∓iπs} |-n, -λ⟩

with exponent-sign convention:
  -π < φ_n < 0  → e^{-iπs}
   0 ≤ φ_n < π  → e^{+iπs}

η is the particle intrinsic parity supplied by the user.
"""
function _parity_helicity_phase(n::Momentum, lambda::Real, s::Real, eta::Real)
    _, phi = _momentum_to_euler(n)
    sign_phase = (0.0 <= phi < pi) ? 1.0 : -1.0
    return Float64(eta) * exp(sign_phase * im * pi * Float64(s))
end
