# ============================================================
# HelicityRotation — Helicity-basis ↔ canonical-polarization-basis rotations
# ============================================================
# Wigner D-matrix implementation, with public support for j = 0, 1/2, 1, 3/2.
# D^j(R_st(n)): Standard rotation taking ẑ to n̂.
# ============================================================

const _D_CACHE = Dict{Tuple{Momentum, Rational{Int}}, Matrix{ComplexF64}}()
const _ROT_VEC_CACHE = Dict{NamedTuple, Vector{ComplexF64}}()
const _SPIN_PROJ_CACHE = Dict{Rational{Int}, Vector{Rational{Int}}}()
const _M_VALS_CACHE = Dict{Tuple, Vector{Vector{Float64}}}()

# A CSC sparse matrix is advantageous only when a helicity block has a
# sufficiently small fill fraction. Dense blocks are faster to build and to
# multiply by projection matrices when most entries are nonzero.
const _V_HEL_DENSE_FILL_THRESHOLD = 0.20

"""
    _finalize_V_hel_block(m, n, triplets) -> AbstractMatrix{ComplexF64}

Store a completed helicity block densely when its nonzero fill fraction is at
least `_V_HEL_DENSE_FILL_THRESHOLD`; otherwise retain the existing CSC sparse
representation. This routine changes only storage, not any matrix element.
Repeated triplets are accumulated in the same way as `sparse`.
"""
function _finalize_V_hel_block(m::Int, n::Int,
                               triplets::Vector{Tuple{Int,Int,ComplexF64}})
    isempty(triplets) && return spzeros(ComplexF64, m, n)

    fill_fraction = length(triplets) / (Float64(m) * Float64(n))
    if fill_fraction >= _V_HEL_DENSE_FILL_THRESHOLD
        block = zeros(ComplexF64, m, n)
        @inbounds for (i, j, value) in triplets
            block[i, j] += value
        end
        return block
    end

    I = Int[t[1] for t in triplets]
    J = Int[t[2] for t in triplets]
    V = ComplexF64[t[3] for t in triplets]
    return sparse(I, J, V, m, n)
end

# ============ Spherical coordinates ============

function _sph_coords(n::Momentum)
    r = sqrt(Float64(sum(abs2, n)))
    if r == 0.0
        return 1.0, 0.0, 1.0, 0.0  # θ=0, φ=0: cosθ=1
    end
    cosθ = n[3] / r
    sinθ = sqrt(n[1]^2 + n[2]^2) / r
    if sinθ == 0.0
        # φ convention: n_z > 0 → φ = 0; n_z < 0 → φ = -π (consistent with _momentum_to_euler).
        if cosθ < 0.0
            return cosθ, sinθ, -1.0, 0.0  # θ=π, φ=-π
        else
            return cosθ, sinθ, 1.0, 0.0   # θ=0, φ=0
        end
    end
    cosφ = n[1] / (r * sinθ)
    sinφ = n[2] / (r * sinθ)
    return cosθ, sinθ, cosφ, sinφ
end

# ============ Wigner small-d matrix ============

function _wigner_small_d(j::Rational{Int}, cosθ::Float64, sinθ::Float64)
    j >= 0 && denominator(j) in (1, 2) ||
        throw(ArgumentError("spin j=$j must be a nonnegative integer or half-integer"))

    dim = Int(2j + 1)
    m_values = Rational{Int}[j - i for i in 0:(dim - 1)]
    c = sqrt(max(0.0, (1.0 + clamp(cosθ, -1.0, 1.0)) / 2.0))
    s = copysign(sqrt(max(0.0, (1.0 - clamp(cosθ, -1.0, 1.0)) / 2.0)),
                 sinθ)
    result = zeros(ComplexF64, dim, dim)

    # Condon–Shortley convention, with rows/columns ordered as
    # m = j, j-1, ..., -j. This reproduces the previous closed forms for
    # j=0, 1/2, 1 and extends the same convention to j=3/2.
    for (row, mprime) in enumerate(m_values), (col, m) in enumerate(m_values)
        prefactor = sqrt(Float64(
            factorial(Int(j + mprime)) * factorial(Int(j - mprime)) *
            factorial(Int(j + m)) * factorial(Int(j - m))))
        kmin = max(0, Int(m - mprime))
        kmax = min(Int(j + m), Int(j - mprime))
        value = 0.0
        for k in kmin:kmax
            denominator_value =
                factorial(Int(j + m - k)) * factorial(k) *
                factorial(Int(mprime - m + k)) *
                factorial(Int(j - mprime - k))
            sign = isodd(Int(mprime - m + k)) ? -1.0 : 1.0
            cpower = Int(2j + m - mprime - 2k)
            spower = Int(mprime - m + 2k)
            value += sign * prefactor / denominator_value *
                     c^cpower * s^spower
        end
        result[row, col] = value
    end
    return result
end

# ============ Wigner D matrix ============

"""
    wigner_D(j::Rational{Int}, n::Momentum) -> Matrix{ComplexF64}

Return D^j(R_st(n)), the Wigner D matrix of the standard rotation taking ẑ to n̂.
D^j_{m,m'}(φ,θ,0) = e^{-imφ} d^j_{m,m'}(θ)。
At zero momentum, return the identity matrix.
"""
function wigner_D(j::Rational{Int}, n::Momentum)
    key = (n, j)
    get!(_D_CACHE, key) do
        _compute_wigner_D(j, n)
    end
end

function _compute_wigner_D(j::Rational{Int}, n::Momentum)
    cosθ, sinθ, cosφ, sinφ = _sph_coords(n)
    dmat = _wigner_small_d(j, cosθ, sinθ)
    dim = size(dmat, 1)
    phi = atan(sinφ, cosφ)
    # Normalize to [-π, π) to match _momentum_to_euler convention
    if phi >= pi - 1e-15
        phi = -pi
    end
    # m = j, j-1, ..., -j  (top to bottom: rows 1,2,...,dim)
    D = similar(dmat)
    for (i, m) in enumerate(Float64(j):-1:(-Float64(j)))
        phase = exp(ComplexF64(0.0, -m * phi))
        for k in 1:dim
            D[i, k] = phase * dmat[i, k]
        end
    end
    return D
end

# ============ Helicity → canonical-polarization rotation-coefficient vector ============

"""
    get_rotation_vector(n_tuple, lambda_tuple, per_spin) -> Vector{ComplexF64}

Return the rotation-coefficient vector c_σ for one subspace state `(n_tuple, λ_tuple)`.
c_σ = ∏_i D^{j_i}_{σ_i, λ_i}(R_st(n_i))

per_spin is the list of per-particle spins (length N; entries are Rational{Int}).
"""
function get_rotation_vector(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             per_spin::AbstractVector{<:Real}) where N
    lam_r = Rational{Int}.(lambda_tuple)
    spin_r = Rational{Int}.(per_spin)
    key = (n_tuple=n_tuple, lambda_tuple=Tuple(lam_r),
           per_spin=Tuple(spin_r))
    return get!(_ROT_VEC_CACHE, key) do
        _compute_rotation_vector(n_tuple, lam_r, spin_r)
    end
end

function _compute_rotation_vector(n_tuple::NTuple{N, Momentum},
                                  lambda_tuple::NTuple{N, Rational{Int}},
                                  per_spin::Vector{Rational{Int}}) where N
    # Per-particle D matrices and helicity indices
    D_mats = [wigner_D(per_spin[i], n_tuple[i]) for i in 1:N]
    # m values: j, j-1, ..., -j (depends only on per_spin)
    m_vals = get!(_M_VALS_CACHE, Tuple(per_spin)) do
        [collect(Float64(j):-1:(-Float64(j))) for j in per_spin]
    end
    # λ column indices in the D matrices (1-based)
    lam_indices = [findfirst(x -> Float64(x) == Float64(lambda_tuple[i]), m_vals[i])
                   for i in 1:N]
    any(x -> x === nothing, lam_indices) &&
        throw(ArgumentError("helicity $lambda_tuple is outside the allowed spin-projection range"))

    # Enumerate all σ configurations
    σ_ranges = [1:length(m_vals[i]) for i in 1:N]
    total_dim = prod(length.(σ_ranges))
    vec = Vector{ComplexF64}(undef, total_dim)

    # Evaluate each σ configuration
    for (flat_idx, σ_indices) in enumerate(Base.Iterators.product(σ_ranges...))
        coeff = ComplexF64(1.0, 0.0)
        for i in 1:N
            coeff *= D_mats[i][σ_indices[i], lam_indices[i]]
        end
        vec[flat_idx] = coeff
    end
    return vec
end

"""
    get_rotation_vector(n_tuple, lambda_tuple, spins, species) -> Vector{ComplexF64}

Convenience function: accept per-species spins and species, then expand them per particle.
"""

# ============ V_can → V_hel transformation ============

"""
    build_V_hel(subspace_states_α, subspace_states_β,
                per_spin_α, per_spin_β, V_can_func, extra_args...)
        -> Matrix{ComplexF64}

Construct helicity-basis V_hel from canonical-polarization-basis V_can (K_α × K_β)。

# Arguments
- `subspace_states_α/β`: `_collect_subspace_states` state list returned by _collect_subspace_states; each entry is `(n_tuple, λ_tuple)`
- `per_spin_α/β`: per-particle spins (length N_α/N_β)
- `V_can_func(n'_tuple, σ'_tuple, n_tuple, σ_tuple, extra_args...)`: returns a V_can matrix element
- `extra_args...`: additional arguments forwarded to V_can_func (for example kapA, kapB, rA, rB, params)

V_hel[k', k] = Σ_{σ',σ} conj(c^(k')_{σ'}) · V_can(n'^(k'), σ', n^(k), σ) · c^(k)_{σ}
"""
function build_V_hel(subspace_states_α::Vector,
                     subspace_states_β::Vector,
                     per_spin_α::AbstractVector{<:Real},
                     per_spin_β::AbstractVector{<:Real},
                     V_can_func::Function, extra_args...;
                     entry_filter=nothing)
    K_α = length(subspace_states_α)
    K_β = length(subspace_states_β)
    K_α == 0 && return spzeros(ComplexF64, 0, 0)
    K_β == 0 && return spzeros(ComplexF64, 0, 0)

    # Precompute rotation-coefficient vectors for all subspace states
    rot_α = [get_rotation_vector(n, lam, per_spin_α) for (n, lam) in subspace_states_α]
    rot_β = [get_rotation_vector(n, lam, per_spin_β) for (n, lam) in subspace_states_β]

    dim_σα = length(rot_α[1])
    dim_σβ = length(rot_β[1])

    # Deduplicate momentum configurations (states with the same n_tuple share a canonical-polarization basis)
    n_α_unique = unique!([n for (n, _) in subspace_states_α])
    n_β_unique = unique!([n for (n, _) in subspace_states_β])

    # Probe the V_can return type: scalar → dim_κ=1; matrix → use its row/column sizes
    σ_probe_α = _σ_configurations(n_α_unique[1], per_spin_α)
    σ_probe_β = _σ_configurations(n_β_unique[1], per_spin_β)
    sample = V_can_func(n_α_unique[1], σ_probe_α[1],
                        n_β_unique[1], σ_probe_β[1], extra_args...)
    dim_κA = sample isa Number ? 1 : size(sample, 1)
    dim_κB = sample isa Number ? 1 : size(sample, 2)

    # momentum → helicity-state indices: states of equal momentum n share a V_can block
    n_to_k_α = Dict{eltype(n_α_unique), Vector{Int}}()
    for (k, (n, _)) in enumerate(subspace_states_α)
        push!(get!(Vector{Int}, n_to_k_α, n), k)
    end
    n_to_k_β = Dict{eltype(n_β_unique), Vector{Int}}()
    for (k, (n, _)) in enumerate(subspace_states_β)
        push!(get!(Vector{Int}, n_to_k_β, n), k)
    end

    # Fused: evaluate V_can blocks and scatter directly into V_hel, without an intermediate Dict
    n_α_list = collect(n_α_unique)
    n_β_list = collect(n_β_unique)
    N_α = length(n_α_list)
    N_β = length(n_β_list)
    n_pairs = N_α * N_β

    m = K_α * dim_κA
    n = K_β * dim_κB

    # σ configurations depend only on per_spin, not n → precompute once
    σ_vals_all_α = _σ_configurations(per_spin_α)
    σ_vals_all_β = _σ_configurations(per_spin_β)

    # Thread-local block-buffer pool (avoid per-pair zeros allocations)
    blk_rows = dim_σα * dim_κA
    blk_cols = dim_σβ * dim_κB
    # threadid() is global across Julia's default and interactive thread
    # pools, whereas nthreads() counts only one pool. Allocate by the
    # largest possible thread ID so a split thread-pool setup is safe.
    n_threads = Threads.maxthreadid()
    block_pool = [zeros(ComplexF64, blk_rows, blk_cols) for _ in 1:n_threads]

    # Per-thread triplet buffer
    triplets = [Tuple{Int,Int,ComplexF64}[] for _ in 1:n_threads]

    Threads.@threads for idx in 1:n_pairs
        i_α = (idx - 1) ÷ N_β + 1
        i_β = (idx - 1) % N_β + 1
        n_α = n_α_list[i_α]
        n_β = n_β_list[i_β]

        # Lightweight prefilter: test one σ pair first; skip if filter rejects everything
        σ_vals_α = σ_vals_all_α
        σ_vals_β = σ_vals_all_β
        if entry_filter !== nothing
            any_pass = false
            for σα in σ_vals_α, σβ in σ_vals_β
                if entry_filter(n_α, σα, n_β, σβ)
                    any_pass = true
                    break
                end
            end
            any_pass || continue
        end

        # take a thread-local block, clear it with fill!, and reuse it
        tid = Threads.threadid()
        block = block_pool[tid]
        fill!(block, 0)
        any_nonzero = false
        for (i_σα, σα) in enumerate(σ_vals_α)
            r0_b = (i_σα - 1) * dim_κA + 1
            for (i_σβ, σβ) in enumerate(σ_vals_β)
                c0_b = (i_σβ - 1) * dim_κB + 1
                if entry_filter !== nothing && !entry_filter(n_α, σα, n_β, σβ)
                    continue
                end
                val = V_can_func(n_α, σα, n_β, σβ, extra_args...)
                if val isa Number
                    iszero(val) && continue
                    block[r0_b, c0_b] = val
                    any_nonzero = true
                else
                    all(iszero, val) && continue
                    block[r0_b:r0_b+dim_κA-1, c0_b:c0_b+dim_κB-1] .= val
                    any_nonzero = true
                end
            end
        end
        any_nonzero || continue

        # scatter to all matching helicity states → collect nonzero triplets
        buf = triplets[tid]
        k_list_α = n_to_k_α[n_α]
        k_list_β = n_to_k_β[n_β]
        for k_α in k_list_α
            c_α = rot_α[k_α]
            r0 = (k_α - 1) * dim_κA + 1
            for k_β in k_list_β
                c_β = rot_β[k_β]
                c0 = (k_β - 1) * dim_κB + 1
                for a in 1:dim_κA, b in 1:dim_κB
                    s = ComplexF64(0.0, 0.0)
                    for i_σα in 1:dim_σα
                        ca = conj(c_α[i_σα])
                        abs2(ca) == 0.0 && continue
                        for i_σβ in 1:dim_σβ
                            cb = c_β[i_σβ]
                            abs2(cb) == 0.0 && continue
                            s += ca * block[(i_σα-1)*dim_κA+a, (i_σβ-1)*dim_κB+b] * cb
                        end
                    end
                    iszero(s) || push!(buf, (r0 + a - 1, c0 + b - 1, s))
                end
            end
        end
    end

    # merge triplets → choose sparse or dense storage based on fill fraction
    all_t = vcat(triplets...)
    return _finalize_V_hel_block(m, n, all_t)
end

# ============ Direct V_hel input (without Wigner rotation) ============

"""
    build_V_hel_direct(subspace_states_α, subspace_states_β,
                       V_hel_adapter, extra_args...) -> Matrix{ComplexF64}

Construct V_hel directly from helicity-basis matrix elements (K_α·dim_κA × K_β·dim_κB)。

Unlike `build_V_hel`, this requires neither Wigner-D rotations nor σ enumeration.
The user supplies `V_hel_adapter(n_α, λ_α, n_β, λ_β, extra_args...)` directly returns
a helicity-basis matrix element (a scalar or a dim_κA × dim_κB matrix).
"""
function build_V_hel_direct(subspace_states_α::Vector,
                            subspace_states_β::Vector,
                            V_hel_adapter::Function, extra_args...;
                            entry_filter=nothing)
    K_α = length(subspace_states_α)
    K_β = length(subspace_states_β)
    K_α == 0 && return spzeros(ComplexF64, 0, 0)
    K_β == 0 && return spzeros(ComplexF64, 0, 0)

    # Probe dim_κ
    n_α_0, λ_α_0 = subspace_states_α[1]
    n_β_0, λ_β_0 = subspace_states_β[1]
    sample = V_hel_adapter(n_α_0, λ_α_0, n_β_0, λ_β_0, extra_args...)
    dim_κA = sample isa Number ? 1 : size(sample, 1)
    dim_κB = sample isa Number ? 1 : size(sample, 2)

    m = K_α * dim_κA
    n = K_β * dim_κB

    # Per-thread triplet buffer. threadid() spans all Julia thread pools;
    # therefore allocate by its global bound rather than nthreads().
    n_threads = Threads.maxthreadid()
    triplets = [Tuple{Int,Int,ComplexF64}[] for _ in 1:n_threads]

    n_pairs = K_α * K_β
    Threads.@threads for idx in 1:n_pairs
        k_α = (idx - 1) ÷ K_β + 1
        k_β = (idx - 1) % K_β + 1
        n_α, λ_α = subspace_states_α[k_α]
        n_β, λ_β = subspace_states_β[k_β]
        r0 = (k_α - 1) * dim_κA + 1
        c0 = (k_β - 1) * dim_κB + 1
        entry_filter !== nothing && !entry_filter(n_α, λ_α, n_β, λ_β) && continue
        val = V_hel_adapter(n_α, λ_α, n_β, λ_β, extra_args...)
        buf = triplets[Threads.threadid()]
        if val isa Number
            iszero(val) && continue
            push!(buf, (r0, c0, val))
        else
            for a in 1:dim_κA, b in 1:dim_κB
                v = val[a, b]
                iszero(v) && continue
                push!(buf, (r0 + a - 1, c0 + b - 1, v))
            end
        end
    end

    # merge triplets → choose sparse or dense storage based on fill fraction
    all_t = vcat(triplets...)
    return _finalize_V_hel_block(m, n, all_t)
end

"""
    _σ_configurations(n_tuple, per_spin) -> Vector{NTuple{N, Rational{Int}}}

Return all possible canonical-polarization σ configurations for a momentum configuration n_tuple.
σ_i ∈ {-j_i, -j_i+1, ..., j_i}。
"""
function _σ_configurations(n_tuple::NTuple{N, Momentum},
                           per_spin::AbstractVector{<:Real}) where N
    m_ranges = [_spin_projections(j) for j in per_spin]
    return collect(Iterators.product(m_ranges...))
end

function _σ_configurations(per_spin::AbstractVector{<:Real})
    m_ranges = [_spin_projections(j) for j in per_spin]
    return collect(Iterators.product(m_ranges...))
end

function _spin_projections(j::Real)
    jr = Rational{Int}(j)
    get!(_SPIN_PROJ_CACHE, jr) do
        vals = Rational{Int}[]
        v = jr
        while v >= -jr
            push!(vals, v)
            v -= 1
        end
        vals
    end
end

function get_rotation_vector(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             spins::AbstractVector{<:Real},
                             species::AbstractVector{<:Integer}) where N
    per_spin = _expand_spins(Rational{Int}.(spins), Int.(species))
    return get_rotation_vector(n_tuple, lambda_tuple, per_spin)
end
