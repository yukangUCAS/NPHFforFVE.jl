# ============================================================
# Projection — I matrices, Löwdin orthogonalization, and irrep-basis expansion
# ============================================================
#
# For every subspace S(n^r, λ, [κ]) and target-group irrep Γ:
#   1. Construct I matrix.
#   2. Löwdin orthogonalization → nonzero eigenvalues Z_r and eigenvectors c^r
#   3. Expand into linear combinations of canonically ordered momentum states → coefficient matrix X
#
# Convention: prioritize single-species support (all particles identical); multi-species generalization follows below.

# ============ Phase-calculation utilities ============

"""
    _total_particle_phase(n::Momentum, g::SMatrix{3,3,Int}, g_idx::Int, n_base::Int,
                          lambda::Float64, spin::Float64, eta_i::Float64, sign::Int=1)
                          -> (ComplexF64, Int)

Compute the total phase and helicity parity P(g) for group element g acting on helicity state |n,λ⟩.

Parity is determined from the group-element index (see _parity_of); g is used only for Wigner-angle and related phase calculations.
sign = ±1 distinguishes the two SU(2) lifts of a double-cover group (and is always 1 for a single-cover group).
n_base is the base O(3) group size (half the full group size for a double cover).
eta_i is the intrinsic parity of this particle.

- Proper rotation (parity=+1): phase = e^{-iλ φ_w(n,g)}, P(g)=+1
- Improper rotation (parity=-1): g = P·R, R=-g is proper
  phase = e^{-iλ φ_w(n,R)} × η_i e^{∓iπs}, P(g)=-1
"""
function _total_particle_phase(n::Momentum, g::SMatrix{3,3,Int},
                                g_idx::Int, n_base::Int,
                                lambda::Float64, spin::Float64, eta_i::Float64, sign::Int=1)
    p = _parity_of(g_idx, n_base)  # ±1
    if p == 1
        phase = helicity_phase(n, g, lambda, sign)
        return phase, 1
    else
        R = -g  # proper part, det(R)=+1
        h_phase = helicity_phase(n, R, lambda, sign)
        Rn = apply_transform(R, n)
        p_phase = _parity_helicity_phase(Rn, lambda, spin, eta_i)
        return h_phase * p_phase, -1
    end
end


""" _total_state_phase(n_tuple, g, g_idx, n_base, lambda_tuple, spin, eta, sign=1) -> ComplexF64

Compute total phase of group element g on an N-particle state: ∏_i phase_i。
sign = ±1 distinguishes the two SU(2) lifts of a double-cover group.
"""

function _total_state_phase(n_tuple::NTuple{N,Momentum}, g::SMatrix{3,3,Int},
                             g_idx::Int, n_base::Int,
                             lambda_tuple::NTuple{N,Float64}, spin::Float64, eta::Float64,
                             sign::Int=1) where N
    total = ComplexF64(1.0, 0.0)
    for i in 1:N
        phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base, lambda_tuple[i], spin, eta, sign)
        total *= phase_i
    end
    return total
end


# ============ Permutation utilities ============

"""
    _permutation_sign(p::Vector{Int}) -> Int

Compute parity of permutation p (+1 even, -1 odd), from inversion count.
"""

function _permutation_sign(p::Vector{Int})
    n = length(p)
    inv_count = 0
    for i in 1:n
        for j in i+1:n
            if p[i] > p[j]
                inv_count += 1
            end
        end
    end
    return iseven(inv_count) ? 1 : -1
end


# ============ Helicity-equivalence-class utilities ============

"""
    _compute_per_single_species(n_tuple, group_elements, n_base)

Compute the Per({n}) group-generator list. Each generator is (parity::Int, inv_perm::Vector{Int})，
Equivalence relation: λ'_i = parity × λ[inv_perm[i]].
"""
function _compute_per_single_species(n_tuple::NTuple{N,Momentum},
                                      group_elements::Vector{<:SMatrix{3,3,Int}},
                                      n_base::Int) where N
    orig_vec = collect(n_tuple)
    generators = Tuple{Int, Vector{Int}}[]

    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        sort(orig_vec) != sort(trans) && continue

        all_perms = _find_all_permutations(orig_vec, trans)
        for p in all_perms
            inv_p = _inverse_permutation(p)
            push!(generators, (parity, inv_p))
        end
    end

    isempty(generators) && return generators

    seen = Set{Tuple{Int, Vector{Int}}}()
    unique_gens = Tuple{Int, Vector{Int}}[]
    for gen in generators
        gen in seen && continue
        push!(seen, gen)
        push!(unique_gens, gen)
    end
    return unique_gens
end

"""
    _compute_helicity_equivalence_class(lambda_tuple, per_generators)

Use BFS to compute lambda_tuple equivalence class under Per-group generators.
Return all helicity tuples in this class (including input) and normalize -0.0 → 0.0.
"""
function _compute_helicity_equivalence_class(lambda_tuple::NTuple{N,Float64},
                                              per_generators::Vector{Tuple{Int, Vector{Int}}}) where N
    class = NTuple{N,Float64}[lambda_tuple]
    visited = Set{NTuple{N,Float64}}([lambda_tuple])
    queue = NTuple{N,Float64}[lambda_tuple]

    while !isempty(queue)
        current = popfirst!(queue)
        for (parity, inv_perm) in per_generators
            raw = ntuple(i -> Float64(parity * current[inv_perm[i]]), N)
            neighbor = Tuple(v == 0.0 ? 0.0 : v for v in raw)
            if neighbor ∉ visited
                push!(visited, neighbor)
                push!(queue, neighbor)
                push!(class, neighbor)
            end
        end
    end

    return class
end

"""
Geometry/helicity data shared by the ordinary single-species I and X builders.
It is intentionally internal while the prepared-orbit API is validated before
being extended to zero-momentum and multi-species paths.
"""
struct _PreparedProjectionOrbit{N}
    lambda_class::Vector{NTuple{N,Float64}}
    subspace_states::Vector{Tuple{NTuple{N,Momentum},NTuple{N,Float64}}}
    state_ordinal::Dict{Tuple{NTuple{N,Momentum},NTuple{N,Float64}},Int}
end

function _collect_subspace_states_from_class(
        n_tuple::NTuple{N,Momentum},
        lambda_class::Vector{NTuple{N,Float64}},
        group_elements::Vector{<:SMatrix{3,3,Int}}, n_base::Int) where N
    seen = Set{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}}()
    for lam_src in lambda_class
        for (g_idx, g) in enumerate(group_elements)
            parity = _parity_of(g_idx, n_base)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]
            keys = collect(zip(trans_mom, trans_hel))
            idx = sortperm(keys)
            canon_mom = Tuple(trans_mom[i] for i in idx)
            canon_hel = Tuple(trans_hel[i] == 0.0 ? 0.0 : trans_hel[i]
                              for i in idx)
            push!(seen, (canon_mom, canon_hel))
        end
    end
    result = collect(seen)
    sort!(result, by=x -> (collect(x[1]), collect(x[2])))
    return result
end

function _prepare_projection_orbit(
        n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
        group_elements::Vector{<:SMatrix{3,3,Int}}, n_base::Int) where N
    generators = _compute_per_single_species(n_tuple, group_elements, n_base)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, generators)
    states = _collect_subspace_states_from_class(
        n_tuple, lambda_class, group_elements, n_base)
    state_ordinal = Dict(st => i for (i, st) in enumerate(states))
    return _PreparedProjectionOrbit{N}(lambda_class, states, state_ordinal)
end


# ============ I matrix ============

"""
    build_I_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                   kappa::String, Gamma::String,
                   group_elements::Vector{<:SMatrix{3,3,Int}},
                   irrep_mats::Vector{<:AbstractMatrix},
                   species_type::Symbol, spin::Float64, etas::Vector{Float64},
                   n_base::Int) where N -> Matrix{ComplexF64}

Construct the I matrix of subspace S(n^r, λ, [κ]).

Formula:
  I_{(b,ν'),(a,ν)} = Σ_{g∈Per({n})} D*_{ν'ν}(g) × phase(g)
                     × Σ_{s∈S_N} δ(s) × R^{[κ]}_{ba}(s)
                     × Π_i δ_{n_{s_i}, g·n_i} × Π_i δ_{λ_{s_i}, P(g)·λ_i}

Here δ(s) is permutation parity sign(s) for fermionic systems and is always 1 for bosonic systems.
Momentum matching: n_{s_i} = g·n_i；Helicity matching: λ_{s_i} = P(g)·λ_i。
No stabilizer precomputation is required; sum directly over full S_N, with Kronecker δ providing the natural selection.

I matrix size: square matrix of dimension n_λ_class × dim([κ]) × dim(Γ).
Index convention: row/column = (λ_idx, b, ν), flattened in column-major order, with λ varying fastest.

λ indices are expanded over the helicity equivalence class, making helicity matching an exact Kronecker δ and correctly including improper-rotation (parity) contributions.

etas is the vector of single-particle intrinsic parities for all species (current single-species support requires length(etas)==1).
"""
function build_I_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         kappa::String, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species_type::Symbol, spin::Float64, etas::Vector{Float64},
                         n_base::Int; prepared_orbit=nothing) where N
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    orig_vec = collect(n_tuple)
    all_s = SN_ELEMENTS[N]

    # Compute helicity equivalence class — expand λ indices over that class.
    lambda_class = if prepared_orbit === nothing
        per_generators = _compute_per_single_species(
            n_tuple, group_elements, n_base)
        _compute_helicity_equivalence_class(lambda_tuple, per_generators)
    else
        prepared_orbit.lambda_class
    end
    n_lam = length(lambda_class)
    lambda_to_idx = Dict(t => k for (k, t) in enumerate(lambda_class))

    d = n_lam * dim_kappa * dim_Gamma
    I = zeros(ComplexF64, d, d)
    d == 0 && return I

    for (g_idx, g) in enumerate(group_elements)
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]

        if sort(orig_vec) != sort(trans)
            continue
        end

        parity = _parity_of(g_idx, n_base)
        D_g = irrep_mats[g_idx]
        sign = g_idx > n_base ? -1 : 1

        for s in all_s
            # Momentum matching: n_{s_i} = g·n_i
            mom_ok = true
            for i in 1:N
                n_tuple[s[i]] != trans[i] && (mom_ok = false; break)
            end
            mom_ok || continue

            inv_s = _inverse_permutation(s)

            fermion_sign = (species_type == :fermion) ? _permutation_sign(s) : 1

            s_idx = get_SN_element_index(N, s)
            R_s = get_SN_irrep_matrix(N, kappa, s_idx)

            # Traverse all source helicities in the equivalence class.
            for (src_idx, lam_src) in enumerate(lambda_class)
                # Phase depends on source λ (different equivalence-class members may have different phases).
                state_phase = _total_state_phase(n_tuple, g, g_idx, n_base, lam_src, spin, etas[1], sign)

                # Target helicity: λ'_i = P(g)·λ_src[s^{-1}[i]], necessarily in the equivalence class.
                lam_tgt_raw = ntuple(i -> parity * lam_src[inv_s[i]], N)
                lam_tgt = Tuple(v == 0.0 ? 0.0 : Float64(v) for v in lam_tgt_raw)
                tgt_idx = lambda_to_idx[lam_tgt]

                for a in 1:dim_kappa, b in 1:dim_kappa
                    R_ba = R_s[b, a]
                    abs(R_ba) < 1e-14 && continue
                    val = ComplexF64(fermion_sign) * R_ba
                    for nu in 1:dim_Gamma, nup in 1:dim_Gamma
                        row = (tgt_idx - 1) * dim_kappa * dim_Gamma + (b - 1) * dim_Gamma + nup
                        col = (src_idx - 1) * dim_kappa * dim_Gamma + (a - 1) * dim_Gamma + nu
                        I[row, col] += conj(D_g[nup, nu]) * state_phase * val
                    end
                end
            end
        end
    end

    return I
end


# ============ Multi-species helper functions ============

"""
    _is_within_species_permutation(p::Vector{Int}, species::Vector{Int}) -> Bool

Check whether permutation p acts only within each species and never across species.
"""
function _is_within_species_permutation(p::Vector{Int}, species::Vector{Int})
    offset = 0
    for Nk in species
        for i in 1:Nk
            pi = p[offset + i]
            if pi <= offset || pi > offset + Nk
                return false
            end
        end
        offset += Nk
    end
    return true
end

"""
    _compute_per_multi_species(n_tuple, group_elements, n_base, species)

Multi-species Per-group generators. Unlike the single-species version, only within-species permutations are considered.
"""
function _compute_per_multi_species(n_tuple::NTuple{N,Momentum},
                                     group_elements::Vector{<:SMatrix{3,3,Int}},
                                     n_base::Int, species::Vector{Int}) where N
    orig_vec = collect(n_tuple)
    generators = Tuple{Int, Vector{Int}}[]

    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        sort(orig_vec) != sort(trans) && continue

        all_perms = _find_all_permutations(orig_vec, trans)
        for p in all_perms
            _is_within_species_permutation(p, species) || continue
            inv_p = _inverse_permutation(p)
            push!(generators, (parity, inv_p))
        end
    end

    isempty(generators) && return generators

    seen = Set{Tuple{Int, Vector{Int}}}()
    unique_gens = Tuple{Int, Vector{Int}}[]
    for gen in generators
        gen in seen && continue
        push!(seen, gen)
        push!(unique_gens, gen)
    end
    return unique_gens
end

"""
    _total_state_phase_multi(n_tuple, g, g_idx, n_base, lam_src,
                             per_particle_spins, per_particle_etas, sign) -> ComplexF64

Multi-species total-state phase: every particle uses spin and eta of its own species.
"""
function _total_state_phase_multi(n_tuple::NTuple{N,Momentum}, g::SMatrix{3,3,Int},
                                   g_idx::Int, n_base::Int,
                                   lam_src::NTuple{N,Float64},
                                   per_particle_spins::Vector{Float64},
                                   per_particle_etas::Vector{Float64},
                                   sign::Int) where N
    total = ComplexF64(1.0, 0.0)
    for i in 1:N
        phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                            lam_src[i], per_particle_spins[i],
                                            per_particle_etas[i], sign)
        total *= phase_i
    end
    return total
end

"""
    _multi_species_permutation_sign(s_full::Vector{Int}, species::Vector{Int},
                                     particle_types::Vector{Symbol}) -> Int

Multi-species fermionic permutation sign: compute sign(s|_k) for each fermionic species and multiply them.
"""
function _multi_species_permutation_sign(s_full::Vector{Int}, species::Vector{Int},
                                          particle_types::Vector{Symbol})
    sign_total = 1
    offset = 0
    for k in 1:length(species)
        if particle_types[k] == :fermion
            Nk = species[k]
            sk = [s_full[offset + i] - offset for i in 1:Nk]
            sign_total *= _permutation_sign(sk)
        end
        offset += species[k]
    end
    return sign_total
end


# ============ Multi-species I matrix ============

"""
    build_I_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                   group_elements, irrep_mats,
                   species, particle_types, spins, etas, n_base)

Multi-species I matrix. Core differences from the single-species version:
- Permutation sums are restricted to direct-product group S_{N₁}×⋯×S_{Nₖ}
- R matrix is the tensor product of S_N irreps
- per-particle spin / eta is determined independently by species
"""
function build_I_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         κ_tuple, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species::Vector{Int}, particle_types::Vector{Symbol},
                         spins::Vector{Float64}, etas::Vector{Float64},
                         n_base::Int) where N
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    dim_Gamma = size(irrep_mats[1], 1)
    orig_vec = collect(n_tuple)

    # Direct-product group elements (replacing SN_ELEMENTS[N])
    prod_gens = _product_group_generators(species)

    # Expand per-particle spin / eta
    per_particle_spins = Float64[]
    per_particle_etas = Float64[]
    for (k, Nk) in enumerate(species)
        for _ in 1:Nk
            push!(per_particle_spins, spins[k])
            push!(per_particle_etas, etas[k])
        end
    end

    # Multi-species Per-group generators + helicity equivalence class
    per_generators = _compute_per_multi_species(n_tuple, group_elements, n_base, species)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)
    n_lam = length(lambda_class)
    lambda_to_idx = Dict(t => k for (k, t) in enumerate(lambda_class))

    d = n_lam * dim_kappa * dim_Gamma
    I = zeros(ComplexF64, d, d)
    d == 0 && return I

    for (g_idx, g) in enumerate(group_elements)
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        sort(orig_vec) != sort(trans) && continue

        parity = _parity_of(g_idx, n_base)
        D_g = irrep_mats[g_idx]
        sign = g_idx > n_base ? -1 : 1

        for prod_gen in prod_gens
            s_full = prod_gen.s_full

            # Momentum matching δ
            mom_ok = true
            for i in 1:N
                n_tuple[s_full[i]] != trans[i] && (mom_ok = false; break)
            end
            mom_ok || continue

            inv_s = _inverse_permutation(s_full)
            ferm_sign = _multi_species_permutation_sign(s_full, species, particle_types)
            per_s_idx = prod_gen.per_s_idx

            for (src_idx, lam_src) in enumerate(lambda_class)
                state_phase = _total_state_phase_multi(n_tuple, g, g_idx, n_base,
                                                        lam_src, per_particle_spins,
                                                        per_particle_etas, sign)

                # Target helicity: λ'_i = P(g)·λ_{s^{-1}[i]}
                lam_tgt_raw = ntuple(i -> parity * lam_src[inv_s[i]], N)
                lam_tgt = Tuple(v == 0.0 ? 0.0 : Float64(v) for v in lam_tgt_raw)
                tgt_idx = lambda_to_idx[lam_tgt]

                for a in 1:dim_kappa, b in 1:dim_kappa
                    R_ba = _multi_R_matrix_element(species, κ_tuple, per_s_idx, a, b)
                    abs(R_ba) < 1e-14 && continue
                    val = ComplexF64(ferm_sign) * R_ba
                    for nu in 1:dim_Gamma, nup in 1:dim_Gamma
                        row = (tgt_idx - 1) * dim_kappa * dim_Gamma + (b - 1) * dim_Gamma + nup
                        col = (src_idx - 1) * dim_kappa * dim_Gamma + (a - 1) * dim_Gamma + nu
                        I[row, col] += conj(D_g[nup, nu]) * state_phase * val
                    end
                end
            end
        end
    end

    return I
end


# ============ Zero-momentum I matrix (new unified framework)============

"""
    _spin_values(j::Rational{Int}) -> Vector{Rational{Int}}

Return all projections of spin j, from high to low.
"""
function _spin_values(j::Rational{Int})
    if j == 1//2
        return Rational{Int}[1//2, -1//2]
    elseif j == 1//1
        return Rational{Int}[1//1, 0//1, -1//1]
    elseif j == 3//2
        return Rational{Int}[3//2, 1//2, -1//2, -3//2]
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end
end

"""
    _spin_tuples(j::Rational{Int}, M::Int) -> Vector{NTuple{M, Rational{Int}}}

Generate all spin-projection configurations of M spin-j particles (σ₁,...,σ_M)，in lexicographic order.
σ index = 1..(2j+1)^M。
"""
function _spin_tuples(j::Rational{Int}, M::Int)
    vals = _spin_values(j)
    n = length(vals)
    nσ = n^M
    result = Vector{NTuple{M, Rational{Int}}}(undef, nσ)
    for flat in 0:(nσ - 1)
        tmp = flat
        tuple_vals = Vector{Rational{Int}}(undef, M)
        for i in M:-1:1
            tuple_vals[i] = vals[tmp % n + 1]
            tmp ÷= n
        end
        result[flat + 1] = NTuple{M, Rational{Int}}(tuple_vals)
    end
    return result
end

"""
    _canonical_spin_tuples(j::Rational{Int}, M::Int) -> Vector{NTuple{M, Rational{Int}}}

Generate canonical spin configurations of M spin-j particles（descending order: σ₁ ≥ σ₂ ≥ ... ≥ σ_M）。
Remove S_M permutation redundancy and retain only the lexicographically largest representative.
"""
function _canonical_spin_tuples(j::Rational{Int}, M::Int)
    M == 0 && return NTuple{0, Rational{Int}}[()]
    vals = _spin_values(j)  # already high to low
    result = NTuple{M, Rational{Int}}[]
    current = Vector{Rational{Int}}(undef, M)
    function descend(pos, start)
        if pos > M
            push!(result, NTuple{M, Rational{Int}}(copy(current)))
            return
        end
        for k in start:length(vals)
            current[pos] = vals[k]
            descend(pos + 1, k)  # k allows equal values
        end
    end
    descend(1, 1)
    return result
end

# Multi-species spin-configuration generation
function _spin_values_float(spin::Float64)
    n = Int(2 * spin + 1)
    return [Float64(spin - i) for i in 0:(n-1)]  # descending, consistent with _spin_values
end

function _multi_spin_tuples(zero_counts::Vector{Int}, spins::Vector{Float64})
    K = length(zero_counts)
    M_total = sum(zero_counts)
    # Generate its own spin-configuration list for each species.
    per_species_lists = Vector{Vector{Float64}}[]
    for k in 1:K
        species_tuples = Vector{Float64}[]
        if zero_counts[k] > 0 && spins[k] != 0.0
            vals = _spin_values_float(spins[k])
            tups = vec(collect(Iterators.product(ntuple(_ -> vals, zero_counts[k])...)))
            for t in tups
                push!(species_tuples, collect(t))
            end
        else
            push!(species_tuples, zeros(Float64, zero_counts[k]))
        end
        push!(per_species_lists, species_tuples)
    end
    combos = vec(collect(Iterators.product(per_species_lists...)))
    result = Vector{NTuple{M_total, Float64}}(undef, length(combos))
    for (idx, combo) in enumerate(combos)
        flat = Float64[]
        for arr in combo
            append!(flat, arr)
        end
        result[idx] = NTuple{M_total, Float64}(flat)
    end
    return result
end

function _multi_spin_to_dj(zero_counts::Vector{Int}, spins::Vector{Float64})
    maps = Dict{Int, Float64}[]
    for k in 1:length(spins)
        if zero_counts[k] > 0 && spins[k] != 0.0
            vals = _spin_values_float(spins[k])
            push!(maps, Dict(v => i for (i, v) in enumerate(vals)))
        else
            push!(maps, Dict{Float64, Int}())
        end
    end
    return maps
end

"""
    _sort_spin_descending(spin_tuple::NTuple{M, Rational{Int}}) -> (NTuple{M, Rational{Int}}, Vector{Int})

Sort a spin configuration into descending canonical form and return (canonical configuration, permutation p).
It satisfies canon[p[i]] == spin_tuple[i], i=1..M.
"""
function _sort_spin_descending(spin_tuple::NTuple{M, Rational{Int}}) where M
    vals = collect(spin_tuple)
    idx = sortperm(vals, rev=true)
    canon = NTuple{M, Rational{Int}}(vals[idx[i]] for i in 1:M)
    p = invperm(idx)
    return canon, p
end

"""
    _SM_x_SNM_elements(N::Int, M::Int) -> Vector{Tuple{Int, Vector{Int}}}

Return a list of (index, permutation vector) for all elements of subgroup S_M × S_{N-M}.
Include only s ∈ S_N satisfying s({1..M}) ⊆ {1..M} and s({M+1..N}) ⊆ {M+1..N}.
"""
function _SM_x_SNM_elements(N::Int, M::Int)
    result = Tuple{Int, Vector{Int}}[]
    for (s_idx, s) in enumerate(SN_ELEMENTS[N])
        in_subgroup = true
        for i in 1:M
            if s[i] > M
                in_subgroup = false
                break
            end
        end
        in_subgroup || continue
        for i in (M + 1):N
            if s[i] <= M
                in_subgroup = false
                break
            end
        end
        in_subgroup || continue
        push!(result, (s_idx, s))
    end
    return result
end

# Multi-species version: ∏_k (S_{M_k} × S_{N_k-M_k}) elements
function _multi_SM_SNM_elements(species::Vector{Int}, zero_counts::Vector{Int})
    K = length(species)
    N = sum(species)
    M_total = sum(zero_counts)
    prod_gens = _product_group_generators(species)
    result = Tuple{Vector{Int}, NTuple{K,Int}}[]
    for gen in prod_gens
        s_full = gen.s_full
        ok = true
        off = 0
        for k in 1:K
            Nk = species[k]
            Mk = zero_counts[k]
            for i in 1:Mk
                s_full[off + i] > off + Mk && (ok = false; break)
            end
            ok || break
            for i in (Mk + 1):Nk
                s_full[off + i] <= off + Mk && (ok = false; break)
            end
            ok || break
            off += Nk
        end
        ok && push!(result, (gen.s_full, gen.per_s_idx))
    end
    return result
end

"""
    _find_spin_stabilizer(spin_tuple::NTuple{M, Rational{Int}}) -> Vector{Vector{Int}}

Return all permutations s ∈ S_M that preserve the spin configuration, satisfying σ_{s_i} = σ_i for all i.
Permutations among equal spins form the stabilizer subgroup.
"""
function _find_spin_stabilizer(spin_tuple::NTuple{M, Rational{Int}}) where M
    vals = collect(spin_tuple)
    results = Vector{Int}[]
    used = falses(M)
    current = Vector{Int}(undef, M)

    function backtrack(i)
        if i > M
            push!(results, copy(current))
            return
        end
        target = vals[i]
        for j in 1:M
            if !used[j] && vals[j] == target
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

"""
    _rotation_axis_angle(R::Matrix{Float64}) -> (n, omega)

Extract rotation axis n (unit vector) and angle ω ∈ [0, π] from a 3×3 rotation matrix.
"""
function _rotation_axis_angle(R::AbstractMatrix{Float64})
    tr = R[1,1] + R[2,2] + R[3,3]
    cos_omega = clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    omega = acos(cos_omega)

    if omega < 1e-12
        n = [0.0, 0.0, 1.0]
    elseif abs(omega - π) < 1e-10
        # 180° rotation: extract axis from (R+I)/2.
        RpI = R + I
        best = 0.0
        n = [0.0, 0.0, 1.0]
        for col in 1:3
            v = Vector{Float64}(RpI[:, col])
            nv = norm(v)
            if nv > best
                best = nv
                n = v / nv
            end
        end
    else
        A = R - R'
        n = [A[3,2], A[1,3], A[2,1]]
        n = n / norm(n)
    end

    return Vector{Float64}(n), Float64(omega)
end

function _wigner_D_for_element(j::Rational{Int}, g_idx::Int, n_base::Int)
    proper_idx = ((g_idx - 1) % 24) + 1
    n, omega = SymmetryGroup._OH_ROTATION_PARAMS[proper_idx]

    if j == 1//2
        D = SymmetryGroup._wigner_D_half(n, omega)
    elseif j == 1//1
        D = SymmetryGroup._wigner_D_one(n, omega)
    elseif j == 3//2
        D = SymmetryGroup._wigner_D_threehalf(n, omega)
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end

    # Oh2 Double cover: second-half elements flip sign for half-integer spin.
    if denominator(j) == 2 && g_idx > n_base
        D = -D
    end

    return D
end

"""
    _wigner_D_for_element(j, g::SMatrix{3,3,Int}, g_idx::Int, n_base::Int)

Compute Wigner D matrices directly from group-element matrices, applicable to any point group.
"""
function _wigner_D_for_element(j::Rational{Int}, g::SMatrix{3,3,Int},
                               g_idx::Int, n_base::Int)
    # Extract proper-rotation part: improper rotation g can be written as -R, where R is proper.
    R_int = det(g) < 0 ? SMatrix{3,3,Int}(-g) : g

    if j == 1//2
        # Use SU(2) lift from Oh table, consistent with helicity_phase.
        D = SymmetryGroup._OH_PROPER_SU2[R_int]
    elseif j == 1//1
        # Find R_int index in Oh table and use consistent axis-angle arguments.
        R_idx = findfirst(x -> x == R_int, SymmetryGroup._OH_ALL[1:24])
        if R_idx === nothing
            # fallback: compute axis-angle directly
            R = Float64.(R_int)
            n, omega = _rotation_axis_angle(R)
            D = SymmetryGroup._wigner_D_one(n, omega)
        else
            n, omega = SymmetryGroup._OH_ROTATION_PARAMS[R_idx]
            D = SymmetryGroup._wigner_D_one(n, omega)
        end
    elseif j == 3//2
        # Use the same axis-angle lift as the tabulated double-cover irreps.
        R_idx = findfirst(x -> x == R_int, SymmetryGroup._OH_ALL[1:24])
        if R_idx === nothing
            R = Float64.(R_int)
            n, omega = _rotation_axis_angle(R)
            D = SymmetryGroup._wigner_D_threehalf(n, omega)
        else
            n, omega = SymmetryGroup._OH_ROTATION_PARAMS[R_idx]
            D = SymmetryGroup._wigner_D_threehalf(n, omega)
        end
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end

    # Double cover: second-half elements flip sign for half-integer spin.
    if denominator(j) == 2 && g_idx > n_base
        D = -D
    end

    return D
end

"""
    build_I_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                  group_elements, irrep_mats,
                                  species_type, spin, etas, n_base)

Construct I matrix of a subspace with M zero-momentum particles (new unified framework).

# Formula (zero_momentum_new.md Eq.15)

I_{(σ'₁..σ'_M; ν'; b), (σ₁..σ_M; ν; a)} =
  Σ_g D*_{ν'ν}(g) × phase(g) ×
  Σ_{s ∈ S_M × S_{N-M}} (fermion sign) ×
  (δ: momentum matching for finite-particles) ×
  (δ: helicity matching for finite-particles) ×
  R_{ba}(s) ×
  D^j_{σ'_{s₁},σ₁}(g) × ... × D^j_{σ'_{s_M},σ_M}(g)

# Arguments
- `M`: number of zero-momentum particles (first M particles; n_tuple[1:M] must be zero)
- `j`: single-particle spin (1//2, 1//1, or 3//2)
- `n_tuple`: representative momenta of all N particles
- `lambda_tuple`: helicities of all N particles
- `kappa`: S_N irrep label
- `Gamma`: target O_h irrep (with parity suffix)
- Other parameters match `build_I_matrix`

# I matrixdimension
d = (2j+1)^M × dim(Γ) × dim([κ])
index: row/col = σ_idx + nσ × (ν-1) + nσ × dim_Γ × (S_N_idx-1) [0-based → +1]
"""
function build_I_matrix_zero_momentum(M::Int, j::Rational{Int},
                                       n_tuple::NTuple{N, Momentum},
                                       lambda_tuple::NTuple{N, Float64},
                                       kappa::String, Gamma::String,
                                       group_elements::Vector{<:SMatrix{3,3,Int}},
                                       irrep_mats::Vector{<:AbstractMatrix},
                                       species_type::Symbol, spin::Float64,
                                       etas::Vector{Float64},
                                       n_base::Int) where {N}
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    nσ = Int((2j + 1)^M)           # number of zero-momentum spin configurations
    d = nσ * dim_Gamma * dim_kappa  # total I-matrix dimension

    I = zeros(ComplexF64, d, d)
    d == 0 && return I  # when M=0 and j is invalid

    # Precompute spin-configuration list and spin-value → matrix-index map.
    spin_tuples = _spin_tuples(j, M)
    spin_vals = _spin_values(j)
    spin_to_idx = Dict{Rational{Int}, Int}(v => k for (k, v) in enumerate(spin_vals))

    # S_M × S_{N-M} subgroup elements
    sm_snm = _SM_x_SNM_elements(N, M)

    # Precompute Wigner D matrices for all group elements.
    wigner_Ds = [_wigner_D_for_element(j, group_elements[g_idx], g_idx, n_base) for g_idx in 1:length(group_elements)]

    fermion = (species_type == :fermion)
    orig_vec = collect(n_tuple)
    eta_val = etas[1]

    for (g_idx, g) in enumerate(group_elements)
        # Prefilter: finite-momentum set is invariant under g.
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        if sort(orig_vec) != sort(trans)
            continue
        end

        parity = _parity_of(g_idx, n_base)
        D_g = irrep_mats[g_idx]
        sign = g_idx > n_base ? -1 : 1
        Dj_g = wigner_Ds[g_idx]  # D^j(g) for zero-momentum spin rotation

        # Finite-momentum particle phases.
        fin_phase = ComplexF64(1.0, 0.0)
        for i in (M + 1):N
            phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                                lambda_tuple[i], spin, eta_val, sign)
            fin_phase *= phase_i
        end

        # Zero-momentum particle intrinsic parity (only improper elements contribute).
        if parity == -1
            fin_phase *= Float64(eta_val)^M
        end

        abs(fin_phase) < 1e-14 && continue

        # Traverse S_M × S_{N-M} subgroup
        for (s_idx, s) in sm_snm
            # Finite-momentum matching: n_{s_i} = g·n_i, i=M+1..N
            mom_ok = true
            for i in (M + 1):N
                n_tuple[s[i]] != trans[i] && (mom_ok = false; break)
            end
            mom_ok || continue

            # Helicity matching: λ_{s_i} = P(g)·λ_i, i=M+1..N
            hel_ok = true
            for i in (M + 1):N
                lambda_tuple[s[i]] != parity * lambda_tuple[i] && (hel_ok = false; break)
            end
            hel_ok || continue

            # Fermionic permutation sign.
            fermion_sign = fermion ? _permutation_sign(s) : 1
            if fermion_sign == 0
                continue  # shouldn't happen, but safety
            end

            R_s = get_SN_irrep_matrix(N, kappa, s_idx)

            # Traverseall indices
            for a in 1:dim_kappa, b in 1:dim_kappa
                R_ba = R_s[b, a]
                abs(R_ba) < 1e-14 && continue
                sN_factor = ComplexF64(fermion_sign) * R_ba

                for nu in 1:dim_Gamma, nup in 1:dim_Gamma
                    Dstar = conj(D_g[nup, nu])
                    abs(Dstar) < 1e-14 && continue
                    g_factor = Dstar * fin_phase * sN_factor

                    # Spin indices: σ' row, σ column.
                    for sigma_idx in 1:nσ, sigmap_idx in 1:nσ
                        # D-product: Π_{i=1}^{M} D^j_{σ'_{s_i}, σ_i}(g)
                        dp = ComplexF64(1.0, 0.0)
                        for i in 1:M
                            si = s[i]  # s_i ∈ {1..M}
                            row_dj = spin_to_idx[spin_tuples[sigmap_idx][si]]
                            col_dj = spin_to_idx[spin_tuples[sigma_idx][i]]
                            dp *= Dj_g[row_dj, col_dj]
                        end
                        abs(dp) < 1e-14 && continue

                        row = (b - 1) * dim_Gamma * nσ + (nup - 1) * nσ + sigmap_idx
                        col = (a - 1) * dim_Gamma * nσ + (nu - 1) * nσ + sigma_idx
                        I[row, col] += g_factor * dp
                    end
                end
            end
        end
    end

    return I
end

# Multi-species version
function build_I_matrix_zero_momentum(zero_counts::Vector{Int},
                                       n_tuple::NTuple{N, Momentum},
                                       lambda_tuple::NTuple{N, Float64},
                                       κ_tuple, Gamma::String,
                                       group_elements::Vector{<:SMatrix{3,3,Int}},
                                       irrep_mats::Vector{<:AbstractMatrix},
                                       species::Vector{Int},
                                       particle_types::Vector{Symbol},
                                       spins::Vector{Float64},
                                       etas::Vector{Float64},
                                       n_base::Int) where {N}
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    dim_Gamma = size(irrep_mats[1], 1)
    M_total = sum(zero_counts)
    K = length(species)

    # Multi-species spin configurations
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    nσ = length(spin_tuples)
    d = nσ * dim_Gamma * dim_kappa
    I = zeros(ComplexF64, d, d)
    d == 0 && return I

    # Global ZM positions → local ZM indices
    zm_global_positions = Int[]
    zm_global_to_local = Dict{Int, Int}()
    for (k, Nk) in enumerate(species)
        off = sum(species[1:k-1]; init=0)
        for i in 1:zero_counts[k]
            gp = off + i
            push!(zm_global_positions, gp)
            zm_global_to_local[gp] = length(zm_global_positions)
        end
    end

    # Species labels of ZM particles (local order)
    zero_particle_species = Int[]
    for (k, mk) in enumerate(zero_counts)
        for _ in 1:mk
            push!(zero_particle_species, k)
        end
    end

    # Wigner D matrix of each species (needed only for species with spin≠0)
    wigner_Ds_by_species = Dict{Int, Vector{Matrix{ComplexF64}}}()
    spin_to_dj_by_species = Dict{Int, Dict{Float64, Int}}()
    for k in 1:K
        if zero_counts[k] > 0 && spins[k] != 0.0
            wigner_Ds_by_species[k] = [
                _wigner_D_for_element(Rational{Int}(Int(2*spins[k]), 2), group_elements[g_idx], g_idx, n_base)
                for g_idx in 1:length(group_elements)]
            vals = _spin_values_float(spins[k])
            spin_to_dj_by_species[k] = Dict(v => i for (i, v) in enumerate(vals))
        end
    end

    # ∏_k (S_{M_k} × S_{N_k-M_k}) subgroup elements
    sm_snm = _multi_SM_SNM_elements(species, zero_counts)

    # Per-particle spin / eta
    per_particle_spins = Float64[]
    per_particle_etas = Float64[]
    for (k, Nk) in enumerate(species)
        for _ in 1:Nk
            push!(per_particle_spins, spins[k])
            push!(per_particle_etas, etas[k])
        end
    end

    orig_vec = collect(n_tuple)

    for (g_idx, g) in enumerate(group_elements)
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        sort(orig_vec) != sort(trans) && continue

        parity = _parity_of(g_idx, n_base)
        D_g = irrep_mats[g_idx]
        sign = g_idx > n_base ? -1 : 1

        # FM-particle phases + ZM intrinsic parity (consistent with single-species version)
        fm_phase = ComplexF64(1.0, 0.0)
        for i in 1:N
            iszero(n_tuple[i]) && continue
            phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                                lambda_tuple[i], per_particle_spins[i],
                                                per_particle_etas[i], sign)
            fm_phase *= phase_i
        end
        if parity == -1
            for k in 1:K
                fm_phase *= Float64(etas[k])^zero_counts[k]
            end
        end
        state_phase = fm_phase
        abs(state_phase) < 1e-14 && continue

        for (s_full, per_s_idx) in sm_snm
            # Momentum matching: at global position i, verify matching ZM/FM status and equal FM momenta
            mom_ok = true
            for i in 1:N
                nzi = iszero(n_tuple[i])
                nzs = iszero(n_tuple[s_full[i]])
                nzi == nzs || (mom_ok = false; break)
                (!nzi && n_tuple[s_full[i]] != trans[i]) && (mom_ok = false; break)
            end
            mom_ok || continue

            # Helicity matching（finite-momentum particles only）
            hel_ok = true
            for i in 1:N
                iszero(n_tuple[i]) && continue
                lambda_tuple[s_full[i]] != parity * lambda_tuple[i] && (hel_ok = false; break)
            end
            hel_ok || continue

            fermion_sign = _multi_species_permutation_sign(s_full, species, particle_types)

            for a in 1:dim_kappa, b in 1:dim_kappa
                R_ba = _multi_R_matrix_element(species, κ_tuple, per_s_idx, a, b)
                abs(R_ba) < 1e-14 && continue
                sN_factor = ComplexF64(fermion_sign) * R_ba

                for nu in 1:dim_Gamma, nup in 1:dim_Gamma
                    Dstar = conj(D_g[nup, nu])
                    abs(Dstar) < 1e-14 && continue
                    g_factor = Dstar * state_phase * sN_factor

                    for sigma_idx in 1:nσ, sigmap_idx in 1:nσ
                        dp = ComplexF64(1.0, 0.0)
                        for (zm_local, sp_k) in enumerate(zero_particle_species)
                            zm_global = zm_global_positions[zm_local]
                            si_global = s_full[zm_global]
                            si_zm_local = zm_global_to_local[si_global]
                            if spins[sp_k] != 0.0
                                Dj_g = wigner_Ds_by_species[sp_k][g_idx]
                                to_idx = spin_to_dj_by_species[sp_k]
                                row_dj = to_idx[spin_tuples[sigmap_idx][si_zm_local]]
                                col_dj = to_idx[spin_tuples[sigma_idx][zm_local]]
                                dp *= Dj_g[row_dj, col_dj]
                            end
                        end
                        abs(dp) < 1e-14 && continue

                        row = (b - 1) * dim_Gamma * nσ + (nup - 1) * nσ + sigmap_idx
                        col = (a - 1) * dim_Gamma * nσ + (nu - 1) * nσ + sigma_idx
                        I[row, col] += g_factor * dp
                    end
                end
            end
        end
    end

    return I
end


# ============ Löwdin orthogonalization ============

"""
    lowdin_orthogonalize(I::Matrix{ComplexF64}; tol::Float64=1e-10)
        -> (Z::Vector{Float64}, C::Matrix{ComplexF64}, nonzero_indices::Vector{Int},
            all_evals::Vector{Float64})

Diagonalize Hermitian matrix I and return nonzero eigenvalues Z_r, corresponding eigenvectors (columns of C), indices of nonzero eigenvalues, and all eigenvalues for caller diagnostics without repeated eigendecomposition.

I matrixis theoretically idempotent up to a constant; eigenvalues should be positive integers Z_r.
"""
function lowdin_orthogonalize(I::Matrix{ComplexF64}; tol::Float64=1e-10)
    # Hermitian diagonalization
    evals, evecs = eigen(Hermitian(I))
    all_evals = Float64.(evals)

    # Select nonzero eigenvalues (relative tolerance avoids numerical-noise misclassification).
    max_ev = maximum(abs, evals)
    threshold = max(tol, max_ev * 1e-12)

    nonzero_idx = Int[]
    nonzero_vals = Float64[]
    for (k, val) in enumerate(evals)
        if abs(val) > threshold
            push!(nonzero_idx, k)
            push!(nonzero_vals, real(val))
        end
    end

    if isempty(nonzero_idx)
        return Float64[], Matrix{ComplexF64}(undef, size(I,1), 0), Int[], all_evals
    end

    Z = Float64.(evals[nonzero_idx])  # should be positive integers (theoretical guarantee)
    C = evecs[:, nonzero_idx]         # corresponding eigenvectors

    return Z, C, nonzero_idx, all_evals
end

# ============ Irrep-basis expansion (X matrix)============

"""
    _collect_subspace_states(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                             group_elements::Vector{<:SMatrix{3,3,Int}},
                             spin::Float64, eta::Float64, n_base::Int) where N
        -> Vector{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}}

Collect all distinct canonically ordered momentum-helicity states in the subspace.
Traverse all λ in the equivalence class and all g ∈ G, compute (canonical_momenta, canonical_helicities), then deduplicate and sort.
n_base is the base O(3) group size，for parity determination.
"""
function _collect_subspace_states(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                                   group_elements::Vector{<:SMatrix{3,3,Int}},
                                   spin::Float64, eta::Float64, n_base::Int) where N
    # Compute helicity equivalence class，expanded over all equivalent λ
    per_generators = _compute_per_single_species(n_tuple, group_elements, n_base)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)

    return _collect_subspace_states_from_class(
        n_tuple, lambda_class, group_elements, n_base)
end

# Multi-species version: sort separately within every species to preserve species boundaries in canonical order
function _collect_subspace_states(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                                   group_elements::Vector{<:SMatrix{3,3,Int}},
                                   species::Vector{Int}, spins::Vector{Float64},
                                   etas::Vector{Float64}, n_base::Int) where N
    per_generators = _compute_per_multi_species(n_tuple, group_elements, n_base, species)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)

    seen = Set{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}}()
    for lam_src in lambda_class
        for (g_idx, g) in enumerate(group_elements)
            parity = _parity_of(g_idx, n_base)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]

            # sort independently within each species, preserving species boundaries
            canon_mom = Momentum[]
            canon_hel = Float64[]
            off = 0
            for Nk in species
                pairs = [(trans_mom[off + i], trans_hel[off + i]) for i in 1:Nk]
                idx = sortperm(pairs)
                for i in idx
                    push!(canon_mom, trans_mom[off + i])
                    v = trans_hel[off + i]
                    push!(canon_hel, v == 0.0 ? 0.0 : v)
                end
                off += Nk
            end
            push!(seen, (Tuple(canon_mom), Tuple(canon_hel)))
        end
    end

    result = collect(seen)
    sort!(result, by = x -> (collect(x[1]), collect(x[2])))
    return result
end

"""
    _collect_fin_subspace_states(fin_n_tuple, fin_lam_tuple, group_elements, spin, eta, n_base)

Collect all canonically ordered subspace states of finite-momentum particles (N-M particles).
Used as finite-momentum blocks when constructing the zero-momentum X matrix.
"""
function _collect_fin_subspace_states(fin_n_tuple::NTuple{Nfm, Momentum},
                                      fin_lam_tuple::NTuple{Nfm, Float64},
                                      group_elements, spin, eta, n_base) where Nfm
    Nfm == 0 && return [(Tuple{}(), Tuple{}())]
    seen = Set{Tuple{NTuple{Nfm, Momentum}, NTuple{Nfm, Float64}}}()
    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        trans_mom = [apply_transform(g, fin_n_tuple[i]) for i in 1:Nfm]
        trans_hel = [parity * fin_lam_tuple[i] for i in 1:Nfm]
        keys_vec = collect(zip(trans_mom, trans_hel))
        idx = sortperm(keys_vec)
        canon_mom = Tuple(trans_mom[i] for i in idx)
        canon_hel = Tuple(trans_hel[i] == 0.0 ? 0.0 : trans_hel[i] for i in idx)
        push!(seen, (canon_mom, canon_hel))
    end
    result = collect(seen)
    sort!(result, by = x -> (collect(x[1]), collect(x[2])))
    return result
end

# Multi-species version: sort independently within each species.
function _collect_fin_subspace_states(fin_n_tuple::NTuple{Nfm, Momentum},
                                       fin_lam_tuple::NTuple{Nfm, Float64},
                                       group_elements, fin_species::Vector{Int},
                                       spins::Vector{Float64}, etas::Vector{Float64},
                                       n_base) where Nfm
    Nfm == 0 && return [(Tuple{}(), Tuple{}())]
    seen = Set{Tuple{NTuple{Nfm, Momentum}, NTuple{Nfm, Float64}}}()
    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        trans_mom = [apply_transform(g, fin_n_tuple[i]) for i in 1:Nfm]
        trans_hel = [parity * fin_lam_tuple[i] for i in 1:Nfm]
        # Sort within each species.
        canon_mom_arr = Momentum[]
        canon_hel_arr = Float64[]
        off = 0
        for Nk in fin_species
            pairs = [(trans_mom[off + i], trans_hel[off + i]) for i in 1:Nk]
            idx = sortperm(pairs)
            for i in idx
                push!(canon_mom_arr, trans_mom[off + i])
                v = trans_hel[off + i]
                push!(canon_hel_arr, v == 0.0 ? 0.0 : v)
            end
            off += Nk
        end
        push!(seen, (Tuple(canon_mom_arr), Tuple(canon_hel_arr)))
    end
    result = collect(seen)
    sort!(result, by = x -> (collect(x[1]), collect(x[2])))
    return result
end

"""
    build_X_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                   kappa::String, Gamma::String,
                   group_elements::Vector{<:SMatrix{3,3,Int}},
                   irrep_mats::Vector{<:AbstractMatrix},
                   species_type::Symbol, spin::Float64, etas::Vector{Float64},
                   n_base::Int, Z::Vector{Float64}, C::Matrix{ComplexF64}) where N -> Matrix{ComplexF64}

Construct coefficient matrix X from nonzero I-matrix eigenvectors c^r.

The I matrix is expanded over the helicity equivalence class, with λ indexing members of that class.
The row-index convention of C matches that of the I matrix: (λ_idx, b, ν), flattened in column-major order.

Formula:
  |Γ, r⟩ = √(dimΓ / (|G|·Z_r)) × Σ_{λ,b,ν} c^r_{λ,b,ν}
              × Σ_{g∈G} D^{*}_{1,ν}(g) × phase(g; λ)
              × Σ_p δ(p) × R_{b'b}(p) × |{n'}, λ'; b'⟩

Here p is the permutation that sorts (g·n) into canonical order {n'}, and λ'_i = P(g) λ_{p̄_i}.
n_base is the size of the base O(3) group and is used for parity and SU(2)-lift signs; |G| is the full group size.

X matrix: rows are canonical basis states (n', λ', b'); columns are r, for a total of n_r columns.


etas is the vector of single-particle intrinsic parities for all species (current single-species support requires length(etas)==1).
"""
function build_X_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         kappa::String, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species_type::Symbol, spin::Float64, etas::Vector{Float64},
                         n_base::Int, Z::Vector{Float64}, C::Matrix{ComplexF64};
                         prepared_orbit=nothing) where N
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    nG = length(group_elements)

    # Helicity equivalence class (consistent with build_I_matrix).
    lambda_class = if prepared_orbit === nothing
        per_generators = _compute_per_single_species(
            n_tuple, group_elements, n_base)
        _compute_helicity_equivalence_class(lambda_tuple, per_generators)
    else
        prepared_orbit.lambda_class
    end
    n_lam = length(lambda_class)

    # Collect all canonical subspace states, expanded over all equivalent λ.
    subspace_states = prepared_orbit === nothing ?
        _collect_subspace_states_from_class(
            n_tuple, lambda_class, group_elements, n_base) :
        prepared_orbit.subspace_states
    n_states = length(subspace_states)
    subspace_dim = n_states * dim_kappa

    n_r = length(Z)
    X = zeros(ComplexF64, subspace_dim, n_r)
    n_r == 0 && return X

    state_ordinal = prepared_orbit === nothing ?
        Dict(st => i for (i, st) in enumerate(subspace_states)) :
        prepared_orbit.state_ordinal

    # Traverse all g ∈ G and all source helicities.
    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        sign = g_idx > n_base ? -1 : 1
        D_g = irrep_mats[g_idx]

        for (src_idx, lam_src) in enumerate(lambda_class)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]

            # Sort into canonical order.
            keys_g = collect(zip(trans_mom, trans_hel))
            idx_sorted = sortperm(keys_g)
            canon_mom = Tuple(trans_mom[i] for i in idx_sorted)
            canon_hel = Tuple(trans_hel[i] == 0.0 ? 0.0 : trans_hel[i] for i in idx_sorted)

            canon_key = (canon_mom, canon_hel)
            state_idx = get(state_ordinal, canon_key, 0)
            state_idx == 0 && continue
            row_base = (state_idx - 1) * dim_kappa + 1

            orig_vec = collect(canon_mom)
            all_perms = _find_all_permutations(orig_vec, trans_mom)
            found_p = nothing
            for p in all_perms
                hel_ok = true
                for i in 1:N
                    canon_hel[p[i]] != trans_hel[i] && (hel_ok = false; break)
                end
                hel_ok && (found_p = p; break)
            end
            found_p === nothing && continue

            state_phase = _total_state_phase(n_tuple, g, g_idx, n_base, lam_src, spin, etas[1], sign)

            fermion_sign = (species_type == :fermion) ? _permutation_sign(found_p) : 1
            prefactor = state_phase * ComplexF64(fermion_sign)

            p_idx = get_SN_element_index(N, found_p)
            R_p = get_SN_irrep_matrix(N, kappa, p_idx)

            for r in 1:n_r
                Z_r = Z[r]
                norm_factor = sqrt(dim_Gamma / (nG * Z_r))

                for b in 1:dim_kappa, nu in 1:dim_Gamma
                    # C index: (src_idx-1)*dim_kappa*dim_Gamma + (b-1)*dim_Gamma + nu
                    c_idx = (src_idx - 1) * dim_kappa * dim_Gamma + (b - 1) * dim_Gamma + nu
                    c_bnu = C[c_idx, r]
                    abs(c_bnu) < 1e-14 && continue

                    Dstar = conj(D_g[1, nu])
                    term = norm_factor * prefactor * Dstar * c_bnu

                    for bp in 1:dim_kappa
                        R_bpb = R_p[bp, b]
                        abs(R_bpb) < 1e-14 && continue
                        row = row_base + bp - 1
                        X[row, r] += term * R_bpb
                    end
                end
            end
        end
    end

    return X
end

# Multi-species version
function build_X_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         κ_tuple, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species::Vector{Int}, particle_types::Vector{Symbol},
                         spins::Vector{Float64}, etas::Vector{Float64},
                         n_base::Int, Z::Vector{Float64}, C::Matrix{ComplexF64}) where N
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    dim_Gamma = size(irrep_mats[1], 1)
    nG = length(group_elements)

    per_particle_spins = Float64[]
    per_particle_etas = Float64[]
    for (k, Nk) in enumerate(species)
        for _ in 1:Nk
            push!(per_particle_spins, spins[k])
            push!(per_particle_etas, etas[k])
        end
    end

    per_generators = _compute_per_multi_species(n_tuple, group_elements, n_base, species)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)
    n_lam = length(lambda_class)

    subspace_states = _collect_subspace_states(n_tuple, lambda_tuple, group_elements,
                                                species, spins, etas, n_base)
    n_states = length(subspace_states)
    subspace_dim = n_states * dim_kappa

    n_r = length(Z)
    X = zeros(ComplexF64, subspace_dim, n_r)
    n_r == 0 && return X

    state_index = Dict{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}, Int}()
    for (k, st) in enumerate(subspace_states)
        state_index[st] = (k - 1) * dim_kappa + 1
    end

    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        sign = g_idx > n_base ? -1 : 1
        D_g = irrep_mats[g_idx]

        for (src_idx, lam_src) in enumerate(lambda_class)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]

            # Sort independently within each species, consistent with _collect_subspace_states.
            canon_mom_arr = Momentum[]
            canon_hel_arr = Float64[]
            off = 0
            for Nk in species
                pairs = [(trans_mom[off + i], trans_hel[off + i]) for i in 1:Nk]
                idx_s = sortperm(pairs)
                for i in idx_s
                    push!(canon_mom_arr, trans_mom[off + i])
                    v = trans_hel[off + i]
                    push!(canon_hel_arr, v == 0.0 ? 0.0 : v)
                end
                off += Nk
            end
            canon_mom = Tuple(canon_mom_arr)
            canon_hel = Tuple(canon_hel_arr)

            canon_key = (canon_mom, canon_hel)
            !haskey(state_index, canon_key) && continue
            row_base = state_index[canon_key]

            orig_vec = collect(canon_mom)
            all_perms = _find_all_permutations(orig_vec, trans_mom)
            found_p = nothing
            for p in all_perms
                # Accept only within-species permutations.
                _is_within_species_permutation(p, species) || continue
                hel_ok = true
                for i in 1:N
                    canon_hel[p[i]] != trans_hel[i] && (hel_ok = false; break)
                end
                hel_ok && (found_p = p; break)
            end
            found_p === nothing && continue

            state_phase = _total_state_phase_multi(n_tuple, g, g_idx, n_base,
                                                    lam_src, per_particle_spins,
                                                    per_particle_etas, sign)

            ferm_sign = _multi_species_permutation_sign(found_p, species, particle_types)
            prefactor = state_phase * ComplexF64(ferm_sign)

            per_s_idx = _decompose_multi_species_permutation(found_p, species)

            for r in 1:n_r
                Z_r = Z[r]
                norm_factor = sqrt(dim_Gamma / (nG * Z_r))

                for b in 1:dim_kappa, nu in 1:dim_Gamma
                    c_idx = (src_idx - 1) * dim_kappa * dim_Gamma + (b - 1) * dim_Gamma + nu
                    c_bnu = C[c_idx, r]
                    abs(c_bnu) < 1e-14 && continue

                    Dstar = conj(D_g[1, nu])
                    term = norm_factor * prefactor * Dstar * c_bnu

                    for bp in 1:dim_kappa
                        R_bpb = _multi_R_matrix_element(species, κ_tuple, per_s_idx, b, bp)
                        abs(R_bpb) < 1e-14 && continue
                        row = row_base + bp - 1
                        X[row, r] += term * R_bpb
                    end
                end
            end
        end
    end

    return X
end

# ============ Zero-momentum X matrix ============

"""
    build_X_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                  group_elements, irrep_mats,
                                  species_type, spin, etas, n_base,
                                  Z, C) -> Matrix{ComplexF64}

Construct coefficient matrix X from nonzero I-matrix eigenvectors c^r for a subspace with M zero-momentum particles.

# Formula (zero_momentum_new.md Eq.25-31)

|Γ, r⟩ = √(dimΓ/(|G|·Z_r)) × Σ_{σ,ν,b} Σ_{g∈G} (c^r)_{σ,ν,b} × D^{Γ*}_{1,ν}(g)
         × (finite-momentum helicity phase) × (intrinsic parity η^M)
         × Σ_{σ'} D^j_{σ'₁,σ₁}(g) × ... × D^j_{σ'_M,σ_M}(g)
         × |σ', {g·n, P(g)λ}; b⟩

canonical basis state: |σ'_canon, {n'_canon, λ'_canon}; b'⟩
Here σ'_canon is in descending canonical order, and {n'_canon, λ'_canon} is in canonical order for finite momenta.

X matrix: row = (σ'_canon, n'_canon, λ'_canon, b'); columns = r, with n_r columns and μ=1.
       columns = r，with n_r columns，μ=1

# Arguments
- `M`: number of zero-momentum particles (first M particles)
- `j`: single-particle spin (1//2, 1//1, or 3//2)
- `n_tuple`, `lambda_tuple`: N N-particle momentum/helicity representative
- `kappa`, `Gamma`: S_N / point-group irrep label
- `Z`: I matrixnonzero eigenvalues
- C: nonzero I-matrix eigenvector matrix; C[:, r] is flattened as (b, ν, σ), with σ varying fastest.
"""
function build_X_matrix_zero_momentum(M::Int, j::Rational{Int},
                                       n_tuple::NTuple{N, Momentum},
                                       lambda_tuple::NTuple{N, Float64},
                                       kappa::String, Gamma::String,
                                       group_elements::Vector{<:SMatrix{3,3,Int}},
                                       irrep_mats::Vector{<:AbstractMatrix},
                                       species_type::Symbol, spin::Float64,
                                       etas::Vector{Float64},
                                       n_base::Int, Z::Vector{Float64},
                                       C::Matrix{ComplexF64};
                                       canonical_spin::Bool=false) where N
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    nG = length(group_elements)
    n_r = length(Z)

    # Spin configurations.
    all_spin_tuples = _spin_tuples(j, M)
    nσ = length(all_spin_tuples)

    if canonical_spin
        canon_spin_tuples = _canonical_spin_tuples(j, M)
        nσ_canon = length(canon_spin_tuples)
        canon_spin_to_idx = Dict(t => k for (k, t) in enumerate(canon_spin_tuples))
        spin_canon_info = Dict{typeof(all_spin_tuples[1]),
                               Tuple{typeof(canon_spin_tuples[1]), Vector{Int}}}()
        for t in all_spin_tuples
            canon, p_M = _sort_spin_descending(t)
            spin_canon_info[t] = (canon, p_M)
        end
    end

    # Wigner D matrices.
    wigner_Ds = [_wigner_D_for_element(j, group_elements[g_idx], g_idx, n_base) for g_idx in 1:nG]

    # Spin value → D-matrix index.
    spin_vals = _spin_values(j)
    spin_to_dj = Dict(v => k for (k, v) in enumerate(spin_vals))

    # Finite-momentum subspace states.
    fin_n = ntuple(i -> n_tuple[M + i], N - M)
    fin_lam = ntuple(i -> lambda_tuple[M + i], N - M)
    fin_states = _collect_fin_subspace_states(fin_n, fin_lam, group_elements, spin, etas[1], n_base)
    n_fin = length(fin_states)

    fin_state_to_rowbase = Dict{Tuple, Int}()
    for (k, st) in enumerate(fin_states)
        fin_state_to_rowbase[st] = (k - 1) * dim_kappa + 1
    end

    # X-matrix row dimension.
    nσ_row = canonical_spin ? length(_canonical_spin_tuples(j, M)) : nσ
    row_dim = nσ_row * n_fin * dim_kappa
    X = zeros(ComplexF64, row_dim, n_r)
    n_r == 0 && return X

    fermion = (species_type == :fermion)
    eta_val = etas[1]

    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        sign = g_idx > n_base ? -1 : 1
        Dg_Gamma = irrep_mats[g_idx]
        Dj_g = wigner_Ds[g_idx]

        # Finite-momentum particle phases (helicity + e^{∓iπs} + intrinsic parity).
        fin_phase = ComplexF64(1.0, 0.0)
        for i in (M + 1):N
            phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                                lambda_tuple[i], spin, eta_val, sign)
            fin_phase *= phase_i
        end
        # Intrinsic parity of zero-momentum particles.
        if parity == -1
            fin_phase *= Float64(eta_val)^M
        end
        abs(fin_phase) < 1e-14 && continue

        # Transform finite momenta/helicities and sort canonically.
        Nfm = N - M
        trans_fin_mom = [apply_transform(g, n_tuple[i]) for i in (M + 1):N]
        trans_fin_hel = [parity * lambda_tuple[i] for i in (M + 1):N]

        if Nfm > 0
            keys_fin = collect(zip(trans_fin_mom, trans_fin_hel))
            idx_fin = sortperm(keys_fin)
            canon_fin_mom_vec = [trans_fin_mom[i] for i in idx_fin]
            canon_fin_hel_vec = [trans_fin_hel[i] == 0.0 ? 0.0 : trans_fin_hel[i] for i in idx_fin]
            canon_fin_mom = Tuple(canon_fin_mom_vec)
            canon_fin_hel = Tuple(canon_fin_hel_vec)
            canon_fin_key = (canon_fin_mom, canon_fin_hel)

            !haskey(fin_state_to_rowbase, canon_fin_key) && continue
            fin_row_base = fin_state_to_rowbase[canon_fin_key]

            all_perms_fin = _find_all_permutations(canon_fin_mom_vec, trans_fin_mom)
            found_p_fin = nothing
            for p_fin in all_perms_fin
                hel_ok = true
                for i in 1:Nfm
                    canon_fin_hel_vec[p_fin[i]] != trans_fin_hel[i] && (hel_ok = false; break)
                end
                hel_ok && (found_p_fin = p_fin; break)
            end
            found_p_fin === nothing && continue
        else
            # M == N: no finite-momentum particles
            fin_row_base = 1
            found_p_fin = Int[]
        end

        # Traversespin indices
        for sigma_idx in 1:nσ
            sigma = all_spin_tuples[sigma_idx]

            for sigmap_idx in 1:nσ
                sigmap = all_spin_tuples[sigmap_idx]

                # D-product: Π_i D^j_{σ'_i, σ_i}(g)
                Dprod = ComplexF64(1.0, 0.0)
                for i in 1:M
                    Dprod *= Dj_g[spin_to_dj[sigmap[i]], spin_to_dj[sigma[i]]]
                end
                abs(Dprod) < 1e-14 && continue

                if canonical_spin
                    # σ' sort into descending canonical form
                    sigmap_canon, p_spin = spin_canon_info[sigmap]
                    σ_canon_idx = canon_spin_to_idx[sigmap_canon]

                    # Construct full permutation p_total ∈ S_M × S_{N-M} ⊂ S_N
                    p_total = Vector{Int}(undef, N)
                    for i in 1:M
                        p_total[i] = p_spin[i]
                    end
                    for i in 1:(N - M)
                        p_total[M + i] = M + found_p_fin[i]
                    end

                    ferm_sign = fermion ? _permutation_sign(p_total) : 1
                    p_idx = get_SN_element_index(N, p_total)
                    R_p = get_SN_irrep_matrix(N, kappa, p_idx)

                    σ_row_base = (σ_canon_idx - 1) * n_fin * dim_kappa
                else
                    # Unsymmetrized basis: spins are not sorted，σ' each retains an independent row
                    # p_total: identity permutation in the ZM-spin sector.
                    p_total = Vector{Int}(undef, N)
                    for i in 1:M
                        p_total[i] = i
                    end
                    for i in 1:(N - M)
                        p_total[M + i] = M + found_p_fin[i]
                    end

                    ferm_sign = fermion ? _permutation_sign(p_total) : 1
                    p_idx = get_SN_element_index(N, p_total)
                    R_p = get_SN_irrep_matrix(N, kappa, p_idx)

                    σ_row_base = (sigmap_idx - 1) * n_fin * dim_kappa
                end

                # Traverse S_N and Γ indices.
                for a in 1:dim_kappa
                    for b in 1:dim_kappa  # b' — output S_N index
                        R_ba = R_p[b, a]
                        abs(R_ba) < 1e-14 && continue

                        for nu in 1:dim_Gamma
                            # C row index: (a-1)*dimΓ*nσ + (nu-1)*nσ + sigma_idx.
                            c_idx = (a - 1) * dim_Gamma * nσ + (nu - 1) * nσ + sigma_idx

                            for r in 1:n_r
                                c_val = C[c_idx, r]
                                abs(c_val) < 1e-14 && continue

                                Dstar = conj(Dg_Gamma[1, nu])  # μ=1
                                norm_factor = sqrt(dim_Gamma / (nG * Z[r]))
                                term = norm_factor * fin_phase * ComplexF64(ferm_sign) *
                                       Dstar * c_val * Dprod * R_ba

                                row = σ_row_base + fin_row_base + b - 1
                                X[row, r] += term
                            end
                        end
                    end
                end
            end
        end
    end

    return X
end

# Multi-species version
function build_X_matrix_zero_momentum(zero_counts::Vector{Int},
                                       n_tuple::NTuple{N,Momentum},
                                       lambda_tuple::NTuple{N,Float64},
                                       κ_tuple, Gamma::String,
                                       group_elements::Vector{<:SMatrix{3,3,Int}},
                                       irrep_mats::Vector{<:AbstractMatrix},
                                       species::Vector{Int},
                                       particle_types::Vector{Symbol},
                                       spins::Vector{Float64},
                                       etas::Vector{Float64},
                                       n_base::Int, Z::Vector{Float64},
                                       C::Matrix{ComplexF64}) where N
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    dim_Gamma = size(irrep_mats[1], 1)
    nG = length(group_elements)
    n_r = length(Z)
    K = length(species)
    M_total = sum(zero_counts)
    Nfm = N - M_total

    # Multi-species spin configurations
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    nσ = length(spin_tuples)

    # Global ZM positions → local ZM indices.
    zm_global_positions = Int[]
    zm_global_to_local = Dict{Int, Int}()
    for (k, Nk) in enumerate(species)
        off = sum(species[1:k-1]; init=0)
        for i in 1:zero_counts[k]
            gp = off + i
            push!(zm_global_positions, gp)
            zm_global_to_local[gp] = length(zm_global_positions)
        end
    end

    # Species labels of ZM particles (local order)
    zm_particle_species = Int[]
    for (k, mk) in enumerate(zero_counts)
        for _ in 1:mk
            push!(zm_particle_species, k)
        end
    end

    # Global FM positions → local FM indices.
    fm_global_positions = Int[]
    fm_global_to_local = Dict{Int, Int}()
    fm_particle_species = Int[]
    for (k, Nk) in enumerate(species)
        off = sum(species[1:k-1]; init=0)
        for i in (zero_counts[k] + 1):Nk
            gp = off + i
            push!(fm_global_positions, gp)
            fm_global_to_local[gp] = length(fm_global_positions)
            push!(fm_particle_species, k)
        end
    end

    # Wigner D matrices for each species.
    wigner_Ds_by_species = Dict{Int, Vector{Matrix{ComplexF64}}}()
    spin_to_dj_by_species = Dict{Int, Dict{Float64, Int}}()
    for k in 1:K
        if zero_counts[k] > 0 && spins[k] != 0.0
            wigner_Ds_by_species[k] = [
                _wigner_D_for_element(Rational{Int}(Int(2*spins[k]), 2), group_elements[g_idx], g_idx, n_base)
                for g_idx in 1:nG]
            vals = _spin_values_float(spins[k])
            spin_to_dj_by_species[k] = Dict(v => i for (i, v) in enumerate(vals))
        end
    end

    # Finite-momentum subspace states (multi-species version).
    if Nfm > 0
        fin_n_list = Momentum[n_tuple[i] for i in 1:N if !iszero(n_tuple[i])]
        fin_lam_list = Float64[lambda_tuple[i] for i in 1:N if !iszero(n_tuple[i])]
        fin_n = Tuple(fin_n_list)
        fin_lam = Tuple(fin_lam_list)
        fin_species_vec = Int[species[k] - zero_counts[k] for k in 1:K]
        fin_species_vec = Int[n for n in fin_species_vec if n > 0]
        fin_spins = Float64[spins[k] for k in 1:K if species[k] > zero_counts[k]]
        fin_etas = Float64[etas[k] for k in 1:K if species[k] > zero_counts[k]]
        fin_states = _collect_fin_subspace_states(fin_n, fin_lam, group_elements,
                                                   fin_species_vec, fin_spins, fin_etas, n_base)
    else
        fin_states = [(Tuple{}(), Tuple{}())]
    end
    n_fin = length(fin_states)

    fin_state_to_rowbase = Dict{Tuple, Int}()
    for (k, st) in enumerate(fin_states)
        fin_state_to_rowbase[st] = (k - 1) * dim_kappa + 1
    end

    # FM species-block sizes (used by _is_within_species_permutation filtering).
    fin_species_for_is = Int[species[k] - zero_counts[k] for k in 1:K]
    fin_species_for_is = Int[n for n in fin_species_for_is if n > 0]

    # X-matrix row dimension.
    row_dim = nσ * n_fin * dim_kappa
    X = zeros(ComplexF64, row_dim, n_r)
    n_r == 0 && return X

    # Per-particle spin / eta (used for phase evaluation).
    per_particle_spins = Float64[]
    per_particle_etas = Float64[]
    for (k, Nk) in enumerate(species)
        for _ in 1:Nk
            push!(per_particle_spins, spins[k])
            push!(per_particle_etas, etas[k])
        end
    end

    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        sign = g_idx > n_base ? -1 : 1
        Dg_Gamma = irrep_mats[g_idx]

        # Finite-momentum particle phases (each particle uses its own spin/eta).
        fin_phase = ComplexF64(1.0, 0.0)
        for (fm_local, gp) in enumerate(fm_global_positions)
            sp_k = fm_particle_species[fm_local]
            phase_i, _ = _total_particle_phase(n_tuple[gp], g, g_idx, n_base,
                                                lambda_tuple[gp], spins[sp_k],
                                                etas[sp_k], sign)
            fin_phase *= phase_i
        end
        # Intrinsic-parity factor of zero-momentum particles: ∏_k η_k^{M_k}.
        if parity == -1
            for k in 1:K
                fin_phase *= Float64(etas[k])^zero_counts[k]
            end
        end
        abs(fin_phase) < 1e-14 && continue

        # Transform finite momenta/helicities and sort separately by species.
        trans_fin_mom = Momentum[apply_transform(g, n_tuple[gp]) for gp in fm_global_positions]
        trans_fin_hel = Float64[parity * lambda_tuple[gp] for gp in fm_global_positions]

        if Nfm > 0
            canon_fin_mom_arr = Momentum[]
            canon_fin_hel_arr = Float64[]
            off_fm = 0
            for k in 1:K
                Nk_fm = species[k] - zero_counts[k]
                if Nk_fm > 0
                    pairs = [(trans_fin_mom[off_fm + i], trans_fin_hel[off_fm + i])
                             for i in 1:Nk_fm]
                    idx_s = sortperm(pairs)
                    for i in idx_s
                        push!(canon_fin_mom_arr, trans_fin_mom[off_fm + i])
                        v = trans_fin_hel[off_fm + i]
                        push!(canon_fin_hel_arr, v == 0.0 ? 0.0 : v)
                    end
                    off_fm += Nk_fm
                end
            end
            canon_fin_mom = Tuple(canon_fin_mom_arr)
            canon_fin_hel = Tuple(canon_fin_hel_arr)
            canon_fin_key = (canon_fin_mom, canon_fin_hel)

            !haskey(fin_state_to_rowbase, canon_fin_key) && continue
            fin_row_base = fin_state_to_rowbase[canon_fin_key]

            # Find stabilizer permutation: canon_fin[p_fin[i]] == trans_fin[i].
            canon_fin_mom_vec = collect(canon_fin_mom)
            canon_fin_hel_vec = collect(canon_fin_hel_arr)
            all_perms_fin = _find_all_permutations(canon_fin_mom_vec, trans_fin_mom)
            found_p_fin = nothing
            for p_fin in all_perms_fin
                _is_within_species_permutation(p_fin, fin_species_for_is) || continue
                hel_ok = true
                for i in 1:Nfm
                    canon_fin_hel_vec[p_fin[i]] != trans_fin_hel[i] && (hel_ok = false; break)
                end
                hel_ok && (found_p_fin = p_fin; break)
            end
            found_p_fin === nothing && continue
        else
            fin_row_base = 1
            found_p_fin = Int[]
        end

        # Construct full permutation p_total: identity on ZM and found_p_fin mapping on FM.
        p_total = Vector{Int}(undef, N)
        for gp in zm_global_positions
            p_total[gp] = gp
        end
        for (i, gp) in enumerate(fm_global_positions)
            p_total[gp] = fm_global_positions[found_p_fin[i]]
        end
        _is_within_species_permutation(p_total, species) || continue

        ferm_sign = _multi_species_permutation_sign(p_total, species, particle_types)
        per_s_idx = _decompose_multi_species_permutation(p_total, species)

        # Traversespin indices
        for sigma_idx in 1:nσ
            sigma = spin_tuples[sigma_idx]

            for sigmap_idx in 1:nσ
                sigmap = spin_tuples[sigmap_idx]

                # D product: every ZM particle uses the Wigner D matrix of its own species.
                Dprod = ComplexF64(1.0, 0.0)
                for (zm_local, sp_k) in enumerate(zm_particle_species)
                    if spins[sp_k] != 0.0
                        Dj_g = wigner_Ds_by_species[sp_k][g_idx]
                        to_idx = spin_to_dj_by_species[sp_k]
                        Dprod *= Dj_g[to_idx[sigmap[zm_local]], to_idx[sigma[zm_local]]]
                    end
                end
                abs(Dprod) < 1e-14 && continue

                σ_row_base = (sigmap_idx - 1) * n_fin * dim_kappa

                for a in 1:dim_kappa, b in 1:dim_kappa
                    R_ba = _multi_R_matrix_element(species, κ_tuple, per_s_idx, a, b)
                    abs(R_ba) < 1e-14 && continue

                    for nu in 1:dim_Gamma
                        c_idx = (a - 1) * dim_Gamma * nσ + (nu - 1) * nσ + sigma_idx

                        for r in 1:n_r
                            c_val = C[c_idx, r]
                            abs(c_val) < 1e-14 && continue

                            Dstar = conj(Dg_Gamma[1, nu])
                            norm_factor = sqrt(dim_Gamma / (nG * Z[r]))
                            term = norm_factor * fin_phase * ComplexF64(ferm_sign) *
                                   Dstar * c_val * Dprod * R_ba

                            row = σ_row_base + fin_row_base + b - 1
                            X[row, r] += term
                        end
                    end
                end
            end
        end
    end

    return X
end

# ============ Canonical-basis overlap matrix S ============

"""
    build_S_matrix(subspace_states, N, kappa, species_type; storage=:dense)

Construct overlap matrix S of the canonical basis.

    S_{(n',λ'),b'; (n',λ'),b''} = Σ_{t ∈ Stab({n'}, λ')} δ(t) × R^{[κ]}_{b'b''}(t)

where:
- Stab({n'}, λ') = {t ∈ S_N : n'_{t_i}=n'_i and λ'_{t_i}=λ'_i, ∀i}
- δ(t): always 1 for bosons and sign(t) for fermions.

S is block diagonal, with one block for each canonical state (n', λ'). storage=:dense retains the original dense return value; storage=:sparse returns SparseMatrixCSC and avoids storing off-block zeros.

"""
function build_S_matrix(subspace_states::Vector, N::Int, kappa::String,
                        species_type::Symbol; storage::Symbol=:dense)
    storage in (:dense, :sparse) ||
        throw(ArgumentError("storage must be :dense or :sparse, got :$storage"))
    dim_kappa = get_SN_irrep_dim(N, kappa)
    n_states = length(subspace_states)
    total_dim = n_states * dim_kappa
    S = storage == :dense ? zeros(Float64, total_dim, total_dim) : nothing
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if storage == :sparse
        sizehint!(rows, n_states * dim_kappa^2)
        sizehint!(cols, n_states * dim_kappa^2)
        sizehint!(vals, n_states * dim_kappa^2)
    end
    fermion = (species_type == :fermion)

    for (k_idx, (n_p, lam_p)) in enumerate(subspace_states)
        stab_mom = _find_all_permutations(collect(n_p), collect(n_p))
        blk = zeros(Float64, dim_kappa, dim_kappa)
        for s in stab_mom
            hel_ok = true
            for i in 1:N
                if lam_p[s[i]] != lam_p[i]
                    hel_ok = false; break
                end
            end
            hel_ok || continue
            delta_s = fermion ? Float64(_permutation_sign(s)) : 1.0
            s_idx = get_SN_element_index(N, s)
            blk .+= delta_s .* get_SN_irrep_matrix(N, kappa, s_idx)
        end
        rb = (k_idx - 1) * dim_kappa + 1
        if storage == :dense
            S[rb:rb+dim_kappa-1, rb:rb+dim_kappa-1] .= blk
        else
            for j in 1:dim_kappa, i in 1:dim_kappa
                value = blk[i, j]
                iszero(value) && continue
                push!(rows, rb + i - 1)
                push!(cols, rb + j - 1)
                push!(vals, value)
            end
        end
    end
    return storage == :dense ? S : sparse(rows, cols, vals, total_dim, total_dim)
end

# Multi-species version: stabilizers are restricted to direct-product group S_{N₁}×⋯×S_{Nₖ}.
function build_S_matrix(subspace_states::Vector, species::Vector{Int}, κ_tuple,
                         particle_types::Vector{Symbol};
                         storage::Symbol=:dense)
    storage in (:dense, :sparse) ||
        throw(ArgumentError("storage must be :dense or :sparse, got :$storage"))
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    n_states = length(subspace_states)
    N = sum(species)
    total_dim = n_states * dim_kappa
    S = storage == :dense ? zeros(Float64, total_dim, total_dim) : nothing
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if storage == :sparse
        sizehint!(rows, n_states * dim_kappa^2)
        sizehint!(cols, n_states * dim_kappa^2)
        sizehint!(vals, n_states * dim_kappa^2)
    end

    for (k_idx, (n_p, lam_p)) in enumerate(subspace_states)
        stab_mom = _find_all_permutations(collect(n_p), collect(n_p))
        blk = zeros(Float64, dim_kappa, dim_kappa)
        for s in stab_mom
            _is_within_species_permutation(s, species) || continue
            hel_ok = true
            for i in 1:N
                if lam_p[s[i]] != lam_p[i]
                    hel_ok = false; break
                end
            end
            hel_ok || continue
            delta_s = Float64(_multi_species_permutation_sign(s, species, particle_types))
            per_s_idx = _decompose_multi_species_permutation(s, species)
            for b in 1:dim_kappa, bp in 1:dim_kappa
                blk[bp, b] += delta_s * _multi_R_matrix_element(species, κ_tuple, per_s_idx, b, bp)
            end
        end
        rb = (k_idx - 1) * dim_kappa + 1
        if storage == :dense
            S[rb:rb+dim_kappa-1, rb:rb+dim_kappa-1] .= blk
        else
            for j in 1:dim_kappa, i in 1:dim_kappa
                value = blk[i, j]
                iszero(value) && continue
                push!(rows, rb + i - 1)
                push!(cols, rb + j - 1)
                push!(vals, value)
            end
        end
    end
    return storage == :dense ? S : sparse(rows, cols, vals, total_dim, total_dim)
end

"""
    build_S_matrix_zero_momentum(M, spin_tuples, fin_subspace_states,
                                  N, kappa, species_type) -> Matrix{Float64}

Construct overlap matrix S for the zero-momentum case (unsymmetrized spin basis).

Formula (zero_momentum_new.md Eq.39):
  S_{(σ',n',λ',a'), (σ,n,λ,a)} = Σ_{s∈S_M×S_{N-M}} δ(s) ×
    Π_i δ_{σ'_{s_i},σ_i} × Π_j δ_{n'_{s_{M+j}-M},n_j} × Π_j δ_{λ'_{s_{M+j}-M},λ_j}
    × R_{a'a}(s)

Its row-index convention matches build_X_matrix_zero_momentum (canonical_spin=false).
"""
function build_S_matrix_zero_momentum(M::Int, spin_tuples::Vector,
                                       fin_subspace_states::Vector,
                                       N::Int, kappa::String, species_type::Symbol)
    dim_kappa = get_SN_irrep_dim(N, kappa)
    nσ = length(spin_tuples)
    n_fin = length(fin_subspace_states)
    fermion = (species_type == :fermion)

    S = zeros(Float64, nσ * n_fin * dim_kappa, nσ * n_fin * dim_kappa)

    # List of S_M × S_{N-M} subgroup elements.
    sm_snm = _SM_x_SNM_elements(N, M)

    for (σ_idx, σ) in enumerate(spin_tuples)
        for (fin_idx, (n_p, lam_p)) in enumerate(fin_subspace_states)
            Nfm = length(n_p)
            col_blk_idx = (σ_idx - 1) * n_fin + fin_idx
            col_cb = (col_blk_idx - 1) * dim_kappa + 1

            for (σp_idx, σp) in enumerate(spin_tuples)
                for (finp_idx, (np_p, lamp_p)) in enumerate(fin_subspace_states)
                    row_blk_idx = (σp_idx - 1) * n_fin + finp_idx
                    row_rb = (row_blk_idx - 1) * dim_kappa + 1

                    blk = zeros(Float64, dim_kappa, dim_kappa)

                    for (s_idx, s) in sm_snm
                        # Spin matching: σ'_{s_i} = σ_i, i=1..M.
                        σ_ok = true
                        for i in 1:M
                            σp[s[i]] != σ[i] && (σ_ok = false; break)
                        end
                        σ_ok || continue

                        # Finite-momentum matching: n'_{s_{M+j}-M} = n_j, λ'_{s_{M+j}-M} = λ_j, j=1..Nfm.
                        fin_ok = true
                        for j in 1:Nfm
                            sj = s[M + j] - M  # map to 1..Nfm
                            np_p[sj] != n_p[j] && (fin_ok = false; break)
                            lamp_p[sj] != lam_p[j] && (fin_ok = false; break)
                        end
                        fin_ok || continue

                        delta_s = fermion ? Float64(_permutation_sign(s)) : 1.0
                        blk .+= delta_s .* get_SN_irrep_matrix(N, kappa, s_idx)
                    end

                    S[row_rb:row_rb+dim_kappa-1, col_cb:col_cb+dim_kappa-1] .= blk
                end
            end
        end
    end

    return S
end

# Multi-species version: zero-momentum overlap matrix.
function build_S_matrix_zero_momentum(zero_counts::Vector{Int},
                                       spin_tuples::Vector,
                                       fin_subspace_states::Vector,
                                       species::Vector{Int}, κ_tuple,
                                       particle_types::Vector{Symbol})
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    nσ = length(spin_tuples)
    n_fin = length(fin_subspace_states)
    N = sum(species)
    K = length(species)
    Nfm = N - sum(zero_counts)

    S = zeros(Float64, nσ * n_fin * dim_kappa, nσ * n_fin * dim_kappa)

    # Global ZM / FM positions → local indices.
    zm_global_positions = Int[]
    zm_global_to_local = Dict{Int, Int}()
    fm_global_positions = Int[]
    fm_global_to_local = Dict{Int, Int}()
    for (k, Nk) in enumerate(species)
        off = sum(species[1:k-1]; init=0)
        for i in 1:zero_counts[k]
            gp = off + i
            push!(zm_global_positions, gp)
            zm_global_to_local[gp] = length(zm_global_positions)
        end
        for i in (zero_counts[k] + 1):Nk
            gp = off + i
            push!(fm_global_positions, gp)
            fm_global_to_local[gp] = length(fm_global_positions)
        end
    end

    sm_snm = _multi_SM_SNM_elements(species, zero_counts)

    for (σ_idx, σ) in enumerate(spin_tuples)
        for (fin_idx, (n_p, lam_p)) in enumerate(fin_subspace_states)
            col_blk_idx = (σ_idx - 1) * n_fin + fin_idx
            col_cb = (col_blk_idx - 1) * dim_kappa + 1

            for (σp_idx, σp) in enumerate(spin_tuples)
                for (finp_idx, (np_p, lamp_p)) in enumerate(fin_subspace_states)
                    row_blk_idx = (σp_idx - 1) * n_fin + finp_idx
                    row_rb = (row_blk_idx - 1) * dim_kappa + 1

                    blk = zeros(Float64, dim_kappa, dim_kappa)

                    for (s_full, per_s_idx) in sm_snm
                        # ZM spin matching: σ'_{s_full[gp]} = σ_{gp}.
                        σ_ok = true
                        for gp in zm_global_positions
                            zm_local = zm_global_to_local[gp]
                            tgt_gp = s_full[gp]
                            tgt_zm_local = zm_global_to_local[tgt_gp]
                            σp[tgt_zm_local] != σ[zm_local] && (σ_ok = false; break)
                        end
                        σ_ok || continue

                        # FM matching: n'_{s_full[gp]} = n_{gp}, λ'_{...} = λ_{...}.
                        fin_ok = true
                        for gp in fm_global_positions
                            fm_local = fm_global_to_local[gp]
                            tgt_gp = s_full[gp]
                            tgt_fm_local = fm_global_to_local[tgt_gp]
                            if Nfm > 0
                                np_p[tgt_fm_local] != n_p[fm_local] && (fin_ok = false; break)
                                lamp_p[tgt_fm_local] != lam_p[fm_local] && (fin_ok = false; break)
                            end
                        end
                        fin_ok || continue

                        delta_s = Float64(_multi_species_permutation_sign(
                            s_full, species, particle_types))
                        for b in 1:dim_kappa, bp in 1:dim_kappa
                            blk[bp, b] += delta_s *
                                _multi_R_matrix_element(species, κ_tuple, per_s_idx, b, bp)
                        end
                    end

                    S[row_rb:row_rb+dim_kappa-1, col_cb:col_cb+dim_kappa-1] .= blk
                end
            end
        end
    end

    return S
end

# ============ High-level API ============

"""
    subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                        d_total, species_type=:boson, spin, etas)

Run the complete projection pipeline for one subspace: I matrix → Löwdin → X matrix.

# Arguments
- n_tuple: representative momentum (N-tuple)
- lambda_tuple: helicity configuration (N-tuple)
- kappa: S_N irrep label, for example "[2]" or "[1,1]"
- Gamma: target-group irrep, for example "A1", "E", or "T1+"
- d_total: total momentum (used to determine the symmetry group)
- species_type: :boson or :fermion
- spin: single-particle spin
- etas: single-particle intrinsic parity for each species, for example [1.0] or [-1.0]

# Returns
- Z: nonzero-eigenvalue vector
- X: coefficient matrix (subspace dimension × number of nonzero eigenvalues); column r corresponds to |Γ, r⟩ (μ=1)
- subspace_states: canonical-basis-state list
- I_evals: all I-matrix eigenvalues (including zero, for idempotency checks)
"""
function subspace_projection(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                              kappa::String, Gamma::String;
                              d_total::Momentum=D000,
                              species_type::Symbol=:boson,
                              spin::Float64, etas::Vector{Float64},
                              prepared_orbit=nothing) where N
    # Detect zero-momentum + nonzero-spin particles → zero-momentum pipeline.
    M = count(n -> n == Momentum(0,0,0), n_tuple)
    if M > 0 && spin != 0.0
        return _subspace_projection_zero(M, n_tuple, lambda_tuple, kappa, Gamma;
                                          d_total=d_total, species_type=species_type,
                                          spin=spin, etas=etas)
    end

    # Determine symmetry group (fermions use the double cover).
    needs_double = (species_type == :fermion)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)

    nG = length(group_els)  # full group size |G|
    n_base = needs_double ? nG ÷ 2 : nG  # base O(3) group size

    orbit = prepared_orbit === nothing ?
        _prepare_projection_orbit(n_tuple, lambda_tuple, group_els, n_base) :
        prepared_orbit

    # 1. I matrix
    I = build_I_matrix(n_tuple, lambda_tuple, kappa, Gamma,
                        group_els, irrep_mats, species_type, spin, etas, n_base;
                        prepared_orbit=orbit)

    # 2. Löwdin orthogonalization (also return all eigenvalues to avoid repeated eigendecomposition).
    Z, C, _, I_evals = lowdin_orthogonalize(I)

    # 3. X matrix.
    X = build_X_matrix(n_tuple, lambda_tuple, kappa, Gamma,
                        group_els, irrep_mats, species_type, spin, etas, n_base, Z, C;
                        prepared_orbit=orbit)

    # 4. Subspace-state list (consistent with internal build_X_matrix calls).
    subspace_states = orbit.subspace_states

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

# ============ Multi-species subspace_projection ============

"""
    _canonicalize_zm_ordering(n_tuple, lambda_tuple, species)

Sort particles of every species: zero-momentum particles first and finite-momentum particles last, while reordering helicities accordingly.
Return (sorted_n, sorted_lam, zero_counts).

The ZM-pipeline functions require ZM-first ordering for each species. This helper is called automatically inside _subspace_projection_zero; users need not handle it manually.

"""
function _canonicalize_zm_ordering(n_tuple::NTuple{N,Momentum},
                                   lambda_tuple::NTuple{N,Float64},
                                   species::Vector{Int}) where N
    sorted_n = Vector{Momentum}(undef, N)
    sorted_lam = Vector{Float64}(undef, N)
    zero_counts = Int[]
    off = 0
    for (k, Nk) in enumerate(species)
        sp_start = sum(species[1:k-1]; init=0) + 1
        sp_zm = Int[p for p in sp_start:sp_start+Nk-1 if iszero(n_tuple[p])]
        sp_fm = Int[p for p in sp_start:sp_start+Nk-1 if !iszero(n_tuple[p])]
        push!(zero_counts, length(sp_zm))
        for (j, p) in enumerate(sp_zm)
            sorted_n[off + j] = n_tuple[p]
            sorted_lam[off + j] = lambda_tuple[p]
        end
        for (j, p) in enumerate(sp_fm)
            sorted_n[off + length(sp_zm) + j] = n_tuple[p]
            sorted_lam[off + length(sp_zm) + j] = lambda_tuple[p]
        end
        off += Nk
    end
    return Tuple(sorted_n), Tuple(sorted_lam), zero_counts
end

"""
    subspace_projection(n_tuple, lambda_tuple, κ_tuple, Gamma;
                        d_total, species, particle_types, spins, etas)

Multi-species projection pipeline: I matrix → Löwdin orthogonalization → X matrix.
Automatically switch to the ZM pipeline when a species contains a zero-momentum particle with nonzero spin.
"""
function subspace_projection(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                             κ_tuple, Gamma::String;
                             d_total::Momentum=D000,
                             species::Vector{Int}, particle_types::Vector{Symbol},
                             spins::Vector{Float64}, etas::Vector{Float64}) where N
    # Detect zero-momentum + nonzero-spin particles → zero-momentum pipeline.
    M_total = 0
    has_zm_spin = false
    off = 0
    for k in 1:length(species)
        zm_k = 0
        for i in 1:species[k]
            if iszero(n_tuple[off + i])
                zm_k += 1
            end
        end
        M_total += zm_k
        if zm_k > 0 && spins[k] != 0.0
            has_zm_spin = true
        end
        off += species[k]
    end
    if M_total > 0 && has_zm_spin
        return _subspace_projection_zero(n_tuple, lambda_tuple, κ_tuple, Gamma;
                                          d_total=d_total, species=species,
                                          particle_types=particle_types,
                                          spins=spins, etas=etas)
    end

    # Finite-momentum pipeline.
    needs_double = any(pt -> pt == :fermion, particle_types)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)

    nG = length(group_els)
    n_base = needs_double ? nG ÷ 2 : nG

    I = build_I_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                        group_els, irrep_mats, species, particle_types, spins, etas, n_base)
    Z, C, _, I_evals = lowdin_orthogonalize(I)

    X = build_X_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                        group_els, irrep_mats, species, particle_types, spins, etas, n_base, Z, C)

    subspace_states = _collect_subspace_states(n_tuple, lambda_tuple, group_els,
                                                species, spins, etas, n_base)

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

"""
    _subspace_projection_zero(n_tuple, lambda_tuple, κ_tuple, Gamma; ...)

Multi-species zero-momentum particle projection pipeline.
Automatically normalize input to ZM-first canonical order, then call
build_I_matrix_zero_momentum → Löwdin → build_X_matrix_zero_momentum。
"""

"""
    _subspace_projection_zero(M, n_tuple, lambda_tuple, kappa, Gamma; ...)

Zero-momentum particle projection pipeline: build_I_matrix_zero_momentum → Löwdin → build_X_matrix_zero_momentum。

Use convention R_st(0)=I (zero-momentum direction is undefined, so use the identity rotation); spin projection σ of zero-momentum particles therefore serves as its "helicity" label. Returned subspace_states use standard (n_full, λ_full) tuples and are fully compatible with build_V_hel.
"""
function _subspace_projection_zero(M::Int, n_tuple::NTuple{N,Momentum},
                                    lambda_tuple::NTuple{N,Float64},
                                    kappa::String, Gamma::String;
                                    d_total::Momentum,
                                    species_type::Symbol,
                                    spin::Float64, etas::Vector{Float64}) where N
    j = Rational{Int}(spin)

    needs_double = (species_type == :fermion)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)
    nG = length(group_els)
    n_base = needs_double ? nG ÷ 2 : nG

    I = build_I_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                      group_els, irrep_mats, species_type, spin, etas, n_base)
    Z, C, _, I_evals = lowdin_orthogonalize(I)

    # Unsymmetrized spin basis: all (2j+1)^M spin configurations, consistent with build_V_hel.
    X = build_X_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                      group_els, irrep_mats, species_type, spin,
                                      etas, n_base, Z, C; canonical_spin=false)

    # Construct standard (n_full, λ_full) format, compatible with build_V_hel.
    # R_st(0)=I → wigner_D(j,0)=I → rotation coefficients reduce to δ_{σ',σ}
    # Zero-momentum particle λ labels take all spin-projection values (unsymmetrized), consistent with the canonical-polarization basis.
    all_spins = _spin_tuples(j, M)
    Nfm = N - M
    if Nfm > 0
        fin_n = ntuple(i -> n_tuple[M + i], Nfm)
        fin_lam = ntuple(i -> lambda_tuple[M + i], Nfm)
        fin_states = _collect_fin_subspace_states(fin_n, fin_lam, group_els, spin, etas[1], n_base)
    else
        fin_states = [(Tuple{}(), Tuple{}())]
    end

    zero_mom = Momentum(0, 0, 0)
    zero_n = ntuple(_ -> zero_mom, M)
    subspace_states = []
    for σ in all_spins
        σ_float = Tuple(Float64.(σ))
        for (fn, fl) in fin_states
            full_n = (zero_n..., fn...)
            full_lam = (σ_float..., fl...)
            push!(subspace_states, (full_n, full_lam))
        end
    end

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

# ============ Multi-species _subspace_projection_zero ============

function _subspace_projection_zero(n_tuple::NTuple{N,Momentum},
                                   lambda_tuple::NTuple{N,Float64},
                                   κ_tuple, Gamma::String;
                                   d_total::Momentum,
                                   species::Vector{Int}, particle_types::Vector{Symbol},
                                   spins::Vector{Float64}, etas::Vector{Float64}) where N
    # Canonical ordering: ZM particles precede FM particles（by species）
    sorted_n, sorted_lam, zero_counts = _canonicalize_zm_ordering(
        n_tuple, lambda_tuple, species)
    M_total = sum(zero_counts)

    needs_double = any(pt -> pt == :fermion, particle_types)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)
    nG = length(group_els)
    n_base = needs_double ? nG ÷ 2 : nG

    I = build_I_matrix_zero_momentum(zero_counts, sorted_n, sorted_lam, κ_tuple, Gamma,
                                      group_els, irrep_mats, species, particle_types,
                                      spins, etas, n_base)
    Z, C, _, I_evals = lowdin_orthogonalize(I)

    X = build_X_matrix_zero_momentum(zero_counts, sorted_n, sorted_lam, κ_tuple, Gamma,
                                      group_els, irrep_mats, species, particle_types,
                                      spins, etas, n_base, Z, C)

    # Construct standard (n_full, λ_full) subspace-state format
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    Nfm = N - M_total
    if Nfm > 0
        # Extract from each species FM part: sorted_n/sorted_lam preserve species grouping，
        # Within each species, ZM precedes FM; cannot use sorted_n[M_total+1:end] global slice.
        fin_mom_vec = Momentum[]
        fin_lam_vec = Float64[]
        off = 0
        for k in 1:length(species)
            M_k = zero_counts[k]
            Nk = species[k]
            for i in (M_k + 1):Nk
                push!(fin_mom_vec, sorted_n[off + i])
                push!(fin_lam_vec, sorted_lam[off + i])
            end
            off += Nk
        end
        fin_n = Tuple(fin_mom_vec)
        fin_lam = Tuple(fin_lam_vec)
        fin_species_vec = Int[species[k] - zero_counts[k] for k in 1:length(species)]
        fin_species_vec = Int[s for s in fin_species_vec if s > 0]
        fin_spins = Float64[spins[k] for k in 1:length(species) if species[k] > zero_counts[k]]
        fin_etas = Float64[etas[k] for k in 1:length(species) if species[k] > zero_counts[k]]
        fin_states = isempty(fin_species_vec) ? [(Tuple{}(), Tuple{}())] :
            _collect_fin_subspace_states(fin_n, fin_lam, group_els,
                                         fin_species_vec, fin_spins, fin_etas, n_base)
    else
        fin_states = [(Tuple{}(), Tuple{}())]
    end

    # Construct (n, λ) by species: ZM precedes FM within every species, preserving species grouping.
    # This ensures per_spin (expanded by species) aligns with full_n/full_lam indices.
    zero_mom = Momentum(0, 0, 0)
    n_species = length(species)
    subspace_states = []
    for σ in spin_tuples
        for (fn, fl) in fin_states
            full_n_vec = Momentum[]
            full_lam_vec = Float64[]
            z_off = 0
            f_off = 0
            for k in 1:n_species
                M_k = zero_counts[k]
                Nk = species[k]
                Fk = Nk - M_k
                # ZM particle (species k)
                for j in 1:M_k
                    push!(full_n_vec, zero_mom)
                    push!(full_lam_vec, σ[z_off + j])
                end
                z_off += M_k
                # FM particle (species k).
                for j in 1:Fk
                    push!(full_n_vec, fn[f_off + j])
                    push!(full_lam_vec, fl[f_off + j])
                end
                f_off += Fk
            end
            push!(subspace_states, (Tuple(full_n_vec), Tuple(full_lam_vec)))
        end
    end

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

"""
    project_interaction(V::AbstractMatrix, X_left::AbstractMatrix, X_right::AbstractMatrix)
        -> Matrix

Interaction-matrix projection: V^Γ = X_left^† × V × X_right
"""
function project_interaction(V::AbstractMatrix, X_left::AbstractMatrix, X_right::AbstractMatrix)
    return X_left' * V * X_right
end

"""
    project_V(X_left, X_right, subspace_states_left, subspace_states_right,
              per_spin_left, per_spin_right, L, V_can_func, extra_args...)
        -> Matrix

Complete projection pipeline: V_can → V_hel → X_left^† × V_hel × X_right

# Arguments
- `X_left, X_right`: X matrices returned by `subspace_projection` or `build_X_matrix`
- `subspace_states_left/right`: subspace basis-state lists (each entry `(n_tuple, λ_tuple)`)
- `per_spin_left/right`: per-particle spins
- `L`: finite-volume size
- `V_can_func(n'_tuple, σ'_tuple, n_tuple, σ_tuple, extra_args...)`: V matrix element in canonical-polarization basis
- `extra_args...`: forwarded to V_can_func

Automatically multiply by (2π ħc/L)^{d/2} factor, d = 3(N_α + N_β) - 6。
"""
function project_V(X_left::AbstractMatrix, X_right::AbstractMatrix,
                   subspace_states_left::Vector,
                   subspace_states_right::Vector,
                   per_spin_left::AbstractVector{<:Real},
                   per_spin_right::AbstractVector{<:Real},
                   L::Real,
                   V_can_func::Function, extra_args...)
    # Fourier factor: (2π ħc/L)^{d/2}, d = 3(N_α + N_β) - 6
    # N_α, N_β obtained from momentum-tuple lengths of first states in subspace_states.
    N_α = length(first(subspace_states_left)[1])
    N_β = length(first(subspace_states_right)[1])
    d = 3 * (N_α + N_β) - 6
    fv_factor = (2π * ħc / L)^(d / 2)

    V_hel = build_V_hel(subspace_states_left, subspace_states_right,
                        per_spin_left, per_spin_right,
                        V_can_func, extra_args...)
    return fv_factor * (X_left' * V_hel * X_right)
end

"""
    project_V_hel(X_left, X_right, subspace_states_left, subspace_states_right,
                  L, V_hel_adapter, extra_args...) -> Matrix{ComplexF64}

Helicity-basis version of project_V. Does not require per_spin; directly calls build_V_hel_direct to fill V_hel.
"""
function project_V_hel(X_left::AbstractMatrix, X_right::AbstractMatrix,
                       subspace_states_left::Vector,
                       subspace_states_right::Vector,
                       L::Real,
                       V_hel_adapter::Function, extra_args...)
    N_α = length(first(subspace_states_left)[1])
    N_β = length(first(subspace_states_right)[1])
    d = 3 * (N_α + N_β) - 6
    fv_factor = (2π * ħc / L)^(d / 2)

    V_hel = build_V_hel_direct(subspace_states_left, subspace_states_right,
                               V_hel_adapter, extra_args...)
    return fv_factor * (X_left' * V_hel * X_right)
end
