# ============================================================
# Hamiltonian — Projected-Hamiltonian orchestration
# ============================================================
# Three-level block hierarchy:
#   Level 1: (rep_α,lam_α) × (rep_β,lam_β) → project_V atomic block
#   Level 2: subchannel pairs (κ_α,r_α, κ_β,r_β, Γ) → Level 1 block assembly
#   Level 3: channel pair (α,β) → Level 2 block assembly → final full matrix
# ============================================================

const _PROJ_CACHE = Dict{NamedTuple, @NamedTuple{
    X::Matrix{ComplexF64}, states::Vector, Z::Vector{Float64}}}()
const _PROJ_LIST_CACHE = Dict{NamedTuple, Vector}()
const _PROJ_ORBIT_CACHE = Dict{NamedTuple, _PreparedProjectionOrbit}()
const _PROJ_ORBIT_CACHE_LOCK = ReentrantLock()

function _clear_projection_orbit_cache!()
    lock(_PROJ_ORBIT_CACHE_LOCK) do
        empty!(_PROJ_ORBIT_CACHE)
    end
    return nothing
end

function _projection_orbit_cache_size()
    return lock(_PROJ_ORBIT_CACHE_LOCK) do
        length(_PROJ_ORBIT_CACHE)
    end
end

function _get_prepared_projection_orbit(
        n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
        d::Momentum, species_type::Symbol) where N
    needs_double = species_type == :fermion
    key = (n_tuple=n_tuple, lambda_tuple=lambda_tuple,
           d=d, double_cover=needs_double)
    cached = lock(_PROJ_ORBIT_CACHE_LOCK) do
        get(_PROJ_ORBIT_CACHE, key, nothing)
    end
    cached === nothing || return cached

    group_els, _ = group_for_momentum(d; double_cover=needs_double)
    n_base = needs_double ? length(group_els) ÷ 2 : length(group_els)
    orbit = _prepare_projection_orbit(
        n_tuple, lambda_tuple, group_els, n_base)
    return lock(_PROJ_ORBIT_CACHE_LOCK) do
        get!(_PROJ_ORBIT_CACHE, key, orbit)
    end
end

# ============ Optimized pipeline: geometry cache shared across irreps ============

"""
    ProjBlockData

Projection data for one (rep, lam) block in irrep Γ.
X The X matrix has size (n_states × n_r)，row_indices row_indices are the indices of these states in the unified basis.
"""
struct ProjBlockData
    X::Matrix{ComplexF64}
    n_r::Int
    row_indices::Vector{Int}
end

"""
    SubBasisData

Unified-basis data for one (channel, subchannel):
- states / state_to_idx: union of all (n_tuple, lambda_tuple) states appearing in all irreps
- per_spin, per_mass, kinetic_type: per-particle spin/mass/dispersion
- rot_coeffs: precomputed rotation coefficients (one vector per state)
- proj_by_irrep: each irrep → ProjBlockData list
"""
struct SubBasisData
    kappa
    r_val::Int
    dim_kappa::Int
    states::Vector
    state_to_idx::Dict
    per_spin::Vector{Rational{Int}}
    per_mass::Vector{Float64}
    kinetic_type::KineticType
    rot_coeffs::Vector{Vector{ComplexF64}}
    proj_by_irrep::Dict{String, Vector{ProjBlockData}}
end

"""
    SystemBasis <: Any

Geometry cache for a FockSystem. It contains state lists and rotation coefficients shared by all irreps,
as well as projection matrices for each irrep. It is independent of V_func/params and need only be rebuilt when (L,a,Ncut) changes.

After construction, call build_V_hel_blocks! to fill interaction matrices, then use project_and_diag to solve eigenvalues irrep by irrep.
"""
mutable struct SystemBasis
    sys::FockSystem
    L_phys::Float64
    chan_sub_data::Vector{Vector{SubBasisData}}
    irrep_total_dim::Dict{String, Int}
    # V_hel cache: (α, sα, β, sβ) → AbstractMatrix{ComplexF64}
    V_hel_blocks::Dict{Tuple{Int,Int,Int,Int}, AbstractMatrix{ComplexF64}}
end

function Base.show(io::IO, basis::SystemBasis)
    sys = basis.sys
    n_ch = length(sys.channels)
    chan_names = [ch.name for ch in sys.channels]

    # States per channel
    chan_nstates = Int[]
    for (α, sd_list) in enumerate(basis.chan_sub_data)
        total = sum(length(sd.states) for sd in sd_list; init=0)
        push!(chan_nstates, total)
    end

    println(io, "SystemBasis:")
    println(io, "  d=$(sys.d), I=$(sys.I), L=$(sys.L), a=$(sys.a)")
    print(io,   "  channels: ")
    for α in 1:n_ch
        print(io, "$(chan_names[α])=$(chan_nstates[α])")
        α < n_ch && print(io, ", ")
    end
    print(io, " states")
    has_V = !isempty(basis.V_hel_blocks)
    has_V && print(io, " [V_hel filled]")
    println(io)
    if !isempty(basis.irrep_total_dim)
        println(io, "  irrep dims: ", basis.irrep_total_dim)
    end
end

"""
    SystemBasis(sys::FockSystem) -> SystemBasis

Build the geometry cache from FockSystem. Collect unified state bases for every channel and subchannel across all irreps,
and precompute rotation coefficients and projection matrices for each irrep.
"""
function SystemBasis(sys::FockSystem;
                     exclude_subchannels::AbstractVector{SubchannelExclusion}=SubchannelExclusion[])
    L_phys = Float64(sys.L) * sys.a
    n_ch = length(sys.channels)

    # Moving frames currently support only N ≤ 2
    if sys.d != D000
        for ch in sys.channels
            ch.N > 2 && throw(ArgumentError(
                "moving frames (d≠0) currently support only channels with N≤2; channel \"$(ch.name)\" has N=$(ch.N)"))
        end
    end

    chan_sub_data = Vector{Vector{SubBasisData}}(undef, n_ch)
    irrep_total_dim = Dict{String, Int}(Gamma => 0 for Gamma in sys.selected_irreps)

    for α in 1:n_ch
        ch = sys.channels[α]
        ncut = get_Ncut(sys, α)
        per_spin_raw, per_etas_arr = _expand_per_particle(ch)
        per_spin_v = Rational{Int}.(per_spin_raw)
        etas_v = Float64.(per_etas_arr)
        multi = length(ch.species) > 1
        per_mass = has_dynamic_mass(ch) ? Float64[] : _expand_per_particle_mass(ch)

        subs = _active_subchannels(ch, sys.I, exclude_subchannels)
        sub_list = Vector{SubBasisData}(undef, length(subs))

        for (si, sub) in enumerate(subs)
            # 1) Collect all states from all irreps (union).
            ST = Tuple{NTuple{ch.N, Momentum}, NTuple{ch.N, Float64}}
            all_states = Vector{ST}()
            state_to_idx = Dict{ST, Int}()
            for Gamma in sys.selected_irreps
                projs = if multi
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
                else
                    spin_val = Float64(only(unique(per_spin_v)))
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
                end
                for p in projs
                    for st in p.states
                        if !haskey(state_to_idx, st)
                            push!(all_states, st)
                            state_to_idx[st] = length(all_states)
                        end
                    end
                end
            end

            # 2) Precompute rotation coefficients.
            rot_coeffs = [get_rotation_vector(n, lam, per_spin_v) for (n, lam) in all_states]

            # 3) Collect projection blocks for each irrep.
            proj_by_irrep = Dict{String, Vector{ProjBlockData}}()
            for Gamma in sys.selected_irreps
                projs = if multi
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
                else
                    spin_val = Float64(only(unique(per_spin_v)))
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
                end
                blocks = ProjBlockData[]
                dim_κ = sub.dim
                for p in projs
                    spatial_idx = [state_to_idx[st] for st in p.states]
                    row_idx = Int[(i - 1) * dim_κ + a for i in spatial_idx for a in 1:dim_κ]
                    push!(blocks, ProjBlockData(p.X, p.n_r, row_idx))
                end
                irrep_total_dim[Gamma] += sum(b.n_r for b in blocks; init=0)
                proj_by_irrep[Gamma] = blocks
            end

            sub_list[si] = SubBasisData(sub.κ, sub.r, sub.dim,
                                        all_states, state_to_idx, per_spin_v,
                                        per_mass, ch.kinetic_type, rot_coeffs, proj_by_irrep)
        end

        chan_sub_data[α] = sub_list
    end

    return SystemBasis(sys, L_phys, chan_sub_data, irrep_total_dim,
                       Dict{Tuple{Int,Int,Int,Int}, AbstractMatrix{ComplexF64}}())
end

function _resolve_basis_masses!(basis::SystemBasis, params)
    for (channel_index, ch) in enumerate(basis.sys.channels)
        has_dynamic_mass(ch) || continue
        resolved = resolve_particle_masses(ch, params)
        for subbasis in basis.chan_sub_data[channel_index]
            empty!(subbasis.per_mass)
            append!(subbasis.per_mass, resolved)
        end
    end
    return basis
end

"""
    build_V_hel_blocks!(basis::SystemBasis, V_func, params) -> SystemBasis

Build complete V_hel matrices in unified state bases for every (sub_α, sub_β) channel-pair
and store them in basis.V_hel_blocks。All irreps share these matrices; only projection is required afterwards.

Call again only when V_func or params changes. Reuse the basis (geometry cache) unchanged.
"""
function build_V_hel_blocks!(basis::SystemBasis, V_func::Function, params;
                             V_basis::Symbol=:canonical,
                             channel_filter=nothing,
                             entry_filter=nothing,
                             validate_hermitian::Bool=false,
                             validation_blocks::Int=3,
                             validation_rtol::Float64=1e-10,
                             validation_atol::Float64=1e-12,
                             _resolve_masses::Bool=true)
    _resolve_masses && _resolve_basis_masses!(basis, params)
    empty!(basis.V_hel_blocks)
    validation_blocks >= 0 || throw(ArgumentError("validation_blocks must be nonnegative"))
    n_ch = length(basis.sys.channels)

    # Collect all subchannel-pair work items.
    tasks = Tuple{Int,Int,SubBasisData,Int,Int,SubBasisData}[]
    for α in 1:n_ch
        ch_α = basis.sys.channels[α]
        sub_data_α = basis.chan_sub_data[α]
        for (sα, sd_α) in enumerate(sub_data_α)
            length(sd_α.states) == 0 && continue
            for β in 1:n_ch
                ch_β = basis.sys.channels[β]
                sub_data_β = basis.chan_sub_data[β]
                for (sβ, sd_β) in enumerate(sub_data_β)
                    length(sd_β.states) == 0 && continue
                    (α, sα) > (β, sβ) && continue
                    push!(tasks, (α, sα, sd_α, β, sβ, sd_β))
                end
            end
        end
    end

    # Schedule channel pairs serially; parallelism is internal to build_V_hel / build_V_hel_direct  .
    n_tasks = length(tasks)
    results = Vector{AbstractMatrix{ComplexF64}}(undef, n_tasks)
    keys    = Vector{Tuple{Int,Int,Int,Int}}(undef, n_tasks)

    for i in 1:n_tasks
        α, sα, sd_α, β, sβ, sd_β = tasks[i]
        V_adapted = _V_adapter(V_func, sd_α.kappa, sd_β.kappa,
                                   sd_α.r_val, sd_β.r_val,
                                   sd_α.dim_kappa, sd_β.dim_kappa,
                                   α, β, basis.L_phys, params)
        filter_adapted = if entry_filter !== nothing
            (n_α, lam_or_sig_α, n_β, lam_or_sig_β, extra...) ->
                entry_filter(n_α, n_β, lam_or_sig_α, lam_or_sig_β,
                             sd_α.kappa, sd_β.kappa,
                             sd_α.r_val, sd_β.r_val,
                             α, β, basis.L_phys, params)
        else
            nothing
        end
        if channel_filter !== nothing && !Bool(channel_filter(α, β, params))
            results[i] = spzeros(ComplexF64, 0, 0)
            keys[i] = (α, sα, β, sβ)
            continue
        end
        results[i] = if V_basis == :helicity
            build_V_hel_direct(sd_α.states, sd_β.states, V_adapted;
                               entry_filter=filter_adapted)
        else
            build_V_hel(sd_α.states, sd_β.states,
                        sd_α.per_spin, sd_β.per_spin, V_adapted;
                        entry_filter=filter_adapted)
        end
        keys[i] = (α, sα, β, sβ)
    end

    # Optional debugging check: first validate filter exchange symmetry independently, then validate unfiltered V.
    if validate_hermitian
        n_checked = 0
        for i in 1:n_tasks
            n_checked >= validation_blocks && break
            α, sα, sd_α, β, sβ, sd_β = tasks[i]
            channel_filter === nothing ||
                _validate_channel_filter_hermiticity(
                    channel_filter, α, β, params)
            entry_filter === nothing ||
                _validate_entry_filter_hermiticity(
                    entry_filter, sd_α, sd_β, α, β,
                    basis.L_phys, params, V_basis)

            V_forward = _V_adapter(V_func, sd_α.kappa, sd_β.kappa,
                                   sd_α.r_val, sd_β.r_val,
                                   sd_α.dim_kappa, sd_β.dim_kappa,
                                   α, β, basis.L_phys, params)
            forward = if channel_filter === nothing && entry_filter === nothing
                results[i]
            elseif V_basis == :helicity
                build_V_hel_direct(sd_α.states, sd_β.states, V_forward)
            else
                build_V_hel(sd_α.states, sd_β.states,
                            sd_α.per_spin, sd_β.per_spin, V_forward)
            end

            if (α, sα) == (β, sβ)
                isapprox(forward, adjoint(forward);
                         rtol=validation_rtol, atol=validation_atol) ||
                    throw(ArgumentError("interaction is not Hermitian in block ($α,$sα)"))
            else
                V_reverse = _V_adapter(V_func, sd_β.kappa, sd_α.kappa,
                                       sd_β.r_val, sd_α.r_val,
                                       sd_β.dim_kappa, sd_α.dim_kappa,
                                       β, α, basis.L_phys, params)
                reverse = if V_basis == :helicity
                    build_V_hel_direct(sd_β.states, sd_α.states, V_reverse)
                else
                    build_V_hel(sd_β.states, sd_α.states,
                                sd_β.per_spin, sd_α.per_spin, V_reverse)
                end
                isapprox(reverse, adjoint(forward);
                         rtol=validation_rtol, atol=validation_atol) ||
                    throw(ArgumentError(
                        "interaction violates Hermiticity between blocks " *
                        "($α,$sα) and ($β,$sβ)"))
            end
            n_checked += 1
        end
    end

    # Fill the dictionary serially (avoid concurrent Dict writes).
    for i in 1:n_tasks
        basis.V_hel_blocks[keys[i]] = results[i]
    end

    return basis
end

function _validate_channel_filter_hermiticity(channel_filter,
                                              α::Int, β::Int, params)
    fwd = Bool(channel_filter(α, β, params))
    rev = Bool(channel_filter(β, α, params))
    fwd == rev || throw(ArgumentError(
        "channel_filter violates exchange symmetry for channels $α ↔ $β"))
    return nothing
end

function _validate_entry_filter_hermiticity(entry_filter, sd_α::SubBasisData,
                                            sd_β::SubBasisData, α::Int, β::Int,
                                            L_phys::Float64, params, V_basis::Symbol)
    inputs_α = if V_basis == :helicity
        sd_α.states
    else
        momenta = unique(first.(sd_α.states))
        sigmas = _σ_configurations(sd_α.per_spin)
        [(n, σ) for n in momenta for σ in sigmas]
    end
    inputs_β = if V_basis == :helicity
        sd_β.states
    else
        momenta = unique(first.(sd_β.states))
        sigmas = _σ_configurations(sd_β.per_spin)
        [(n, σ) for n in momenta for σ in sigmas]
    end

    for (nA, qA) in inputs_α, (nB, qB) in inputs_β
        fwd = Bool(entry_filter(nA, nB, qA, qB,
                                sd_α.kappa, sd_β.kappa,
                                sd_α.r_val, sd_β.r_val,
                                α, β, L_phys, params))
        rev = Bool(entry_filter(nB, nA, qB, qA,
                                sd_β.kappa, sd_α.kappa,
                                sd_β.r_val, sd_α.r_val,
                                β, α, L_phys, params))
        fwd == rev || throw(ArgumentError(
            "entry_filter violates exchange symmetry between blocks " *
            "($α,$(sd_α.r_val)) and ($β,$(sd_β.r_val))"))
    end
    return nothing
end

@inline function _get_V_hel_block(basis::SystemBasis,
                                  α::Int, sα::Int, β::Int, sβ::Int)
    V = get(basis.V_hel_blocks, (α, sα, β, sβ), nothing)
    if V === nothing
        reverse = get(basis.V_hel_blocks, (β, sβ, α, sα), nothing)
        reverse === nothing || return adjoint(reverse)
    end
    return V
end

# ============ Optimized pipeline: irrep projection + eigenvalues ============

"""
    _assemble_irrep_hamiltonian(basis::SystemBasis, Gamma::String) -> Matrix{ComplexF64}

Assemble the complete projected Hamiltonian matrix for irrep Γ from SystemBasis V_hel_blocks and projection matrices.
"""
function _assemble_irrep_hamiltonian(basis::SystemBasis, Gamma::String)
    dim = basis.irrep_total_dim[Gamma]
    dim == 0 && return zeros(ComplexF64, 0, 0)

    H = zeros(ComplexF64, dim, dim)
    n_ch = length(basis.sys.channels)
    L_phys = basis.L_phys

    row_start = 1
    for α in 1:n_ch
        sub_data_α = basis.chan_sub_data[α]

        for (sα, sd_α) in enumerate(sub_data_α)
            blocks_α = sd_α.proj_by_irrep[Gamma]
            n_r_α = sum(b.n_r for b in blocks_α; init=0)
            n_r_α == 0 && continue

            col_start = 1
            for β in 1:n_ch
                sub_data_β = basis.chan_sub_data[β]

                for (sβ, sd_β) in enumerate(sub_data_β)
                    blocks_β = sd_β.proj_by_irrep[Gamma]
                    n_r_β = sum(b.n_r for b in blocks_β; init=0)

                    if n_r_β == 0
                        col_start += 0  # no contribution, advance nothing
                        continue
                    end
                    if (α, sα) > (β, sβ)
                        col_start += n_r_β
                        continue
                    end
                    # n_r_β > 0: advance col_start after processing

                    V_hel = _get_V_hel_block(basis, α, sα, β, sβ)

                    if V_hel !== nothing && size(V_hel, 1) > 0 && size(V_hel, 2) > 0
                        # Finite-volume factor
                        N_α_val = length(first(sd_α.states)[1])
                        N_β_val = length(first(sd_β.states)[1])
                        d_val = 3 * (N_α_val + N_β_val) - 6
                        fv = (2π * ħc / L_phys)^(d_val / 2)

                        # Project V_hel: each (blk_α, blk_β) block pair
                        row_off = 1
                        for (i_blk_α, blk_α) in enumerate(blocks_α)
                            col_off = 1
                            for (i_blk_β, blk_β) in enumerate(blocks_β)
                                if (α, sα) == (β, sβ) &&
                                   i_blk_α > i_blk_β
                                    col_off += blk_β.n_r
                                    continue
                                end
                                V_sub = view(V_hel, blk_α.row_indices, blk_β.row_indices)
                                tmp = V_sub * blk_β.X
                                V_proj = fv * (blk_α.X' * tmp)
                                r_rng = row_start + row_off - 1 : row_start + row_off + blk_α.n_r - 2
                                c_rng = col_start + col_off - 1 : col_start + col_off + blk_β.n_r - 2
                                H[r_rng, c_rng] .+= V_proj
                                if r_rng != c_rng
                                    H[c_rng, r_rng] .+= adjoint(V_proj)
                                end
                                col_off += blk_β.n_r
                            end
                            row_off += blk_α.n_r
                        end
                    end

                    # Kinetic-energy diagonal term (channel- and subchannel-diagonal only).
                    if α == β && sα == sβ
                        T_off = 1
                        for blk_α in blocks_α
                            n_tuple = sd_α.states[(blk_α.row_indices[1]-1) ÷ sd_α.dim_kappa + 1][1]
                            T_rep = _kinetic_energy_rep(n_tuple, sd_α.per_mass,
                                                        L_phys, sd_α.kinetic_type; d=basis.sys.d)
                            r_rng = row_start + T_off - 1 : row_start + T_off + blk_α.n_r - 2
                            for i in r_rng
                                H[i, i] += T_rep
                            end
                            T_off += blk_α.n_r
                        end
                    end

                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end

    return H
end

# ============ Channel-decomposition utilities ============

"""
    _channel_row_ranges(basis::SystemBasis, Gamma::String) -> Vector{UnitRange{Int}}

Return the row-index range of each channel in the H matrix for irrep Γ.
Traversal order is identical to `_assemble_irrep_hamiltonian`.
"""
function _channel_row_ranges(basis::SystemBasis, Gamma::String)
    n_ch = length(basis.sys.channels)
    ranges = Vector{UnitRange{Int}}(undef, n_ch)
    dim = basis.irrep_total_dim[Gamma]

    if dim == 0
        for α in 1:n_ch
            ranges[α] = 1:0
        end
        return ranges
    end

    row_start = 1
    for α in 1:n_ch
        chan_start = row_start
        for sd_α in basis.chan_sub_data[α]
            blocks = sd_α.proj_by_irrep[Gamma]
            n_r = sum(b.n_r for b in blocks; init=0)
            row_start += n_r
        end
        ranges[α] = row_start > chan_start ? (chan_start:row_start-1) : (1:0)
    end
    return ranges
end

"""
    channel_decomposition(basis::SystemBasis, Gamma::String, evecs::Matrix{ComplexF64})
        -> Vector{Dict{String, Float64}}

For the eigenvector matrix of irrep Γ (one eigenvector per column), compute the |v|² fraction of each channel.
Return a vector of length size(evecs,2), whose entries are Dict(channel_name => fraction).
"""
function channel_decomposition(basis::SystemBasis, Gamma::String, evecs::Matrix{ComplexF64})
    ranges = _channel_row_ranges(basis, Gamma)
    chan_names = [ch.name for ch in basis.sys.channels]
    n_ev = size(evecs, 2)
    result = Vector{Dict{String, Float64}}(undef, n_ev)

    for j in 1:n_ev
        v = evecs[:, j]
        total = sum(abs2, v)
        total == 0.0 && (total = 1.0)
        fracs = Dict{String, Float64}()
        for (α, rng) in enumerate(ranges)
            isempty(rng) && continue
            fracs[chan_names[α]] = round(sum(abs2, v[rng]) / total, digits=4)
        end
        result[j] = fracs
    end
    return result
end

# ============ Sparse linear operator + eigs ============

"""
    _VBlock — Storage for one projected V block

mat is fv * X_α' * V_sub * X_β，with size n_r_α × n_r_β。
"""
struct _VBlock
    r_rng::UnitRange{Int}
    c_rng::UnitRange{Int}
    mat::Matrix{ComplexF64}
    add_adjoint::Bool
end

"""
    _build_hamiltonian_operator(basis, Gamma) -> (T_diag, v_blocks)

Construct a Hamiltonian linear-operator representation from SystemBasis.
Return kinetic diagonal vector T_diag and projected V-block list, replacing the dense H matrix (dim×dim).
Traversal logic is identical to _assemble_irrep_hamiltonian; only the output representation differs.
"""
function _build_hamiltonian_operator(basis::SystemBasis, Gamma::String;
                                     include_kinetic::Bool=true)
    dim = basis.irrep_total_dim[Gamma]
    dim == 0 && return Float64[], _VBlock[]

    T_diag = zeros(Float64, dim)
    v_blocks = _VBlock[]
    n_ch = length(basis.sys.channels)
    L_phys = basis.L_phys

    row_start = 1
    for α in 1:n_ch
        sub_data_α = basis.chan_sub_data[α]
        for (sα, sd_α) in enumerate(sub_data_α)
            blocks_α = sd_α.proj_by_irrep[Gamma]
            n_r_α = sum(b.n_r for b in blocks_α; init=0)
            n_r_α == 0 && continue

            col_start = 1
            for β in 1:n_ch
                sub_data_β = basis.chan_sub_data[β]
                for (sβ, sd_β) in enumerate(sub_data_β)
                    blocks_β = sd_β.proj_by_irrep[Gamma]
                    n_r_β = sum(b.n_r for b in blocks_β; init=0)
                    n_r_β == 0 && continue
                    if (α, sα) > (β, sβ)
                        col_start += n_r_β
                        continue
                    end

                    V_hel = _get_V_hel_block(basis, α, sα, β, sβ)

                    if V_hel !== nothing && size(V_hel, 1) > 0 && size(V_hel, 2) > 0
                        N_α_val = length(first(sd_α.states)[1])
                        N_β_val = length(first(sd_β.states)[1])
                        d_val = 3 * (N_α_val + N_β_val) - 6
                        fv = (2π * ħc / L_phys)^(d_val / 2)

                        row_off = 1
                        for (i_blk_α, blk_α) in enumerate(blocks_α)
                            col_off = 1
                            for (i_blk_β, blk_β) in enumerate(blocks_β)
                                if (α, sα) == (β, sβ) &&
                                   i_blk_α > i_blk_β
                                    col_off += blk_β.n_r
                                    continue
                                end
                                V_sub = view(V_hel, blk_α.row_indices, blk_β.row_indices)
                                tmp = V_sub * blk_β.X
                                V_proj = fv * (blk_α.X' * tmp)
                                if !all(iszero, V_proj)
                                    r_rng = row_start+row_off-1 : row_start+row_off+blk_α.n_r-2
                                    c_rng = col_start+col_off-1 : col_start+col_off+blk_β.n_r-2
                                    push!(v_blocks, _VBlock(
                                        r_rng, c_rng, V_proj, r_rng != c_rng))
                                end
                                col_off += blk_β.n_r
                            end
                            row_off += blk_α.n_r
                        end
                    end

                    if include_kinetic && α == β && sα == sβ
                        T_off = 1
                        for blk_α in blocks_α
                            n_tuple = sd_α.states[(blk_α.row_indices[1]-1) ÷ sd_α.dim_kappa + 1][1]
                            T_rep = _kinetic_energy_rep(n_tuple, sd_α.per_mass,
                                                        L_phys, sd_α.kinetic_type; d=basis.sys.d)
                            r_rng = row_start+T_off-1 : row_start+T_off+blk_α.n_r-2
                            T_diag[r_rng] .+= T_rep
                            T_off += blk_α.n_r
                        end
                    end

                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end
    return T_diag, v_blocks
end


"""
    _H_matvec(x, T_diag, v_blocks) -> y

Linear operator y = H * x, without storing dense H.
Traverse all V blocks: y[r_rng] += V_proj * x[c_rng]，then add the kinetic diagonal contribution.
"""
function _H_matvec(x::AbstractVector{<:ComplexF64}, T_diag::Vector{Float64},
                   v_blocks::Vector{_VBlock})
    y = T_diag .* x
    for vb in v_blocks
        mul!(view(y, vb.r_rng), vb.mat, view(x, vb.c_rng), true, true)
        if vb.add_adjoint
            mul!(view(y, vb.c_rng), adjoint(vb.mat),
                 view(x, vb.r_rng), true, true)
        end
    end
    return y
end

# ============ Truly factorized Q† V_hel Q operator ============

struct _ProjectionApplyBlock
    X::Matrix{ComplexF64}
    row_indices::Vector{Int}
    projected_range::UnitRange{Int}
end

struct _ProjectionPlan
    blocks::Vector{_ProjectionApplyBlock}
    unified_dim::Int
end

struct _FactorizedInteractionBlock
    left_plan::Int
    right_plan::Int
    mat::AbstractMatrix{ComplexF64}
    fv::Float64
    add_adjoint::Bool
end

mutable struct _FactorizedHamiltonianOperator
    T_diag::Vector{Float64}
    plans::Vector{_ProjectionPlan}
    interactions::Vector{_FactorizedInteractionBlock}
    u::Vector{Vector{ComplexF64}}
    z::Vector{Vector{ComplexF64}}
end

function _apply_Q!(u::Vector{ComplexF64}, plan::_ProjectionPlan,
                   x::AbstractVector{<:ComplexF64})
    fill!(u, 0)
    for block in plan.blocks
        X = block.X
        rows = block.row_indices
        cols = block.projected_range
        @inbounds for j in axes(X, 2)
            xj = x[cols[j]]
            iszero(xj) && continue
            for i in axes(X, 1)
                u[rows[i]] += X[i, j] * xj
            end
        end
    end
    return u
end

function _apply_Qadj_add!(y::Vector{ComplexF64}, plan::_ProjectionPlan,
                          z::Vector{ComplexF64})
    for block in plan.blocks
        X = block.X
        rows = block.row_indices
        cols = block.projected_range
        @inbounds for j in axes(X, 2)
            acc = zero(ComplexF64)
            for i in axes(X, 1)
                acc += conj(X[i, j]) * z[rows[i]]
            end
            y[cols[j]] += acc
        end
    end
    return y
end

function LinearAlgebra.mul!(y::Vector{ComplexF64},
                            H::_FactorizedHamiltonianOperator,
                            x::AbstractVector{<:ComplexF64})
    length(y) == length(H.T_diag) == length(x) || throw(DimensionMismatch())
    @. y = H.T_diag * x

    for (u, plan) in zip(H.u, H.plans)
        _apply_Q!(u, plan, x)
    end
    for z in H.z
        fill!(z, 0)
    end

    for block in H.interactions
        mul!(H.z[block.left_plan], block.mat, H.u[block.right_plan],
             block.fv, true)
        if block.add_adjoint
            mul!(H.z[block.right_plan], adjoint(block.mat),
                 H.u[block.left_plan], block.fv, true)
        end
    end

    for (z, plan) in zip(H.z, H.plans)
        _apply_Qadj_add!(y, plan, z)
    end
    return y
end

function (H::_FactorizedHamiltonianOperator)(x::AbstractVector{<:ComplexF64})
    y = similar(x, ComplexF64, length(H.T_diag))
    return mul!(y, H, x)
end

function _build_factorized_hamiltonian_operator(basis::SystemBasis, Gamma::String)
    dim = basis.irrep_total_dim[Gamma]
    dim == 0 && return _FactorizedHamiltonianOperator(
        Float64[], _ProjectionPlan[], _FactorizedInteractionBlock[],
        Vector{ComplexF64}[], Vector{ComplexF64}[])

    plans = _ProjectionPlan[]
    plan_index = Dict{Tuple{Int,Int},Int}()
    T_diag = zeros(Float64, dim)
    projected_start = 1

    for α in eachindex(basis.chan_sub_data)
        for (sα, sd) in enumerate(basis.chan_sub_data[α])
            blocks = sd.proj_by_irrep[Gamma]
            n_r = sum(b.n_r for b in blocks; init=0)
            n_r == 0 && continue

            apply_blocks = _ProjectionApplyBlock[]
            local_start = projected_start
            for block in blocks
                rng = local_start:local_start + block.n_r - 1
                push!(apply_blocks, _ProjectionApplyBlock(
                    block.X, block.row_indices, rng))

                n_tuple = sd.states[
                    (block.row_indices[1] - 1) ÷ sd.dim_kappa + 1][1]
                T_rep = _kinetic_energy_rep(
                    n_tuple, sd.per_mass, basis.L_phys, sd.kinetic_type;
                    d=basis.sys.d)
                T_diag[rng] .= T_rep
                local_start += block.n_r
            end

            unified_dim = length(sd.states) * sd.dim_kappa
            push!(plans, _ProjectionPlan(apply_blocks, unified_dim))
            plan_index[(α, sα)] = length(plans)
            projected_start += n_r
        end
    end
    projected_start == dim + 1 || error("factorized projected layout mismatch")

    interactions = _FactorizedInteractionBlock[]
    for (key, V_hel) in basis.V_hel_blocks
        α, sα, β, sβ = key
        left = get(plan_index, (α, sα), 0)
        right = get(plan_index, (β, sβ), 0)
        (left == 0 || right == 0 || isempty(V_hel)) && continue

        sd_α = basis.chan_sub_data[α][sα]
        sd_β = basis.chan_sub_data[β][sβ]
        N_α = length(first(sd_α.states)[1])
        N_β = length(first(sd_β.states)[1])
        exponent = (3 * (N_α + N_β) - 6) / 2
        fv = (2π * ħc / basis.L_phys)^exponent
        push!(interactions, _FactorizedInteractionBlock(
            left, right, V_hel, fv, left != right))
    end

    u = [zeros(ComplexF64, p.unified_dim) for p in plans]
    z = [zeros(ComplexF64, p.unified_dim) for p in plans]
    return _FactorizedHamiltonianOperator(T_diag, plans, interactions, u, z)
end

"""
    compute_spectrum_eigs(basis::SystemBasis; n_levels=20, tol=1e-8)
        -> Dict{String, Vector{Float64}}

Use a sparse linear operator plus KrylovKit.eigsolve to compute the lowest n_levels eigenvalues of each irrep.

Do not construct a dense dim×dim Hamiltonian matrix; suitable for large systems (>5000).
"""
function compute_spectrum_eigs(basis::SystemBasis;
                                n_levels::Union{Int, Dict{String, Int}} = 20,
                                tol::Float64 = 1e-8,
                                return_vectors::Bool = false)
    result = Dict{String, Vector{Float64}}()
    vecs   = Dict{String, Matrix{ComplexF64}}()
    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue

        n_g = n_levels isa Int ? n_levels : get(n_levels, Gamma, 20)
        T_diag, v_blocks = _build_hamiltonian_operator(basis, Gamma)

        if isempty(v_blocks)
            ev = sort(T_diag)[1:min(n_g, dim)]
            result[Gamma] = ev
            if return_vectors
                vecs[Gamma] = Matrix(Diagonal(ones(ComplexF64, length(ev))))
            end
            continue
        end

        n_ev = min(n_g, dim)
        H_op = x -> _H_matvec(x, T_diag, v_blocks)

        evals_raw, evecs_raw, info = eigsolve(
            H_op, _deterministic_krylov_start(dim), n_ev, :SR,
            Lanczos(; tol=tol, krylovdim=max(20, 3n_ev + 8), maxiter=200)
        )
        info.converged < n_ev &&
            @warn "KrylovKit converged for only $(info.converged)/$n_ev eigenvalues (Γ=$Gamma)"

        # Sort by real part and reorder eigenvectors consistently.
        perm = sortperm(real.(evals_raw))
        n_out = min(length(evals_raw), n_ev)
        ev = real.(evals_raw[perm][1:n_out])

        if basis.sys.d != D000
            P_mag = (2π * ħc / basis.L_phys) * sqrt(Float64(sum(abs2, basis.sys.d)))
            ev = [sqrt(E^2 + P_mag^2) for E in ev]
        end
        result[Gamma] = ev

        if return_vectors
            V = reduce(hcat, evecs_raw[perm][1:n_out])
            vecs[Gamma] = V
        end
    end
    return return_vectors ? (result, vecs) : result
end


"""
    _compute_spectrum_preprojected(basis, interactions; n_levels, backend,
                                   return_vectors=false)

Solve a Hamiltonian whose interaction has already been projected into the
irrep basis. This is the evaluation path used by `prepare_affine`: only the
kinetic diagonal is rebuilt for the current masses.
"""
function _compute_spectrum_preprojected(
        basis::SystemBasis,
        interactions::Dict{String, Vector{_VBlock}};
        n_levels::Union{Nothing, Int, Dict{String, Int}}=20,
        backend::Symbol=:projected_blocks,
        return_vectors::Bool=false)
    result = Dict{String, Vector{Float64}}()
    vecs = Dict{String, Matrix{ComplexF64}}()

    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue
        n_g = n_levels === nothing ? dim :
              n_levels isa Int ? n_levels : get(n_levels, Gamma, 20)
        n_ev = min(n_g, dim)

        T_diag, unused_blocks = _build_hamiltonian_operator(basis, Gamma)
        isempty(unused_blocks) || error(
            "affine preprojected evaluation requires empty V_hel_blocks")
        v_blocks = get(interactions, Gamma, _VBlock[])

        if backend == :complete_matrix
            H = Matrix{ComplexF64}(Diagonal(complex.(T_diag)))
            for vb in v_blocks
                H[vb.r_rng, vb.c_rng] .+= vb.mat
                vb.add_adjoint && (H[vb.c_rng, vb.r_rng] .+= adjoint(vb.mat))
            end
            Hh = Hermitian(H)
            if return_vectors
                evals_full, evecs_full = n_ev >= dim ? eigen(Hh) : eigen(Hh, 1:n_ev)
                perm = sortperm(real.(evals_full))
                result[Gamma] = real.(evals_full[perm])
                vecs[Gamma] = evecs_full[:, perm]
            else
                result[Gamma] = n_ev >= dim ?
                    sort(real.(eigvals(Hh))) : sort(real.(eigvals(Hh, 1:n_ev)))
            end
            if basis.sys.d != D000
                P_mag = (2π * ħc / basis.L_phys) *
                        sqrt(Float64(sum(abs2, basis.sys.d)))
                result[Gamma] = sqrt.(result[Gamma].^2 .+ P_mag^2)
            end
            continue
        end

        if isempty(v_blocks)
            perm = sortperm(T_diag)[1:n_ev]
            result[Gamma] = T_diag[perm]
            if return_vectors
                V = zeros(ComplexF64, dim, n_ev)
                for (j, i) in enumerate(perm)
                    V[i, j] = 1
                end
                vecs[Gamma] = V
            end
            if basis.sys.d != D000
                P_mag = (2π * ħc / basis.L_phys) *
                        sqrt(Float64(sum(abs2, basis.sys.d)))
                result[Gamma] = sqrt.(result[Gamma].^2 .+ P_mag^2)
            end
            continue
        end

        H_op = x -> _H_matvec(x, T_diag, v_blocks)
        evals_raw, evecs_raw, info = eigsolve(
            H_op, _deterministic_krylov_start(dim), n_ev, :SR,
            Lanczos(; tol=1e-8, krylovdim=max(20, 3n_ev + 8), maxiter=200))
        info.converged < n_ev && @warn(
            "KrylovKit converged for only $(info.converged)/$n_ev eigenvalues (Γ=$Gamma)")
        perm = sortperm(real.(evals_raw))
        n_out = min(length(evals_raw), n_ev)
        result[Gamma] = real.(evals_raw[perm][1:n_out])
        if basis.sys.d != D000
            P_mag = (2π * ħc / basis.L_phys) *
                    sqrt(Float64(sum(abs2, basis.sys.d)))
            result[Gamma] = sqrt.(result[Gamma].^2 .+ P_mag^2)
        end
        if return_vectors
            vecs[Gamma] = reduce(hcat, evecs_raw[perm][1:n_out])
        end
    end
    return return_vectors ? (result, vecs) : result
end

function _deterministic_krylov_start(dim::Int)
    x = ComplexF64[sin(i) + im * cos(sqrt(2.0) * i) for i in 1:dim]
    return x ./ norm(x)
end

function compute_spectrum_factorized(
        basis::SystemBasis;
        n_levels::Union{Int, Dict{String, Int}}=7,
        tol::Float64=1e-8,
        return_vectors::Bool=false)
    result = Dict{String, Vector{Float64}}()
    vecs = Dict{String, Matrix{ComplexF64}}()

    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue
        n_g = n_levels isa Int ? n_levels : get(n_levels, Gamma, 7)
        n_ev = min(n_g, dim)
        H_op = _build_factorized_hamiltonian_operator(basis, Gamma)

        if isempty(H_op.interactions)
            perm = sortperm(H_op.T_diag)[1:n_ev]
            result[Gamma] = H_op.T_diag[perm]
            if return_vectors
                V = zeros(ComplexF64, dim, n_ev)
                for (j, i) in enumerate(perm)
                    V[i, j] = 1
                end
                vecs[Gamma] = V
            end
            continue
        end

        initial = _deterministic_krylov_start(dim)
        evals_raw, evecs_raw, info = eigsolve(
            H_op, initial, n_ev, :SR,
            Lanczos(; tol=tol, krylovdim=max(20, 3n_ev + 8), maxiter=200))
        info.converged < n_ev && @warn(
            "KrylovKit converged for only $(info.converged)/$n_ev eigenvalues (Γ=$Gamma)")

        perm = sortperm(real.(evals_raw))
        n_out = min(length(evals_raw), n_ev)
        ev = real.(evals_raw[perm][1:n_out])
        if basis.sys.d != D000
            P_mag = (2π * ħc / basis.L_phys) *
                    sqrt(Float64(sum(abs2, basis.sys.d)))
            ev = [sqrt(E^2 + P_mag^2) for E in ev]
        end
        result[Gamma] = ev
        if return_vectors
            vecs[Gamma] = reduce(hcat, evecs_raw[perm][1:n_out])
        end
    end
    return return_vectors ? (result, vecs) : result
end

"""
    compute_spectrum(basis::SystemBasis; n_levels = nothing) -> Dict{String, Vector{Float64}}

Compute projected-Hamiltonian eigenvalues (ascending) for each irrep from a SystemBasis with filled V_hel.

If `n_levels::Dict{String,Int}` is provided, compute only the lowest n_levels[Γ] eigenvalues of each irrep.
"""
function compute_spectrum(basis::SystemBasis;
                          n_levels::Union{Nothing, Dict{String, Int}} = nothing,
                          return_vectors::Bool = false)
    result = Dict{String, Vector{Float64}}()
    vecs   = Dict{String, Matrix{ComplexF64}}()
    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue

        H = _assemble_irrep_hamiltonian(basis, Gamma)
        n = if n_levels === nothing
            dim
        else
            min(get(n_levels, Gamma, dim), dim)
        end
        Hh = Hermitian(H)
        if return_vectors
            evals_full, evecs_full = n >= dim ? eigen(Hh) : eigen(Hh, 1:n)
            perm = sortperm(real.(evals_full))
            ev = real.(evals_full[perm])
            # Moving-frame boost
            if basis.sys.d != D000
                P_mag = (2π * ħc / basis.L_phys) * sqrt(Float64(sum(abs2, basis.sys.d)))
                ev = [sqrt(E^2 + P_mag^2) for E in ev]
            end
            result[Gamma] = ev
            vecs[Gamma] = evecs_full[:, perm]
        else
            ev = if n >= dim
                sort(real.(eigvals(Hh)))
            else
                sort(real.(eigvals(Hh, 1:n)))
            end
            if basis.sys.d != D000
                P_mag = (2π * ħc / basis.L_phys) * sqrt(Float64(sum(abs2, basis.sys.d)))
                ev = [sqrt(E^2 + P_mag^2) for E in ev]
            end
            result[Gamma] = ev
        end
    end
    return return_vectors ? (result, vecs) : result
end

# ============ Single-projection cache ============

function _get_projection(n_tuple::NTuple{N, Momentum},
                         lambda_tuple::NTuple{N, Float64},
                         kappa::String, Gamma::String,
                         d::Momentum, spin::Float64,
                         etas::Vector{Float64}) where N
    key = (n_tuple=n_tuple, lambda_tuple=lambda_tuple,
           kappa=kappa, Gamma=Gamma, d=d, spin=spin, etas=Tuple(etas))
    return get!(_PROJ_CACHE, key) do
        st = isinteger(spin) ? :boson : :fermion
        M = count(iszero, n_tuple)
        prepared_orbit = M > 0 && spin != 0.0 ? nothing :
            _get_prepared_projection_orbit(n_tuple, lambda_tuple, d, st)
        res = subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                                  d_total=d, species_type=st, spin=spin, etas=etas,
                                  prepared_orbit=prepared_orbit)
        (X=res.X, states=res.subspace_states, Z=res.Z)
    end
end

function _get_projection(n_tuple::NTuple{N, Momentum},
                         lambda_tuple::NTuple{N, Float64},
                         kappa::Tuple, Gamma::String,
                         d::Momentum,
                         species::Vector{Int}, particle_types::Vector{Symbol},
                         spins::Vector{Float64}, etas::Vector{Float64}) where N
    key = (n_tuple=n_tuple, lambda_tuple=lambda_tuple,
           kappa=kappa, Gamma=Gamma, d=d,
           species=Tuple(species), pt=Tuple(particle_types),
           spins=Tuple(spins), etas=Tuple(etas))
    return get!(_PROJ_CACHE, key) do
        res = subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                                  d_total=d, species=species,
                                  particle_types=particle_types,
                                  spins=spins, etas=etas)
        (X=res.X, states=res.subspace_states, Z=res.Z)
    end
end

# ============ Utility functions ============

function _expand_per_particle(ch::FockChannel)
    per_spin = Rational{Int}[]
    per_etas = Float64[]
    for (s, j, eta) in zip(ch.species, ch.spins, ch.etas)
        append!(per_spin, fill(j, s))
        append!(per_etas, fill(eta, s))
    end
    return per_spin, per_etas
end

# ============ Representative-state projection lists for channels ============

function _get_channel_proj_list(ch::FockChannel, Ncut::Int, d::Momentum,
                                kappa::String, Gamma::String,
                                spin::Float64, etas::Vector{Float64})
    key = (species=Tuple(ch.species), pt=Tuple(ch.particle_types),
           Ncut=Ncut, d=d, kappa=kappa, Gamma=Gamma,
           spin=spin, etas=Tuple(etas))
    return get!(_PROJ_LIST_CACHE, key) do
        result = []
        reps = get_momentum_reps(ch.species, ch.particle_types, Ncut, d)
        for rep in reps
            M = count(n -> n == Momentum(0, 0, 0), rep)
            if M > 0 && spin != 0.0
                # Zero momentum + nonzero spin: use helicity representatives for the finite-momentum part.
                N_total = length(rep)
                N_fin = N_total - M
                if N_fin > 0
                    fin_rep = ntuple(i -> rep[M + i], N_fin)
                    fin_hels = get_helicity_reps(fin_rep, [N_fin],
                        [ch.particle_types[1]], [ch.spins[1]], d)
                else
                    fin_hels = [()]  # All particles have zero momentum.
                end
                for fin_lam in fin_hels
                    fin_lam_f = Tuple(Float64.(fin_lam))
                    full_lam = (ntuple(_ -> 0.0, M)..., fin_lam_f...)
                    proj = _get_projection(rep, full_lam, kappa, Gamma, d, spin, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=full_lam, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            else
                hels = get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, d)
                for lam in hels
                    lam_f = Tuple(Float64.(lam))
                    proj = _get_projection(rep, lam_f, kappa, Gamma, d, spin, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=lam_f, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            end
        end
        result
    end
end

function _get_channel_proj_list(ch::FockChannel, Ncut::Int, d::Momentum,
                                kappa::Tuple, Gamma::String,
                                spins::Vector{Float64}, etas::Vector{Float64})
    key = (species=Tuple(ch.species), pt=Tuple(ch.particle_types),
           Ncut=Ncut, d=d, kappa=kappa, Gamma=Gamma,
           spins=Tuple(spins), etas=Tuple(etas))
    return get!(_PROJ_LIST_CACHE, key) do
        result = []
        reps = get_momentum_reps(ch.species, ch.particle_types, Ncut, d)
        for rep in reps
            hels = get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, d)
            if !isempty(hels)
                for lam in hels
                    lam_f = Tuple(Float64.(lam))
                    proj = _get_projection(rep, lam_f, kappa, Gamma, d,
                                           ch.species, ch.particle_types, spins, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=lam_f, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            else
                N_total = length(rep)
                M_total = count(n -> iszero(n), rep)
                N_fin = N_total - M_total

                fin_momenta = Momentum[]
                fin_species_vec = Int[]
                fin_types_vec = Symbol[]
                fin_spins_vec = Rational{Int}[]
                off = 0
                for (k, Nk) in enumerate(ch.species)
                    fm_in_sp = 0
                    for i in 1:Nk
                        !iszero(rep[off + i]) && (fm_in_sp += 1)
                    end
                    if fm_in_sp > 0
                        push!(fin_species_vec, fm_in_sp)
                        push!(fin_types_vec, ch.particle_types[k])
                        push!(fin_spins_vec, Rational{Int}(Int(2*ch.spins[k]), 2))
                        for i in 1:Nk
                            !iszero(rep[off + i]) && push!(fin_momenta, rep[off + i])
                        end
                    end
                    off += Nk
                end

                if N_fin > 0
                    fin_rep = Tuple(fin_momenta)
                    fin_hels = get_helicity_reps(fin_rep, fin_species_vec,
                        fin_types_vec, fin_spins_vec, d)
                else
                    fin_hels = [()]
                end

                for fin_lam in fin_hels
                    fin_lam_f = Float64[Float64(x) for x in fin_lam]
                    full_lam_vec = Float64[]
                    fm_idx = 1
                    off2 = 0
                    for (k, Nk) in enumerate(ch.species)
                        for i in 1:Nk
                            if iszero(rep[off2 + i])
                                push!(full_lam_vec, 0.0)
                            else
                                push!(full_lam_vec, fin_lam_f[fm_idx])
                                fm_idx += 1
                            end
                        end
                        off2 += Nk
                    end
                    full_lam = Tuple(full_lam_vec)

                    proj = _get_projection(rep, full_lam, kappa, Gamma, d,
                                           ch.species, ch.particle_types, spins, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=full_lam, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            end
        end
        result
    end
end

# ============ Moving-frame utilities ============

"""
    boost_to_cm(p_mov, masses, d, L_phys) -> (p_cm, factor)

Boost moving-frame physical momenta to the center-of-mass frame and compute the kinematic factor.

# Arguments
- `p_mov`: moving-frame physical momenta (iterable 3-vectors in MeV)
- `masses`: per-particle masses (MeV), with the same length as p_mov
- `d`: integer total-momentum vector d = (L/2πħc)·P
- `L_phys`: physical finite-volume size (fm)

# Returns
- `p_cm`: center-of-mass momentum (same type as p_mov)
- `factor`: kinematic factor = [ΣE'/ΣE^cm · ∏(E^cm/E')]^{1/2}

When d == (0,0,0),，p_cm = p_mov, factor = 1.0。
"""
function boost_to_cm(p_mov, masses::Vector{Float64}, d::Momentum, L_phys::Float64)
    N = length(p_mov)
    P_tot = (2π * ħc / L_phys) .* SVector{3,Float64}(d)
    P2 = sum(abs2, P_tot)

    # Rest frame: identity transformation.
    if P2 == 0.0
        return collect(p_mov), 1.0
    end

    # Moving-frame on-shell energies (boosts always use relativistic dispersion).
    E_prime = [sqrt(m^2 + sum(abs2, p)) for (p, m) in zip(p_mov, masses)]
    E_tot = sum(E_prime)

    # Lorentz boost
    gamma = E_tot / sqrt(E_tot^2 - P2)

    p_cm = similar(p_mov, eltype(p_mov))
    T = eltype(p_mov)
    for i in 1:N
        p = p_mov[i]
        p_dot_P = p[1]*P_tot[1] + p[2]*P_tot[2] + p[3]*P_tot[3]
        coeff = (gamma - 1.0) * p_dot_P / P2 - gamma * E_prime[i] / E_tot
        p_cm[i] = T(p[1] + coeff * P_tot[1],
                     p[2] + coeff * P_tot[2],
                     p[3] + coeff * P_tot[3])
    end

    # Center-of-mass on-shell energies.
    E_cm = [sqrt(m^2 + sum(abs2, p)) for (p, m) in zip(p_cm, masses)]
    E_tot_cm = sum(E_cm)

    # kinematic factor
    prod_ratio = prod(E_cm[i] / E_prime[i] for i in 1:N)
    factor = sqrt(E_tot / E_tot_cm * prod_ratio)

    return p_cm, factor
end

# ============ Kinetic-energy helpers ============

_expand_per_particle_mass(ch::FockChannel, params=nothing) =
    resolve_particle_masses(ch, params)

function _kinetic_energy_rep(n_tuple, per_mass::Vector{Float64}, L_phys::Float64, kt::KineticType;
                             d::Momentum = D000)
    length(per_mass) == length(n_tuple) || throw(DimensionMismatch(
        "kinetic-energy mass count $(length(per_mass)) does not match " *
        "particle count $(length(n_tuple)); resolve dynamic masses before solving"))
    pref = (2π * ħc / L_phys)^2
    T = 0.0
    # Moving frame: first transform to center-of-mass momenta.
    if d != D000
        pv = 2π * ħc / L_phys
        p_mov = [pv .* Float64.(n) for n in n_tuple]
        p_cm, _ = boost_to_cm(p_mov, per_mass, d, L_phys)
        for (p, m) in zip(p_cm, per_mass)
            p2 = Float64(sum(abs2, p))
            if kt == relativistic
                T += sqrt(m^2 + p2)
            else
                T += m + p2 / (2m)
            end
        end
    else
        for (n, m) in zip(n_tuple, per_mass)
            n2 = Float64(sum(abs2, n))
            if kt == relativistic
                T += sqrt(m^2 + pref * n2)
            else
                T += m + pref * n2 / (2m)
            end
        end
    end
    return T
end

function _build_kinetic_diag(projs::Vector, per_mass::Vector{Float64}, L_phys::Float64, kt::KineticType;
                             d::Momentum = D000)
    n_r = sum(p.n_r for p in projs; init=0)
    T = zeros(ComplexF64, n_r, n_r)
    n_r == 0 && return T
    row = 1
    for p in projs
        T_rep = _kinetic_energy_rep(p.rep, per_mass, L_phys, kt; d=d)
        for i in 1:p.n_r
            T[row, row] = T_rep
            row += 1
        end
    end
    return T
end

# ============ V-function argument-reordering adapter ============
# build_V_hel call: V_inner(n_α, σ_α, n_β, σ_β, extra_args...)
# User V_func:      V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
#
# For non-one-dimensional κ, V_func returns a complete dimA×dimB matrix at once;
# when one side has dimension 1, a vector may represent the row/column block.

function _V_adapter(V_func, kapA, kapB, rA, rB, dimA, dimB,
                    ch_α, ch_β, L_phys, params)
    return (n_α, σ_α, n_β, σ_β, extra...) -> begin
        value = V_func(
            n_α, n_β, σ_α, σ_β, kapA, kapB, rA, rB,
            ch_α, ch_β, L_phys, params)
        if value isa Number
            if dimA == 1 && dimB == 1
                return value
            end
            iszero(value) && return zeros(ComplexF64, dimA, dimB)
            throw(DimensionMismatch(
                "V_func returned a nonzero scalar for κ=($kapA, $kapB); " *
                "expected a ($dimA, $dimB) matrix (scalar zero is allowed)"))
        end
        if value isa AbstractMatrix
            size(value) == (dimA, dimB) || throw(DimensionMismatch(
                "V_func returned matrix of size $(size(value)); " *
                "expected ($dimA, $dimB) for κ=($kapA, $kapB)"))
            return value
        end
        if value isa AbstractVector
            (dimA == 1 || dimB == 1) || throw(DimensionMismatch(
                "V_func returned a vector for expected matrix size " *
                "($dimA, $dimB); vectors are accepted only for row/column blocks"))
            length(value) == dimA * dimB || throw(DimensionMismatch(
                "V_func returned vector of length $(length(value)); " *
                "expected $(dimA * dimB) for κ=($kapA, $kapB)"))
            return reshape(value, dimA, dimB)
        end
        throw(ArgumentError(
            "V_func must return a Number, AbstractMatrix, or row/column " *
            "AbstractVector, got $(typeof(value))"))
    end
end

# ============ Level 2: subchannel-pair blocks ============

function _build_subchannel_block(projs_α::Vector, projs_β::Vector,
                                 per_spin_α, per_spin_β,
                                 L_phys::Float64, V_adapter::Function;
                                 V_basis::Symbol=:canonical)
    n_r_α = sum(p.n_r for p in projs_α; init=0)
    n_r_β = sum(p.n_r for p in projs_β; init=0)
    (n_r_α == 0 || n_r_β == 0) && return zeros(ComplexF64, n_r_α, n_r_β)

    block = zeros(ComplexF64, n_r_α, n_r_β)
    row_start = 1
    for p_α in projs_α
        col_start = 1
        for p_β in projs_β
            sub = if V_basis == :helicity
                project_V_hel(p_α.X, p_β.X, p_α.states, p_β.states,
                             L_phys, V_adapter)
            else
                project_V(p_α.X, p_β.X, p_α.states, p_β.states,
                         per_spin_α, per_spin_β, L_phys, V_adapter)
            end
            block[row_start:row_start+p_α.n_r-1,
                  col_start:col_start+p_β.n_r-1] .= sub
            col_start += p_β.n_r
        end
        row_start += p_α.n_r
    end
    return block
end

# ============ Level 3: main orchestration function ============

"""
    build_hamiltonian_block(sys::FockSystem, Gamma::String, V_func, params)
        -> Matrix{ComplexF64}

Construct the complete projected Hamiltonian matrix for specified irrep Γ.

`V_func` signature matches potential_defs.jl:
    V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, params)
"""
function build_hamiltonian_block(sys::FockSystem, Gamma::String,
                                 V_func::Function, params;
                                 V_basis::Symbol=:canonical)
    Gamma in sys.selected_irreps ||
        throw(ArgumentError("Γ=$Gamma is not in sys.selected_irreps ($(sys.selected_irreps))"))

    # Moving frames currently support only N ≤ 2
    if sys.d != D000
        for ch in sys.channels
            ch.N > 2 && throw(ArgumentError(
                "moving frames (d≠0) currently support only channels with N≤2; channel \"$(ch.name)\" has N=$(ch.N)"))
        end
    end

    I = sys.I
    n_ch = length(sys.channels)
    d = sys.d

    # ===== Preprocessing: projection lists for every channel and subchannel. =====
    chan_sub_data = []
    for α in 1:n_ch
        ch = sys.channels[α]
        ncut = get_Ncut(sys, α)
        per_spin, per_etas_arr = _expand_per_particle(ch)
        per_spin_v = Rational{Int}.(per_spin)
        etas_v = Float64.(per_etas_arr)
        multi = length(ch.species) > 1
        per_mass = _expand_per_particle_mass(ch, params)

        subs = get_isospin_subchannels(ch, I)
        sub_entries = []
        for sub in subs
            projs = if multi
                _get_channel_proj_list(ch, ncut, d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
            else
                spin_val = Float64(only(unique(per_spin_v)))
                _get_channel_proj_list(ch, ncut, d, sub.κ, Gamma, spin_val, etas_v)
            end
            push!(sub_entries, (sub=sub, projs=projs, per_spin=per_spin_v))
        end
        push!(chan_sub_data, (ch=ch, subs=sub_entries, per_mass=per_mass))
    end

    # ===== Compute total dimension. =====
    total_dim = 0
    for α in 1:n_ch
        for se in chan_sub_data[α].subs
            total_dim += sum(p.n_r for p in se.projs; init=0)
        end
    end
    total_dim == 0 && return zeros(ComplexF64, 0, 0)

    H_full = zeros(ComplexF64, total_dim, total_dim)

    # ===== Fill all blocks. =====
    row_start = 1
    for α in 1:n_ch
        ch_α = chan_sub_data[α].ch
        for se_α in chan_sub_data[α].subs
            s_α = se_α.sub
            projs_α = se_α.projs
            pspin_α = se_α.per_spin
            n_r_α = sum(p.n_r for p in projs_α; init=0)

            col_start = 1
            for β in 1:n_ch
                ch_β = chan_sub_data[β].ch
                for se_β in chan_sub_data[β].subs
                    s_β = se_β.sub
                    projs_β = se_β.projs
                    pspin_β = se_β.per_spin
                    n_r_β = sum(p.n_r for p in projs_β; init=0)

                    L_phys = Float64(sys.L) * sys.a

                    V_adapted = _V_adapter(V_func, s_α.κ, s_β.κ,
                                           s_α.r, s_β.r,
                                           s_α.dim, s_β.dim, α, β, L_phys, params)
                    sub_block = _build_subchannel_block(
                        projs_α, projs_β, pspin_α, pspin_β, L_phys, V_adapted;
                        V_basis=V_basis)

                    # Kinetic-energy diagonal matrix (channel- and subchannel-diagonal only).
                    if α == β && s_α.κ == s_β.κ && s_α.r == s_β.r
                        T_diag = _build_kinetic_diag(projs_α, chan_sub_data[α].per_mass,
                                                     L_phys, ch_α.kinetic_type; d=d)
                        sub_block += T_diag
                    end

                    if n_r_α > 0 && n_r_β > 0
                        H_full[row_start:row_start+n_r_α-1,
                               col_start:col_start+n_r_β-1] .= sub_block
                    end
                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end

    return H_full
end

# ============ Eigenvalue calculation ============

"""
    compute_spectrum(sys::FockSystem, V_func, params) -> Dict{String, Vector{Float64}}

Given a FockSystem and interaction function, return projected-Hamiltonian eigenvalues of each irrep in ascending order.
Keys are irrep names and values are eigenvalue vectors (MeV).

# Example
```julia
evals = compute_spectrum(sys, my_V_11, MyParams(C0=2.0))
evals["T1-"]  # T1- energy-level vector
```
"""
function compute_spectrum(sys::FockSystem, V_func::Function, params;
                          n_levels::Union{Nothing, Dict{String, Int}} = nothing,
                          V_basis::Symbol=:canonical,
                          backend::Symbol=:complete_matrix,
                          eigs::Union{Nothing,Bool}=nothing,
                          channel_filter=nothing,
                          entry_filter=nothing,
                          validate_hermitian::Bool=false,
                          exclude_subchannels::AbstractVector{SubchannelExclusion}=SubchannelExclusion[])
    if eigs !== nothing
        backend == :complete_matrix || throw(ArgumentError(
            "cannot specify both backend=$backend and legacy eigs=$eigs"))
        backend = eigs ? :projected_blocks : :complete_matrix
    end
    backend in (:complete_matrix, :projected_blocks, :factorized) ||
        throw(ArgumentError(
            "backend must be :complete_matrix, :projected_blocks, or :factorized"))
    basis = SystemBasis(sys; exclude_subchannels=exclude_subchannels)
    build_V_hel_blocks!(basis, V_func, params; V_basis=V_basis,
                        channel_filter=channel_filter,
                        entry_filter=entry_filter,
                        validate_hermitian=validate_hermitian)
    if backend == :projected_blocks
        n_use = something(n_levels, 20)
        return compute_spectrum_eigs(basis; n_levels=n_use)
    elseif backend == :factorized
        n_use = something(n_levels, 7)
        return compute_spectrum_factorized(basis; n_levels=n_use)
    else
        return compute_spectrum(basis; n_levels=n_levels)
    end
end

"""
    compute_kinetic_spectrum(sys::FockSystem, params=nothing) -> Dict{String, Vector{Float64}}

Kinetic-only (no interaction) eigenvalues. Systems containing `mass_unfixed` must provide
`params`；Fixed-mass systems retain the original calling convention.
"""
function compute_kinetic_spectrum(sys::FockSystem, params=nothing)
    zero_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p) =
        zero(ComplexF64)
    return compute_spectrum(sys, zero_V, params)
end

# ============ Spectrum output ============

"""
    write_energy_spectrum(sys::FockSystem, filename::String; V_func=nothing, params=nothing)

Write kinetic spectra (and optional interacting spectra) of all irreps to a text file.

If `V_func` is provided, output both bare kinetic spectra and interacting spectra, with shifts ΔE.
"""
function write_energy_spectrum(sys::FockSystem, filename::String; V_func=nothing, params=nothing)
    io = open(filename, "w")

    println(io, "="^72)
    println(io, "NPHFforFVE spectrum")
    println(io, "="^72)
    println(io, "  Total momentum d = $(sys.d)")
    println(io, "  Total isospin I = $(sys.I)")
    println(io, "  L = $(sys.L), a = $(sys.a) fm  →  L_phys = $(Float64(sys.L) * sys.a) fm")
    println(io, "  Number of channels: $(length(sys.channels))")
    for (i, ch) in enumerate(sys.channels)
        ncut_i = get_Ncut(sys, i)
        n_str = join(["$(s)×$(pt)(j=$(j),I=$(Ij),η=$η,m=$(m isa DynamicMass ? "params.$(m.name)" : m))" for (s,pt,j,Ij,η,m) in
                       zip(ch.species, ch.particle_types, ch.spins, ch.isospins, ch.etas, ch.masses)], ", ")
        println(io, "    Channel $i: \"$(ch.name)\"  N=$(ch.N)  Ncut=$ncut_i  ($n_str)")
    end
    has_V = V_func !== nothing
    println(io, "  Interaction: $(has_V ? "provided" : "none (kinetic energy only)")")
    println(io)

    zero_V(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p) = zero(ComplexF64)

    for Gamma in sys.selected_irreps
        H_kin = build_hamiltonian_block(sys, Gamma, zero_V, params)
        dim = size(H_kin, 1)
        dim == 0 && continue

        ev_kin = sort(real.(eigvals(Hermitian(H_kin))))

        if has_V
            H_full = build_hamiltonian_block(sys, Gamma, V_func, params)
            ev_full = sort(real.(eigvals(Hermitian(H_full))))
        end

        # Moving frame: boost center-of-mass eigenvalues to the moving frame.
        if sys.d != D000
            L_phys = Float64(sys.L) * sys.a
            P_mag = (2π * ħc / L_phys) * sqrt(Float64(sum(abs2, sys.d)))
            ev_kin = [sqrt(E^2 + P_mag^2) for E in ev_kin]
            if has_V
                ev_full = [sqrt(E^2 + P_mag^2) for E in ev_full]
            end
        end

        # Channel-dimension decomposition
        ch_dims = Int[]
        for α in 1:length(sys.channels)
            ch_dims_α = _channel_dim_for_irrep(sys, Gamma, α)
            append!(ch_dims, ch_dims_α)
        end

        println(io, "─"^72)
        println(io, "Irrep: $Gamma  (projected dimension = $dim)")
        if has_V
            println(io, rpad("  #", 5), rpad("E_kin (MeV)", 14), rpad("E_full (MeV)", 14), "ΔE (MeV)")
            println(io, "  " * "─"^50)
            for i in 1:dim
                Δ = ev_full[i] - ev_kin[i]
                mark = abs(Δ) > 0.01 ? (Δ > 0 ? "↑" : "↓") : " "
                println(io, rpad("  $i", 5),
                        rpad(lpad(round(ev_kin[i], digits=4), 10), 14),
                        rpad(lpad(round(ev_full[i], digits=4), 10), 14),
                        lpad(round(Δ, digits=4), 8), "  $mark")
            end
        else
            println(io, rpad("  #", 5), "E_kin (MeV)")
            println(io, "  " * "─"^25)
            for i in 1:dim
                println(io, rpad("  $i", 5), lpad(round(ev_kin[i], digits=4), 10))
            end
        end
        println(io)
    end

    println(io, "="^72)
    close(io)
    println("Spectrum written to: $(abspath(filename))")
    return nothing
end

function _channel_dim_for_irrep(sys::FockSystem, Gamma::String, α::Int)
    ch = sys.channels[α]
    ncut = get_Ncut(sys, α)
    subs = get_isospin_subchannels(ch, sys.I)
    per_spin, per_etas = _expand_per_particle(ch)
    etas_v = Float64.(per_etas)
    multi = length(ch.species) > 1
    dims = Int[]
    for sub in subs
        projs = if multi
            _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
        else
            spin_val = Float64(only(unique(Rational{Int}.(per_spin))))
            _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
        end
        push!(dims, sum(p.n_r for p in projs; init=0))
    end
    return dims
end
