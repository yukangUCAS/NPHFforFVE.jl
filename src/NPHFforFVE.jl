module NPHFforFVE

using StaticArrays
using LinearAlgebra
using KrylovKit
using SparseArrays

export single_particle_basis, momentum_states, count_momentum_states
export D000, D001, D011, D111, Momentum
export group_elements, group_for_momentum, apply_transform
export O_h, C4v, C2v, C3v
export irrep_matrices, irrep_matrix, OH_IRREP_NAMES
export find_representatives, group_orbit
export helicity_representatives

include("SymmetryGroup.jl")
using .SymmetryGroup: group_elements, group_for_momentum, apply_transform
using .SymmetryGroup: O_h, C4v, C2v, C3v
using .SymmetryGroup: irrep_matrices, irrep_matrix, OH_IRREP_NAMES
using .SymmetryGroup: OH2_IRREP_NAMES, LG_IRREP_NAMES

const Momentum = SVector{3, Int}

const D000 = Momentum(0, 0, 0)
const D001 = Momentum(0, 0, 1)
const D011 = Momentum(0, 1, 1)
const D111 = Momentum(1, 1, 1)

"""
    single_particle_basis(Ncut::Int) -> Vector{Momentum}

Generate all three-dimensional integer momentum vectors satisfying
|n|² = n_x² + n_y² + n_z² ≤ Ncut, sorted lexicographically.
"""
function single_particle_basis(Ncut::Int)
    basis = Momentum[]
    rmax = floor(Int, sqrt(Ncut))
    for nx in -rmax:rmax
        nx2 = nx * nx
        nx2 > Ncut && continue
        for ny in -rmax:rmax
            ny2 = ny * ny
            nxy2 = nx2 + ny2
            nxy2 > Ncut && continue
            for nz in -rmax:rmax
                nz2 = nz * nz
                nxy2 + nz2 > Ncut && continue
                push!(basis, Momentum(nx, ny, nz))
            end
        end
    end
    sort!(basis)
    return basis
end

"""
    momentum_states(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                    species=nothing, particle_types=nothing)

Return a lazy `Channel` that iterates over all constrained `N`-particle momentum states.

# Arguments
- `N`: total number of particles (≥ 1)
- `d`: total momentum; defaults to `D000 = (0,0,0)`
- `Ncut`: momentum cutoff; every particle satisfies |n_i|² ≤ Ncut
- `particle_type`: `:distinguishable`, `:boson`, or `:fermion` for a single species

# Multiple species
- `species`: particle count for each species; `[1,2]` means one particle of species A and two of species B, with `sum(species) == N`
- `particle_types`: particle type for each species, with the same length as `species`; if omitted, all species use `particle_type`

# Examples
```julia
# One species
for state in momentum_states(3, Ncut=4, particle_type=:fermion, d=D001)
    println(state)
end

# Multiple species: N=3, with one distinguishable A particle and two identical B bosons
for state in momentum_states(3, Ncut=4, species=[1,2], particle_types=[:distinguishable, :boson])
    println(state)
end
```
"""
function momentum_states(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                         species=nothing, particle_types=nothing)
    d_vec = _to_momentum(d)
    basis = single_particle_basis(Ncut)
    spec_sizes, spec_types = _normalize_species(N, particle_type, species, particle_types)
    return Channel{NTuple{N, Momentum}}() do ch
        _generate!(ch, basis, N, d_vec, Ncut, spec_sizes, spec_types)
    end
end

"""
    count_momentum_states(N; d, Ncut, particle_type, species, particle_types) -> Int

Count all constrained `N`-particle momentum states without constructing the complete list.
Arguments are the same as for `momentum_states`.
"""

function count_momentum_states(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                               species=nothing, particle_types=nothing)
    d_vec = _to_momentum(d)
    basis = single_particle_basis(Ncut)
    spec_sizes, spec_types = _normalize_species(N, particle_type, species, particle_types)
    counter = Ref(0)
    _generate!(counter, basis, N, d_vec, Ncut, spec_sizes, spec_types)
    return counter[]
end

# ============ Internal functions ============

function _to_momentum(d)
    if d isa Momentum
        return d
    elseif d isa NTuple{3,Int}
        return Momentum(d)
    elseif d isa AbstractVector{<:Integer}
        return Momentum(d[1], d[2], d[3])
    else
        throw(ArgumentError("total momentum d must be an SVector{3,Int}, NTuple{3,Int}, or an integer vector of length 3"))
    end
end

function _validate_particle_type(pt::Symbol)
    pt in (:distinguishable, :boson, :fermion) && return nothing
    throw(ArgumentError("particle_type must be :distinguishable, :boson, or :fermion"))
end

# Normalize species parameters to (spec_sizes, spec_types) for backward compatibility.
function _normalize_species(N::Int, particle_type::Symbol, species, particle_types)
    if species === nothing
        _validate_particle_type(particle_type)
        return [N], [particle_type]
    end
    if !(species isa AbstractVector{<:Integer})
        throw(ArgumentError("species must be an integer vector"))
    end
    if sum(species) != N
        throw(ArgumentError("the entries of species must sum to N"))
    end
    if particle_types === nothing
        particle_types = fill(particle_type, length(species))
    end
    if length(species) != length(particle_types)
        throw(ArgumentError("particle_types must have the same length as species"))
    end
    for pt in particle_types
        _validate_particle_type(pt)
    end
    return species, particle_types
end

# Construct particle-to-species map: spec_of[k] is the species index of particle k.
function _build_spec_of(N::Int, spec_sizes::Vector{Int})
    spec_of = Vector{Int}(undef, N)
    idx = 1
    for (s, sz) in enumerate(spec_sizes)
        for _ in 1:sz
            spec_of[idx] = s
            idx += 1
        end
    end
    return spec_of
end

# Recursive generation core
function _generate!(output, basis::Vector{Momentum}, N::Int, d::Momentum,
                    Ncut::Int, spec_sizes::Vector{Int}, spec_types)
    n_basis = length(basis)
    current = Vector{Momentum}(undef, N)
    spec_of = _build_spec_of(N, spec_sizes)

    function recurse(level::Int, partial_sum::Momentum, start_idx::Int)
        if level == N
            n_last = d - partial_sum
            if sum(abs2, n_last) <= Ncut
                current[N] = n_last
                # Check ordering constraints only when the last two particles have the same species.
                if N > 1 && spec_of[N-1] == spec_of[N]
                    pt = spec_types[spec_of[N]]
                    if pt == :boson || pt == :fermion
                        isless(n_last, current[N-1]) && return
                    end
                end
                _emit!(output, current)
            end
            return
        end

        for i in start_idx:n_basis
            n_i = basis[i]
            new_sum = partial_sum + n_i
            diff = d - new_sum
            remaining = N - level

            if sum(abs2, diff) > remaining * remaining * Ncut
                continue
            end

            current[level] = n_i

            # Determine the next-level starting index: impose ordering constraints only within a species.
            if level < N && spec_of[level] == spec_of[level+1]
                pt = spec_types[spec_of[level]]
                next_start = if pt == :boson || pt == :fermion
                    i
                else
                    1
                end
            else
                next_start = 1  # Different species: no constraint.
            end

            recurse(level + 1, new_sum, next_start)
        end
    end

    recurse(1, zero(Momentum), 1)
    return nothing
end

function _emit!(ch::Channel, current::Vector{Momentum})
    state = ntuple(i -> current[i], length(current))
    put!(ch, state)
end

function _emit!(counter::Ref{Int}, current::Vector{Momentum})
    counter[] += 1
end

# ============ Orbit decomposition ============

"""
    find_representatives(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                         species=nothing, particle_types=nothing)

Generate all representative momentum states. The lexicographically smallest state in each group-action orbit is selected as the representative.

Arguments are the same as for `momentum_states`, including multiple-species support.
"""
function find_representatives(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                              species=nothing, particle_types=nothing)
    d_vec = _to_momentum(d)
    spec_sizes, spec_types = _normalize_species(N, particle_type, species, particle_types)
    _, group_name = group_for_momentum(d_vec)
    group = group_elements(group_name)

    seen = Set{NTuple{N, Momentum}}()
    representatives = NTuple{N, Momentum}[]

    for state in momentum_states(N, d=d_vec, Ncut=Ncut, species=spec_sizes, particle_types=spec_types)
        state in seen && continue
        push!(representatives, state)
        orbit = _compute_orbit(state, group, spec_sizes, spec_types)
        union!(seen, orbit)
    end

    return representatives
end

"""
    group_orbit(representative::NTuple{N, Momentum}; d=Momentum(0,0,0),
                particle_type::Symbol=:distinguishable, species=nothing, particle_types=nothing) where N

Return the complete orbit of a representative momentum state under the symmetry group.
"""
function group_orbit(representative::NTuple{N, Momentum}; d=Momentum(0,0,0),
                     particle_type::Symbol=:distinguishable, species=nothing, particle_types=nothing) where N
    d_vec = _to_momentum(d)
    spec_sizes, spec_types = _normalize_species(N, particle_type, species, particle_types)
    _, group_name = group_for_momentum(d_vec)
    group = group_elements(group_name)
    return _compute_orbit(representative, group, spec_sizes, spec_types)
end

function _compute_orbit(state::NTuple{N, Momentum}, group::Vector{<:SMatrix{3,3,Int}},
                        spec_sizes::Vector{Int}, spec_types) where N
    orbit_states = Set{NTuple{N, Momentum}}()
    for g in group
        transformed = _apply_group_to_state(g, state, spec_sizes, spec_types)
        push!(orbit_states, transformed)
    end
    return collect(orbit_states)
end

# Apply a group element to a state; reorder into canonical order separately within each species.
function _apply_group_to_state(g::SMatrix{3,3,Int}, state::NTuple{N, Momentum},
                               spec_sizes::Vector{Int}, spec_types) where N
    transformed = Momentum[apply_transform(g, state[i]) for i in 1:N]
    offset = 1
    for (s, sz) in enumerate(spec_sizes)
        if spec_types[s] != :distinguishable
            sort!(@view transformed[offset:offset+sz-1])
        end
        offset += sz
    end
    return Tuple(transformed)
end

include("SpinSpace.jl")

include("IsospinSpace.jl")
export get_SN_irrep_names, get_SN_irrep_dim, get_SN_irrep_matrices, get_SN_irrep_matrix
export get_SN_element_index
export isospin_decomposition, multi_isospin_decomposition
export charge_to_isospin_cg
export IsospinDecomposition, MultiIsospinDecomposition, MultiIsospinEntry
export ChargeToIsospinCG
export SN_ELEMENTS

include("SpinCG.jl")
export SpinCG, spin_cg_coefficients, get_coeffs

include("RepCache.jl")
export get_momentum_reps, get_helicity_reps, cache_channel_reps!
export get_subspace_states

include("HelicityRotation.jl")
export wigner_D, get_rotation_vector, build_V_hel

include("ParamStruct.jl")
export @params, to_vector, from_vector, param_names, param_defaults
export param_count, print_params

include("FockSpace.jl")
using .FockSpace: FockChannel, FockSystem, setup_fock_system, SubchannelExclusion
using .FockSpace: get_N, get_num_species, get_total_N, get_Ncut, get_isospin_subchannels
using .FockSpace: _is_subchannel_excluded, _active_subchannels
using .FockSpace: KineticType, relativistic, nonrelativistic
using .FockSpace: DynamicMass, dynamic_mass, has_dynamic_mass
using .FockSpace: resolve_mass, resolve_masses, resolve_particle_masses
export FockChannel, FockSystem, setup_fock_system
export SubchannelExclusion
export get_N, get_num_species, get_total_N, get_Ncut, get_isospin_subchannels
export KineticType, relativistic, nonrelativistic
export DynamicMass, dynamic_mass, has_dynamic_mass
export resolve_mass, resolve_masses, resolve_particle_masses

const ħc = 197.327  # MeV·fm

include("Projection.jl")
export build_I_matrix, lowdin_orthogonalize
export build_X_matrix, build_S_matrix
export subspace_projection, project_interaction, project_V
export _collect_subspace_states
export build_I_matrix_zero_momentum
export build_X_matrix_zero_momentum
export build_S_matrix_zero_momentum

include("Hamiltonian.jl")
export build_hamiltonian_block, compute_spectrum, compute_spectrum_eigs
export compute_spectrum_factorized, compute_kinetic_spectrum, write_energy_spectrum
export channel_decomposition
export SystemBasis, build_V_hel_blocks!
export boost_to_cm

include("CacheManagement.jl")
export cache_info, clear_caches!

include("UserAPI.jl")
export Project, Config, add_config!, exclude_subchannel!, include_subchannel!
export ProjectRunInfo, ProjectResult, compute!, write_spectrum, setup_project
export PreparedSpectrumProject, prepare_spectrum
export SpectrumLevel, SpectrumGroup, SpectrumDataset, spectrum_chisq
export SpectrumFitProblem
function minuit end
function best_params end
export minuit, best_params
export AffinePreparedProject, AffineBasisCache, prepare_affine
export generate_potential_template
export generate_subchannel_report

end # module
