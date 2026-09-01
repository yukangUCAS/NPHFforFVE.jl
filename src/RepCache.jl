# ============================================================
# RepCache — Global cache for representative momenta/helicities
# ============================================================
# Cache keys use physical properties, not channel names or indices; channels with identical physical properties share entries automatically.
# Filled automatically after setup_fock_system constructs a FockSystem.
# ============================================================

const _MOM_REP_CACHE = Dict{NamedTuple, Vector}()
const _HEL_REP_CACHE = Dict{NamedTuple, Vector}()

"""
    get_momentum_reps(species, particle_types, Ncut, d) -> Vector

Return all representative momentum states for this physical configuration. On a cache hit, return them directly; otherwise compute and cache them.
"""
function get_momentum_reps(species::Vector{Int}, particle_types::Vector{Symbol},
                           Ncut::Int, d::Momentum)
    key = (species=Tuple(species), particle_types=Tuple(particle_types),
           Ncut=Ncut, d=d)
    return get!(_MOM_REP_CACHE, key) do
        N = sum(species)
        find_representatives(N; d=d, Ncut=Ncut, species=species, particle_types=particle_types)
    end
end

"""
    get_helicity_reps(representative, species, particle_types, spins, d) -> Vector

Return all independent helicity configurations for one representative momentum. On a cache hit, return them directly; otherwise compute and cache them.
Return an empty list when the representative momentum contains a zero-momentum particle with nonzero spin.
"""
function get_helicity_reps(representative::NTuple{N, Momentum},
                           species::Vector{Int}, particle_types::Vector{Symbol},
                           spins::Vector{Rational{Int}}, d::Momentum) where N
    key = (rep=representative, species=Tuple(species),
           particle_types=Tuple(particle_types), spins=Tuple(spins), d=d)
    return get!(_HEL_REP_CACHE, key) do
        if _helicity_undefined(representative, species, spins)
            return NTuple{N, Rational{Int}}[]
        end
        helicity_representatives(representative; species=species,
                                 particle_types=particle_types, spins=spins, d=d)
    end
end

"""
    cache_channel_reps!(species, particle_types, Ncut, d, spins)

Prepopulate the cache with all representative momenta and helicities of a channel. Skip representative states containing a zero-momentum particle with nonzero spin.
Called after FockSystem construction.
"""
function cache_channel_reps!(species::Vector{Int}, particle_types::Vector{Symbol},
                             Ncut::Int, d::Momentum, spins::Vector{Rational{Int}})
    reps = get_momentum_reps(species, particle_types, Ncut, d)
    for rep in reps
        _helicity_undefined(rep, species, spins) && continue
        get_helicity_reps(rep, species, particle_types, spins, d)
    end
    return reps
end

const _SUBSPACE_STATE_CACHE = Dict{NamedTuple, Vector}()

"""
    get_subspace_states(n_tuple, lambda_tuple, d, spin) -> Vector

Return the subspace-basis-state list for a representative momentum/helicity configuration (canonical order, deduplicated and sorted).
This list corresponds one-to-one with the row indices of X. On a cache hit, return it directly.
"""
function get_subspace_states(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             d::Momentum, spin::Real) where N
    lam_float = Tuple(Float64.(lambda_tuple))
    spin_float = Float64(spin)
    _, group_name = group_for_momentum(d)
    nG = length(group_elements(group_name))
    needs_double = !isinteger(N * spin_float)
    n_base = needs_double ? nG ÷ 2 : nG
    key = (n_tuple=n_tuple, lambda_tuple=lam_float, d=d, spin=spin_float)
    return get!(_SUBSPACE_STATE_CACHE, key) do
        group_els = group_elements(group_name)
        _collect_subspace_states(n_tuple, lam_float, group_els, spin_float, 1.0, n_base)
    end
end

# ============ Internal ============

function _helicity_undefined(representative::NTuple{N, Momentum},
                             species::Vector{Int}, spins::Vector{Rational{Int}}) where N
    per_spin = _expand_spins(spins, species)
    for i in 1:N
        per_spin[i] != 0 && representative[i] == Momentum(0, 0, 0) && return true
    end
    return false
end
