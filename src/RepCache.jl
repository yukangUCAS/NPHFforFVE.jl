# ============================================================
# RepCache — 代表动量/螺旋度的全局缓存
# ============================================================
# 缓存键基于物理属性（非道名/索引），相同物理属性的道自动共享缓存。
# setup_fock_system 构造 FockSystem 后自动填充。
# ============================================================

const _MOM_REP_CACHE = Dict{NamedTuple, Vector}()
const _HEL_REP_CACHE = Dict{NamedTuple, Vector}()

"""
    get_momentum_reps(species, particle_types, Ncut, d) -> Vector

返回该物理配置下的所有代表动量态。命中缓存则直接返回，否则计算并写入缓存。
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

返回某个代表动量下所有独立的螺旋度构型。命中缓存则直接返回，否则计算并写入缓存。
若代表动量含零动量+非零自旋粒子，返回空列表。
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

预填充某个道的所有代表动量和代表螺旋度到缓存中。跳过零动量+非零自旋的代表态。
在 FockSystem 构造后调用。
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

返回某个代表动量+螺旋度构型对应的子空间基态列表（规范序，去重排序）。
此列表与 X 矩阵的行索引一一对应。命中缓存则直接返回。
"""
function get_subspace_states(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             d::Momentum, spin::Real) where N
    lam_float = Tuple(Float64.(lambda_tuple))
    spin_float = Float64(spin)
    _, group_name = group_for_momentum(d)
    nG = length(group_elements(group_name))
    needs_double = !isinteger(spin_float)
    n_base = needs_double ? nG ÷ 2 : nG
    key = (n_tuple=n_tuple, lambda_tuple=lam_float, d=d, spin=spin_float)
    return get!(_SUBSPACE_STATE_CACHE, key) do
        group_els = group_elements(group_name)
        _collect_subspace_states(n_tuple, lam_float, group_els, spin_float, 1.0, n_base)
    end
end

# ============ 内部 ============

function _helicity_undefined(representative::NTuple{N, Momentum},
                             species::Vector{Int}, spins::Vector{Rational{Int}}) where N
    per_spin = _expand_spins(spins, species)
    for i in 1:N
        per_spin[i] != 0 && representative[i] == Momentum(0, 0, 0) && return true
    end
    return false
end
