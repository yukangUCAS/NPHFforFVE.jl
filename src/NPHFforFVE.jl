module NPHFforFVE

using StaticArrays
using LinearAlgebra

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

# 三维整数动量矢量类型别名
const Momentum = SVector{3, Int}

# 常用总动量预设
const D000 = Momentum(0, 0, 0)
const D001 = Momentum(0, 0, 1)
const D011 = Momentum(0, 1, 1)
const D111 = Momentum(1, 1, 1)

"""
    single_particle_basis(Ncut::Int) -> Vector{Momentum}

生成所有满足 |n|² = n_x² + n_y² + n_z² ≤ Ncut 的三维整数动量矢量，
按字典序升序排列。
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

返回一个惰性 Channel，迭代生成所有满足约束的 N 粒子动量态。

# 参数
- `N`: 粒子总数 (≥1)
- `d`: 总动量，默认为 `D000 = (0,0,0)`
- `Ncut`: 动量截断，每个粒子满足 |n_i|^2 ≤ Ncut
- `particle_type`: `:distinguishable`, `:boson` 或 `:fermion`（单物种时使用）

# 多物种（分组全同粒子）
- `species`: 各物种粒子数，如 `[1,2]` 表示 1个A + 2个B；`sum(species) == N`
- `particle_types`: 各物种对应的粒子类型，长度同 `species`
  若省略则所有物种共用 `particle_type`

# 示例
```julia
# 单物种
for state in momentum_states(3, Ncut=4, particle_type=:fermion, d=D001)
    println(state)
end

# 多物种：N=3，物种A有1个可区分粒子，物种B有2个全同玻色子
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

计算满足约束的 N 粒子动量态总数，不生成完整态列表。
参数同 `momentum_states`。
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

# ============ 内部函数 ============

function _to_momentum(d)
    if d isa Momentum
        return d
    elseif d isa NTuple{3,Int}
        return Momentum(d)
    elseif d isa AbstractVector{<:Integer}
        return Momentum(d[1], d[2], d[3])
    else
        throw(ArgumentError("总动量 d 必须是 SVector{3,Int}, NTuple{3,Int} 或长度为3的整数向量"))
    end
end

function _validate_particle_type(pt::Symbol)
    pt in (:distinguishable, :boson, :fermion) && return nothing
    throw(ArgumentError("particle_type 必须是 :distinguishable, :boson 或 :fermion"))
end

# 统一物种参数为 (spec_sizes, spec_types) 形式，保证向下兼容
function _normalize_species(N::Int, particle_type::Symbol, species, particle_types)
    if species === nothing
        _validate_particle_type(particle_type)
        return [N], [particle_type]
    end
    if !(species isa AbstractVector{<:Integer})
        throw(ArgumentError("species 必须是整数向量"))
    end
    if sum(species) != N
        throw(ArgumentError("species 各元素之和必须等于 N"))
    end
    if particle_types === nothing
        particle_types = fill(particle_type, length(species))
    end
    if length(species) != length(particle_types)
        throw(ArgumentError("particle_types 长度必须与 species 一致"))
    end
    for pt in particle_types
        _validate_particle_type(pt)
    end
    return species, particle_types
end

# 构建粒子→物种映射: spec_of[k] = 粒子 k 所属物种编号
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

# 递归生成核心
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
                # 仅当最后两个粒子同物种时才检查排序约束
                if N > 1 && spec_of[N-1] == spec_of[N]
                    pt = spec_types[spec_of[N]]
                    if pt == :boson
                        isless(n_last, current[N-1]) && return
                    elseif pt == :fermion
                        !isless(current[N-1], n_last) && return
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

            # 决定下一层起始索引：仅同物种内才施加排序约束
            if level < N && spec_of[level] == spec_of[level+1]
                pt = spec_types[spec_of[level]]
                next_start = if pt == :fermion
                    i + 1
                elseif pt == :boson
                    i
                else
                    1
                end
            else
                next_start = 1  # 不同物种：无约束
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

# ============ 轨道分解 ============

"""
    find_representatives(N::Int; d=Momentum(0,0,0), Ncut::Int, particle_type::Symbol=:distinguishable,
                         species=nothing, particle_types=nothing)

生成所有代表动量态。每个群作用轨道中字典序最小的态被选为代表。

参数同 `momentum_states`，支持多物种。
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

返回某个代表动量在对称群作用下的完整轨道（所有互异态）。
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

# 群元作用于态，仅在各物种内部分别重排至规范序
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
export param_bounds, param_errors, param_limits, param_count, print_params

include("FockSpace.jl")
using .FockSpace: FockChannel, FockSystem, setup_fock_system
using .FockSpace: get_N, get_num_species, get_total_N, get_Ncut, get_isospin_subchannels
using .FockSpace: KineticType, relativistic, nonrelativistic
export FockChannel, FockSystem, setup_fock_system
export get_N, get_num_species, get_total_N, get_Ncut, get_isospin_subchannels
export KineticType, relativistic, nonrelativistic

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
export build_hamiltonian_block, compute_spectrum, compute_kinetic_spectrum, write_energy_spectrum
export SystemBasis, build_V_hel_blocks!
export boost_to_cm

include("UserAPI.jl")
export Project, Config, add_config!, ProjectResult, compute!, setup_project
export generate_potential_template

end # module
