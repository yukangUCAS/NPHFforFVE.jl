# ============ 螺旋度空间 ============
# 本文件由 NPHFforFVE.jl include，函数直接属于 NPHFforFVE 模块

"""
    helicity_representatives(representative::NTuple{N, Momentum};
                             species::Vector{Int}, particle_types::Vector{Symbol},
                             spins::Vector{<:Real}, d=Momentum(0,0,0),
                             isospins=nothing, subsystem_isospins=nothing,
                             total_isospin=nothing) -> Vector

返回某个代表动量下所有独立的螺旋度配置。

先校验零动量+非零自旋约束，再计算 Per({n⃗}) 小群，
建立螺旋度等价关系后返回等价类代表（按字典序排列）。

# 参数
- `representative`: N 粒子代表动量态
- `species`: 各物种粒子数，如 `[1,2]`
- `particle_types`: 各物种类型 (`:distinguishable`, `:boson`, `:fermion`)
- `spins`: 各物种的自旋量子数，支持整数和半整数（Rational{Int} 或 .5）
- `d`: 总动量
- `isospins`: 各物种单粒子同位旋（可选），长度同 species，如 `[1//2, 1//2]`
- `subsystem_isospins`: 各子体系耦合后的总同位旋（可选），长度同 species，如 `[0, 1]`
- `total_isospin`: 总体系总同位旋（可选）

三个同位旋参数要么全为 `nothing`，要么全提供。

# 返回
螺旋度元组列表（按粒子展开为 N 元组），每元组内 λᵢ ∈ {-s, -s+1, ..., s}
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

# ============ 输入校验 ============

function _validate_helicity_inputs(representative, species, particle_types, spins)
    n_particles = length(representative)
    if sum(species) != n_particles
        throw(ArgumentError("species 之和 ($(sum(species))) 必须等于代表动量的粒子数 ($n_particles)"))
    end
    if length(species) != length(particle_types)
        throw(ArgumentError("particle_types 长度 ($(length(particle_types))) 必须与 species ($(length(species))) 一致"))
    end
    if length(spins) != length(species)
        throw(ArgumentError("spins 长度 ($(length(spins))) 必须与 species ($(length(species))) 一致"))
    end

    all_spins = _expand_spins(spins, species)
    for i in 1:n_particles
        s = all_spins[i]
        n = representative[i]
        if s != 0 && n == Momentum(0, 0, 0)
            throw(ArgumentError("粒子 $i 自旋 s=$s ≠ 0 但动量为零，螺旋度未定义"))
        end
    end
end

# ============ 同位旋参数校验 ============

function _validate_isospin_inputs(species::Vector{Int}, isospins, subsystem_isospins, total_isospin)
    any_provided = isospins !== nothing || subsystem_isospins !== nothing || total_isospin !== nothing
    if !any_provided
        return nothing
    end

    if isospins === nothing || subsystem_isospins === nothing || total_isospin === nothing
        throw(ArgumentError("isospins, subsystem_isospins, total_isospin 必须同时提供或同时省略"))
    end

    if length(isospins) != length(species)
        throw(ArgumentError("isospins 长度 ($(length(isospins))) 必须与 species ($(length(species))) 一致"))
    end
    if length(subsystem_isospins) != length(species)
        throw(ArgumentError("subsystem_isospins 长度 ($(length(subsystem_isospins))) 必须与 species ($(length(species))) 一致"))
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
        throw(ArgumentError("$label = $val < 0，同位旋必须非负"))
    end
    twice = 2 * val
    if !isinteger(twice)
        throw(ArgumentError("$label = $val 不是整数或半整数"))
    end
    return nothing
end

# ============ Per({n⃗}) 计算 ============

"""
    _compute_per(representative, group, spec_sizes, spec_types)

返回等价关系生成器列表。每个生成器为 (parity::Int, inv_perm::Vector{Int})，
parity = 1（proper）或 -1（improper），inv_perm 为全局逆置换（1-indexed）。

对于 group 中每个群元 g，检查每个物种子系统的动量多集在 g 作用下是否不变，
若是则枚举所有合法置换，搜集为生成器。
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

        # 枚举所有物种置换组合，构建全局置换
        for combo in _cartesian_product(all_perms_per_species)
            # combo 是所有物种子系统局部置换拼接的平坦向量
            # 需要加上各物种的偏移量转为全局索引
            global_perm = Vector{Int}(undef, N)
            pos = 1
            for (s, sz) in enumerate(spec_sizes)
                base = offsets[s] - 1
                for local_idx in 1:sz
                    # combo[pos] 是局部目标索引（1..sz），转为全局
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

# ============ 置换工具 ============

# 回溯枚举所有 orig[i] → trans 的双射置换（返回的每个置换 p 满足 trans[i] == orig[p[i]]）
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

# ============ 螺旋度配置生成 ============

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

# s 对应的螺旋度可能值: (-s, -s+1, ..., s)，用最具体类型存储
function _helicity_values(s::Real)
    if isinteger(s)
        s_int = Int(s)
        return collect(Int, -s_int:s_int)
    else
        s_rat = Rational{Int}(s)
        den = denominator(s_rat)
        if den != 2
            throw(ArgumentError("自旋 $s 不是整数或半整数"))
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

# ============ 等价关系与并查集 ============

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

    # 每等价类取字典序最小者
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

# ============ 群元宇称 ============

"""
    _parity_of(g_idx::Int, n_bosonic::Int) -> Int

通过群元索引判断宇称。约定：群元按 [固有旋转 ..., 非固有旋转 ...] 排列，
前 n_bosonic/2 个为固有旋转 (parity=+1)，后 n_bosonic/2 个为非固有旋转 (parity=-1)。
对于双覆盖群，此规律以 n_bosonic 为周期扩展。
"""
function _parity_of(g_idx::Int, n_bosonic::Int)
    return ((g_idx - 1) % n_bosonic) < div(n_bosonic, 2) ? 1 : -1
end

# ============ Wigner 角与螺旋度相位 ============

"""
    _momentum_to_euler(n::Momentum) -> (θ::Float64, φ::Float64)

计算将 z 轴转到动量 n 方向的标准旋转 R_st(n) = e^{-iφ J_z} e^{-iθ J_y} 的 Euler 角。

约定: 0 ≤ θ ≤ π, -π ≤ φ < π。
当 n ∥ z 轴时: n_z > 0 → φ = 0; n_z < 0 → φ = -π。
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
        # atan 返回 (-π, π]，归一化到 [-π, π)
        if phi >= pi - 1e-15
            phi = -pi
        end
    end
    return theta, phi
end

"""
    _su2_standard_rotation(theta, phi) -> Matrix{ComplexF64}

返回 D^{1/2}(R_st(n)) = exp(-i φ J_z) exp(-i θ J_y)，2×2 矩阵。
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

返回 D^{1/2}(R_st(n))^{-1}，即上式的厄米共轭。
"""
function _su2_standard_rotation_inv(theta::Float64, phi::Float64)
    # D^{-1} = D† for SU(2)
    return _su2_standard_rotation(theta, phi)'
end

"""
    _so3_to_su2(R::SMatrix{3,3,Int}, sign::Int=1) -> Matrix{ComplexF64}

将 SO(3) 旋转矩阵提升为其 spin-1/2 SU(2) 表示。
使用参数表 _OH_ROTATION_PARAMS 的轴-角参数 (m, ω):
  D^{1/2}(g) = cos(ω/2) I - i sin(ω/2) (m·σ)

sign = ±1 区分双覆盖群的两个 SU(2) 提升。
"""
function _so3_to_su2(R::SMatrix{3,3,Int}, sign::Int=1)
    return sign * SymmetryGroup._OH_PROPER_SU2[R]
end

"""
    _wigner_angle(n::Momentum, g::SMatrix{3,3,Int}, sign::Int=1) -> Float64

计算 Wigner 旋转角 φ_w(n, g)，由方程
D^{1/2}(e^{-iφ_w J_z}) = D^{1/2}(R_st^{-1}(gn) g R_st(n)) 定义。

返回值在 (-π, π] 范围内。

sign = ±1 区分双覆盖群的两个 SU(2) 提升。
仅适用于固有旋转（det(g) = +1）。
"""
function _wigner_angle(n::Momentum, g::SMatrix{3,3,Int}, sign::Int=1)
    gn = apply_transform(g, n)

    θ_n,  φ_n  = _momentum_to_euler(n)
    θ_gn, φ_gn = _momentum_to_euler(gn)

    D_n    = _su2_standard_rotation(θ_n, φ_n)
    D_gn_inv = _su2_standard_rotation_inv(θ_gn, φ_gn)
    D_g    = _so3_to_su2(g, sign)

    U = D_gn_inv * D_g * D_n

    # U 应对角化为 [[e^{-iφ_w/2}, 0], [0, e^{+iφ_w/2}]]
    # atan(imag, real) ∈ (-π, π], 故 φ_w ∈ [-2π, 2π) — 这正是 SU(2) 需要的 4π 范围
    # 不能归一化到 [-π, π]：对半整数自旋 e^{-iφ_w/2} 周期为 4π，±2π 会改变符号
    φ_w = -2 * atan(imag(U[1,1]), real(U[1,1]))
    return φ_w
end

"""
    helicity_phase(n::Momentum, g::SMatrix{3,3,Int}, lambda::Real, sign::Int=1) -> ComplexF64

计算固有旋转 g 作用在螺旋度态上的相位 e^{-iλ φ_w(n, g)}。

sign = ±1 区分双覆盖群的两个 SU(2) 提升。
仅适用于固有旋转。
"""
function helicity_phase(n::Momentum, g::SMatrix{3,3,Int}, lambda::Real, sign::Int=1)
    φ_w = _wigner_angle(n, g, sign)
    return exp(-im * Float64(lambda) * φ_w)
end

"""
    _parity_helicity_phase(n::Momentum, lambda::Real, s::Real, eta::Real) -> ComplexF64

计算空间反射 P 作用在螺旋度态上的相位:
P |n, λ⟩ = η e^{∓iπs} |-n, -λ⟩

其中指数符号约定:
  -π < φ_n < 0  → e^{-iπs}
   0 ≤ φ_n < π  → e^{+iπs}

η 为粒子的内禀宇称，由用户输入。
"""
function _parity_helicity_phase(n::Momentum, lambda::Real, s::Real, eta::Real)
    _, phi = _momentum_to_euler(n)
    sign_phase = (0.0 <= phi < pi) ? 1.0 : -1.0
    return Float64(eta) * exp(sign_phase * im * pi * Float64(s))
end
