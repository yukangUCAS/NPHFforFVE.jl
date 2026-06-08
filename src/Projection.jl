# ============================================================
# Projection — I 矩阵、Löwdin 正交化、不可约表示基展开
# ============================================================
#
# 对每个子空间 S(n^r, λ, [κ]) 和目标群不可约表示 Γ：
#   1. 构造 I 矩阵
#   2. Löwdin 正交化 → 非零本征值 Z_r 和本征矢 c^r
#   3. 展开为规范序动量态的线性组合 → 系数矩阵 X
#
# 约定：优先支持单物种（所有粒子全同），多物种留待后续推广。

# ============ 相位计算工具 ============

"""
    _total_particle_phase(n::Momentum, g::SMatrix{3,3,Int}, g_idx::Int, n_base::Int,
                          lambda::Float64, spin::Float64, eta_i::Float64, sign::Int=1)
                          -> (ComplexF64, Int)

计算群元 g 作用在螺旋度态 |n,λ⟩ 上的总相位和螺旋度宇称 P(g)。

宇称通过群元索引判断（参见 _parity_of），g 矩阵仅用于计算 Wigner 角等相位。
sign = ±1 区分双覆盖群的两个 SU(2) 提升（非双覆盖群始终为 1）。
n_base 为基础 O(3) 群大小（双覆盖群为全群的一半）。
eta_i 为该粒子的内禀宇称。

- 固有旋转 (parity=+1): phase = e^{-iλ φ_w(n,g)}, P(g)=+1
- 非固有旋转 (parity=-1): g = P·R, R=-g 为固有旋转
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
        R = -g  # 固有部分，det(R)=+1
        h_phase = helicity_phase(n, R, lambda, sign)
        Rn = apply_transform(R, n)
        p_phase = _parity_helicity_phase(Rn, lambda, spin, eta_i)
        return h_phase * p_phase, -1
    end
end


""" _total_state_phase(n_tuple, g, g_idx, n_base, lambda_tuple, spin, eta, sign=1) -> ComplexF64

计算群元 g 对 N 粒子态的总相位：∏_i phase_i。
sign = ±1 区分双覆盖群的两个 SU(2) 提升。
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


# ============ 置换工具 ============

"""
    _permutation_sign(p::Vector{Int}) -> Int

计算置换 p 的奇偶性（+1 偶, -1 奇），通过逆序数。
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


# ============ 螺旋度等价类工具 ============

"""
    _compute_per_single_species(n_tuple, group_elements, n_base)

计算单物种情形的 Per({n}) 群生成元列表。
每个生成元为 (parity::Int, inv_perm::Vector{Int})，
等价关系: λ'_i = parity × λ[inv_perm[i]]。
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

通过 BFS 计算 lambda_tuple 在 Per 群生成元下的等价类。
返回该类中所有螺旋度元组（含输入元组自身），并归一化 -0.0 → 0.0。
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


# ============ I 矩阵 ============

"""
    build_I_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                   kappa::String, Gamma::String,
                   group_elements::Vector{<:SMatrix{3,3,Int}},
                   irrep_mats::Vector{<:AbstractMatrix},
                   species_type::Symbol, spin::Float64, etas::Vector{Float64},
                   n_base::Int) where N -> Matrix{ComplexF64}

构造子空间 S(n^r, λ, [κ]) 的 I 矩阵。

公式:
  I_{(b,ν'),(a,ν)} = Σ_{g∈Per({n})} D*_{ν'ν}(g) × phase(g)
                     × Σ_{s∈S_N} δ(s) × R^{[κ]}_{ba}(s)
                     × Π_i δ_{n_{s_i}, g·n_i} × Π_i δ_{λ_{s_i}, P(g)·λ_i}

其中 δ(s) 对费米子体系为置换奇偶性 sign(s)，对玻色子体系恒为 1。
动量匹配：n_{s_i} = g·n_i；螺旋度匹配：λ_{s_i} = P(g)·λ_i。
无需预计算稳定子，直接对全 S_N 求和，由 Kronecker δ 自然筛选。

I 矩阵大小: n_λ_class × dim([κ]) × dim(Γ) 的方阵。
索引约定: 行/列 = (λ_idx, b, ν), 按列优先展平，λ 最快变。

λ 指标在螺旋度等价类上展开，使得螺旋度匹配退化为精确 Kronecker δ，
从而正确包含非固有旋转（宇称）贡献。

etas 为各物种单粒子内禀宇称向量（当前单物种支持 length(etas)==1）。
"""
function build_I_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         kappa::String, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species_type::Symbol, spin::Float64, etas::Vector{Float64},
                         n_base::Int) where N
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    orig_vec = collect(n_tuple)
    all_s = SN_ELEMENTS[N]

    # 计算螺旋度等价类 — 在等价类上展开 λ 指标
    per_generators = _compute_per_single_species(n_tuple, group_elements, n_base)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)
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
            # 动量匹配: n_{s_i} = g·n_i
            mom_ok = true
            for i in 1:N
                n_tuple[s[i]] != trans[i] && (mom_ok = false; break)
            end
            mom_ok || continue

            inv_s = _inverse_permutation(s)

            fermion_sign = (species_type == :fermion) ? _permutation_sign(s) : 1

            s_idx = get_SN_element_index(N, s)
            R_s = get_SN_irrep_matrix(N, kappa, s_idx)

            # 遍历等价类中所有源螺旋度
            for (src_idx, lam_src) in enumerate(lambda_class)
                # 相位依赖源 λ（不同等价类成员可给不同相位）
                state_phase = _total_state_phase(n_tuple, g, g_idx, n_base, lam_src, spin, etas[1], sign)

                # 目标螺旋度: λ'_i = P(g)·λ_src[s^{-1}[i]]，必定在等价类内
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


# ============ 多物种辅助函数 ============

"""
    _is_within_species_permutation(p::Vector{Int}, species::Vector{Int}) -> Bool

检查置换 p 是否仅在各个物种内部置换，不跨物种。
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

多物种版 Per 群生成元。与单物种版的区别：只考虑物种内置换。
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

多物种总态相位：每粒子使用各自所属物种的 spin 和 eta。
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

多物种费米子置换符号：对每个费米子物种分别计算 sign(s|_k)，再相乘。
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


# ============ 多物种 I 矩阵 ============

"""
    build_I_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                   group_elements, irrep_mats,
                   species, particle_types, spins, etas, n_base)

多物种版 I 矩阵。与单物种版的核心区别：
- 置换求和限制在直积群 S_{N₁}×⋯×S_{Nₖ} 内
- R 矩阵为各物种 S_N irrep 的张量积
- 每粒子 spin / eta 按所属物种独立
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

    # 直积群元素 (替代 SN_ELEMENTS[N])
    prod_gens = _product_group_generators(species)

    # 每粒子 spin / eta 展开
    per_particle_spins = Float64[]
    per_particle_etas = Float64[]
    for (k, Nk) in enumerate(species)
        for _ in 1:Nk
            push!(per_particle_spins, spins[k])
            push!(per_particle_etas, etas[k])
        end
    end

    # 多物种 Per 群生成元 + 螺旋度等价类
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

            # 动量匹配 δ
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

                # 目标螺旋度: λ'_i = P(g)·λ_{s^{-1}[i]}
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


# ============ 零动量 I 矩阵（新统一框架）============

"""
    _spin_values(j::Rational{Int}) -> Vector{Rational{Int}}

返回自旋 j 的所有投影值（从高到低）。
"""
function _spin_values(j::Rational{Int})
    if j == 1//2
        return Rational{Int}[1//2, -1//2]
    elseif j == 1//1
        return Rational{Int}[1//1, 0//1, -1//1]
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end
end

"""
    _spin_tuples(j::Rational{Int}, M::Int) -> Vector{NTuple{M, Rational{Int}}}

生成 M 个自旋 j 粒子的所有自旋投影组态 (σ₁,...,σ_M)，按字典序排列。
σ 索引 = 1..(2j+1)^M。
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

生成 M 个自旋 j 粒子的规范自旋组态（降序排列: σ₁ ≥ σ₂ ≥ ... ≥ σ_M）。
去除 S_M 置换冗余，仅保留字典序最大的代表元。
"""
function _canonical_spin_tuples(j::Rational{Int}, M::Int)
    M == 0 && return NTuple{0, Rational{Int}}[()]
    vals = _spin_values(j)  # 已从高到低
    result = NTuple{M, Rational{Int}}[]
    current = Vector{Rational{Int}}(undef, M)
    function descend(pos, start)
        if pos > M
            push!(result, NTuple{M, Rational{Int}}(copy(current)))
            return
        end
        for k in start:length(vals)
            current[pos] = vals[k]
            descend(pos + 1, k)  # k 允许等值
        end
    end
    descend(1, 1)
    return result
end

# 多物种自旋组态生成
function _spin_values_float(spin::Float64)
    n = Int(2 * spin + 1)
    return [Float64(-spin + i) for i in 0:(n-1)]
end

function _multi_spin_tuples(zero_counts::Vector{Int}, spins::Vector{Float64})
    K = length(zero_counts)
    M_total = sum(zero_counts)
    # 每个物种生成自己的自旋组态列表
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

将自旋组态排序为降序规范形，返回 (规范组态, 置换 p)。
满足 canon[p[i]] == spin_tuple[i], i=1..M。
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

返回 S_M × S_{N-M} 子群中所有元素的 (索引, 置换向量) 列表。
仅包含满足 s({1..M}) ⊆ {1..M} 且 s({M+1..N}) ⊆ {M+1..N} 的 s ∈ S_N。
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

# 多物种版本：∏_k (S_{M_k} × S_{N_k-M_k}) 元素
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

返回所有保持自旋组态不变的置换 s ∈ S_M，即满足 σ_{s_i} = σ_i, ∀i。
等值自旋间的置换构成稳定子群。
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

从 3×3 旋转矩阵提取旋转轴 n（单位向量）和旋转角 ω ∈ [0, π]。
"""
function _rotation_axis_angle(R::AbstractMatrix{Float64})
    tr = R[1,1] + R[2,2] + R[3,3]
    cos_omega = clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    omega = acos(cos_omega)

    if omega < 1e-12
        n = [0.0, 0.0, 1.0]
    elseif abs(omega - π) < 1e-10
        # 180° 旋转: 轴从 (R+I)/2 提取
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
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end

    # Oh2 双覆盖：后半群元对半整数自旋取反号
    if denominator(j) == 2 && g_idx > n_base
        D = -D
    end

    return D
end

"""
    _wigner_D_for_element(j, g::SMatrix{3,3,Int}, g_idx::Int, n_base::Int)

从群元矩阵直接计算 Wigner D 矩阵，适用任意点群。
"""
function _wigner_D_for_element(j::Rational{Int}, g::SMatrix{3,3,Int},
                               g_idx::Int, n_base::Int)
    # 提取正常转动部分: 非正常转动 g 可写为 -R，其中 R 为正常转动
    R = det(g) < 0 ? Float64.(-g) : Float64.(g)
    n, omega = _rotation_axis_angle(R)

    if j == 1//2
        D = SymmetryGroup._wigner_D_half(n, omega)
    elseif j == 1//1
        D = SymmetryGroup._wigner_D_one(n, omega)
    else
        throw(ArgumentError("Unsupported spin j=$j"))
    end

    # 双覆盖：后半群元对半整数自旋取反号
    if denominator(j) == 2 && g_idx > n_base
        D = -D
    end

    return D
end

"""
    build_I_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                  group_elements, irrep_mats,
                                  species_type, spin, etas, n_base)

构造含 M 个零动量粒子的子空间 I 矩阵（新统一框架）。

# 公式（zero_momentum_new.md Eq.15）

I_{(σ'₁..σ'_M; ν'; b), (σ₁..σ_M; ν; a)} =
  Σ_g D*_{ν'ν}(g) × phase(g) ×
  Σ_{s ∈ S_M × S_{N-M}} (fermion sign) ×
  (δ: momentum matching for finite-particles) ×
  (δ: helicity matching for finite-particles) ×
  R_{ba}(s) ×
  D^j_{σ'_{s₁},σ₁}(g) × ... × D^j_{σ'_{s_M},σ_M}(g)

# 参数
- `M`: 零动量粒子数（前 M 个粒子，n_tuple[1:M] 应为 0）
- `j`: 单粒子自旋（1//2 或 1//1）
- `n_tuple`: 所有 N 个粒子的代表动量
- `lambda_tuple`: 所有 N 个粒子的螺旋度
- `kappa`: S_N 不可约表示标签
- `Gamma`: 目标 O_h 不可约表示（含宇称后缀）
- 其余参数同 `build_I_matrix`

# I 矩阵维度
d = (2j+1)^M × dim(Γ) × dim([κ])
索引: row/col = σ_idx + nσ × (ν-1) + nσ × dim_Γ × (S_N_idx-1) [0-based → +1]
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
    nσ = Int((2j + 1)^M)           # 零动量自旋组态数
    d = nσ * dim_Gamma * dim_kappa  # I 矩阵总维度

    I = zeros(ComplexF64, d, d)
    d == 0 && return I  # M=0 且 j 无效时

    # 预计算自旋组态列表及 spin value → matrix index 映射
    spin_tuples = _spin_tuples(j, M)
    spin_vals = _spin_values(j)
    spin_to_idx = Dict{Rational{Int}, Int}(v => k for (k, v) in enumerate(spin_vals))

    # S_M × S_{N-M} 子群元素
    sm_snm = _SM_x_SNM_elements(N, M)

    # 预计算所有群元的 Wigner D 矩阵
    wigner_Ds = [_wigner_D_for_element(j, g_idx, n_base) for g_idx in 1:length(group_elements)]

    fermion = (species_type == :fermion)
    orig_vec = collect(n_tuple)
    eta_val = etas[1]

    for (g_idx, g) in enumerate(group_elements)
        # 预筛选：有限动量集在 g 下不变
        trans = [apply_transform(g, n_tuple[i]) for i in 1:N]
        if sort(orig_vec) != sort(trans)
            continue
        end

        parity = _parity_of(g_idx, n_base)
        D_g = irrep_mats[g_idx]
        sign = g_idx > n_base ? -1 : 1
        Dj_g = wigner_Ds[g_idx]  # D^j(g) for zero-momentum spin rotation

        # 有限动量粒子相位
        fin_phase = ComplexF64(1.0, 0.0)
        for i in (M + 1):N
            phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                                lambda_tuple[i], spin, eta_val, sign)
            fin_phase *= phase_i
        end

        # 零动量粒子内禀宇称（仅非固有群元贡献）
        if parity == -1
            fin_phase *= Float64(eta_val)^M
        end

        abs(fin_phase) < 1e-14 && continue

        # 遍历 S_M × S_{N-M} 子群
        for (s_idx, s) in sm_snm
            # 有限动量匹配: n_{s_i} = g·n_i, i=M+1..N
            mom_ok = true
            for i in (M + 1):N
                n_tuple[s[i]] != trans[i] && (mom_ok = false; break)
            end
            mom_ok || continue

            # 螺旋度匹配: λ_{s_i} = P(g)·λ_i, i=M+1..N
            hel_ok = true
            for i in (M + 1):N
                lambda_tuple[s[i]] != parity * lambda_tuple[i] && (hel_ok = false; break)
            end
            hel_ok || continue

            # 费米子置换符号
            fermion_sign = fermion ? _permutation_sign(s) : 1
            if fermion_sign == 0
                continue  # shouldn't happen, but safety
            end

            R_s = get_SN_irrep_matrix(N, kappa, s_idx)

            # 遍历所有指标
            for a in 1:dim_kappa, b in 1:dim_kappa
                R_ba = R_s[b, a]
                abs(R_ba) < 1e-14 && continue
                sN_factor = ComplexF64(fermion_sign) * R_ba

                for nu in 1:dim_Gamma, nup in 1:dim_Gamma
                    Dstar = conj(D_g[nup, nu])
                    abs(Dstar) < 1e-14 && continue
                    g_factor = Dstar * fin_phase * sN_factor

                    # 自旋指标: σ' 行，σ 列
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

# 多物种版本
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

    # 多物种自旋组态
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    nσ = length(spin_tuples)
    d = nσ * dim_Gamma * dim_kappa
    I = zeros(ComplexF64, d, d)
    d == 0 && return I

    # 全局 ZM 位置 → 本地 ZM 索引
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

    # 每 ZM 粒子（本地序）的物种标签
    zero_particle_species = Int[]
    for (k, mk) in enumerate(zero_counts)
        for _ in 1:mk
            push!(zero_particle_species, k)
        end
    end

    # 每物种的 Wigner D 矩阵（仅 spin≠0 的物种需要）
    wigner_Ds_by_species = Dict{Int, Vector{Matrix{ComplexF64}}}()
    spin_to_dj_by_species = Dict{Int, Dict{Float64, Int}}()
    for k in 1:K
        if zero_counts[k] > 0 && spins[k] != 0.0
            wigner_Ds_by_species[k] = [
                _wigner_D_for_element(Rational{Int}(Int(2*spins[k]), 2), g_idx, n_base)
                for g_idx in 1:length(group_elements)]
            vals = _spin_values_float(spins[k])
            spin_to_dj_by_species[k] = Dict(v => i for (i, v) in enumerate(vals))
        end
    end

    # ∏_k (S_{M_k} × S_{N_k-M_k}) 子群元素
    sm_snm = _multi_SM_SNM_elements(species, zero_counts)

    # 每粒子 spin / eta
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

        # 仅 FM 粒子相位 + ZM 内禀宇称（与单物种版一致）
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
            # 动量匹配：对于全局位置 i，检验 ZM/FM 状态一致且 FM 动量相等
            mom_ok = true
            for i in 1:N
                nzi = iszero(n_tuple[i])
                nzs = iszero(n_tuple[s_full[i]])
                nzi == nzs || (mom_ok = false; break)
                (!nzi && n_tuple[s_full[i]] != trans[i]) && (mom_ok = false; break)
            end
            mom_ok || continue

            # 螺旋度匹配（仅有限动量粒子）
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


# ============ Löwdin 正交化 ============

"""
    lowdin_orthogonalize(I::Matrix{ComplexF64}; tol::Float64=1e-12)
        -> (Z::Vector{Float64}, C::Matrix{ComplexF64}, nonzero_indices::Vector{Int})

对角化厄米矩阵 I，返回非零本征值 Z_r、对应本征矢（C 的列）和非零本征值序号。

I 矩阵理论上幂等（up to 常数），本征值应为正整数 Z_r。
"""
function lowdin_orthogonalize(I::Matrix{ComplexF64}; tol::Float64=1e-12)
    # 厄米对角化
    evals, evecs = eigen(Hermitian(I))

    # 筛选非零本征值
    nonzero_idx = Int[]
    nonzero_vals = Float64[]
    for (k, val) in enumerate(evals)
        if abs(val) > tol
            push!(nonzero_idx, k)
            push!(nonzero_vals, real(val))
        end
    end

    if isempty(nonzero_idx)
        return Float64[], Matrix{ComplexF64}(undef, size(I,1), 0), Int[]
    end

    Z = Float64.(evals[nonzero_idx])  # 应为正整数（理论保证）
    C = evecs[:, nonzero_idx]         # 对应本征矢

    return Z, C, nonzero_idx
end

# ============ 不可约表示基展开（X 矩阵）============

"""
    _collect_subspace_states(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                             group_elements::Vector{<:SMatrix{3,3,Int}},
                             spin::Float64, eta::Float64, n_base::Int) where N
        -> Vector{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}}

收集子空间中所有互异的规范序动量-螺旋度态。
遍历等价类中所有 λ 和所有 g ∈ G，计算 (canonical_momenta, canonical_helicities)，去重排序。
n_base 为基础 O(3) 群大小，用于宇称判断。
"""
function _collect_subspace_states(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                                   group_elements::Vector{<:SMatrix{3,3,Int}},
                                   spin::Float64, eta::Float64, n_base::Int) where N
    # 计算螺旋度等价类，在所有等价 λ 上展开
    per_generators = _compute_per_single_species(n_tuple, group_elements, n_base)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)

    seen = Set{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}}()
    for lam_src in lambda_class
        for (g_idx, g) in enumerate(group_elements)
            parity = _parity_of(g_idx, n_base)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]

            # 排序为规范序：先动量后螺旋度，确保等动量时规范形唯一
            keys = collect(zip(trans_mom, trans_hel))
            idx = sortperm(keys)
            canon_mom = Tuple(trans_mom[i] for i in idx)
            # 归一化 -0.0 → 0.0，避免 Set 重复
            canon_hel = Tuple(trans_hel[i] == 0.0 ? 0.0 : trans_hel[i] for i in idx)
            push!(seen, (canon_mom, canon_hel))
        end
    end

    result = collect(seen)
    sort!(result, by = x -> (collect(x[1]), collect(x[2])))
    return result
end

# 多物种版本：每个物种内部分别排序，保证规范序中物种边界不被打乱
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

            # 每个物种内部独立排序，保证物种边界不变
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

收集有限动量粒子(N-M 个）的子空间中所有规范序态。
用于零动量 X 矩阵构建时的有限动量分块。
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

# 多物种版本：每个物种内部分别排序
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
        # 每个物种内部排序
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

从 I 矩阵非零本征矢 c^r 出发，构造系数矩阵 X。

I 矩阵在螺旋度等价类上展开（λ 指标为等价类内成员索引），
C 矩阵的行索引约定与 I 矩阵一致: (λ_idx, b, ν) 按列优先展平。

公式:
  |Γ, r⟩ = √(dimΓ / (|G|·Z_r)) × Σ_{λ,b,ν} c^r_{λ,b,ν}
              × Σ_{g∈G} D^{*}_{1,ν}(g) × phase(g; λ)
              × Σ_p δ(p) × R_{b'b}(p) × |{n'}, λ'; b'⟩

其中 p 是将 (g·n) 排序为规范序 {n'} 的置换，λ'_i = P(g) λ_{p̄_i}。
n_base 为基础 O(3) 群大小，用于宇称和 SU(2) 提升符号；|G| 为全群大小。

X 矩阵: 行 = 规范基态 (n', λ', b'),
       列 = r，共 n_r 列

etas 为各物种单粒子内禀宇称向量（当前单物种支持 length(etas)==1）。
"""
function build_X_matrix(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                         kappa::String, Gamma::String,
                         group_elements::Vector{<:SMatrix{3,3,Int}},
                         irrep_mats::Vector{<:AbstractMatrix},
                         species_type::Symbol, spin::Float64, etas::Vector{Float64},
                         n_base::Int, Z::Vector{Float64}, C::Matrix{ComplexF64}) where N
    dim_kappa = get_SN_irrep_dim(N, kappa)
    dim_Gamma = size(irrep_mats[1], 1)
    nG = length(group_elements)

    # 螺旋度等价类（与 build_I_matrix 一致）
    per_generators = _compute_per_single_species(n_tuple, group_elements, n_base)
    lambda_class = _compute_helicity_equivalence_class(lambda_tuple, per_generators)
    n_lam = length(lambda_class)

    # 收集子空间中所有规范序态（在所有等价 λ 上展开）
    subspace_states = _collect_subspace_states(n_tuple, lambda_tuple, group_elements, spin, etas[1], n_base)
    n_states = length(subspace_states)
    subspace_dim = n_states * dim_kappa

    n_r = length(Z)
    X = zeros(ComplexF64, subspace_dim, n_r)
    n_r == 0 && return X

    state_index = Dict{Tuple{NTuple{N,Momentum}, NTuple{N,Float64}}, Int}()
    for (k, st) in enumerate(subspace_states)
        state_index[st] = (k - 1) * dim_kappa + 1
    end

    # 遍历所有 g ∈ G 和所有源螺旋度
    for (g_idx, g) in enumerate(group_elements)
        parity = _parity_of(g_idx, n_base)
        sign = g_idx > n_base ? -1 : 1
        D_g = irrep_mats[g_idx]

        for (src_idx, lam_src) in enumerate(lambda_class)
            trans_mom = [apply_transform(g, n_tuple[i]) for i in 1:N]
            trans_hel = [parity * lam_src[i] for i in 1:N]

            # 排序为规范序
            keys_g = collect(zip(trans_mom, trans_hel))
            idx_sorted = sortperm(keys_g)
            canon_mom = Tuple(trans_mom[i] for i in idx_sorted)
            canon_hel = Tuple(trans_hel[i] == 0.0 ? 0.0 : trans_hel[i] for i in idx_sorted)

            canon_key = (canon_mom, canon_hel)
            !haskey(state_index, canon_key) && continue
            row_base = state_index[canon_key]

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
                    # C 索引: (src_idx-1)*dim_kappa*dim_Gamma + (b-1)*dim_Gamma + nu
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

# 多物种版本
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

            # 每个物种内部独立排序，与 _collect_subspace_states 一致
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
                # 只接受物种内置换
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

# ============ 零动量 X 矩阵 ============

"""
    build_X_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                  group_elements, irrep_mats,
                                  species_type, spin, etas, n_base,
                                  Z, C) -> Matrix{ComplexF64}

从含 M 个零动量粒子的 I 矩阵非零本征矢 c^r 出发，构造系数矩阵 X。

# 公式（zero_momentum_new.md Eq.25-31）

|Γ, r⟩ = √(dimΓ/(|G|·Z_r)) × Σ_{σ,ν,b} Σ_{g∈G} (c^r)_{σ,ν,b} × D^{Γ*}_{1,ν}(g)
         × (有限动量螺旋度相位) × (内禀宇称 η^M)
         × Σ_{σ'} D^j_{σ'₁,σ₁}(g) × ... × D^j_{σ'_M,σ_M}(g)
         × |σ', {g·n, P(g)λ}; b⟩

规范基态: |σ'_canon, {n'_canon, λ'_canon}; b'⟩
其中 σ'_canon 为降序排列，{n'_canon, λ'_canon} 为有限动量规范序。

X 矩阵: 行 = (σ'_canon, n'_canon, λ'_canon, b')
       列 = r，共 n_r 列，μ=1

# 参数
- `M`: 零动量粒子数（前 M 个粒子）
- `j`: 单粒子自旋（1//2 或 1//1）
- `n_tuple`, `lambda_tuple`: N 粒子动量/螺旋度代表元
- `kappa`, `Gamma`: S_N / 点群不可约表示标签
- `Z`: I 矩阵非零本征值
- `C`: I 矩阵非零本征矢矩阵，C[:, r] 按 (b, ν, σ) 展平（σ 最快变）
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

    # 自旋组态
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

    # Wigner D 矩阵
    wigner_Ds = [_wigner_D_for_element(j, g_idx, n_base) for g_idx in 1:nG]

    # 自旋值 → D 矩阵索引
    spin_vals = _spin_values(j)
    spin_to_dj = Dict(v => k for (k, v) in enumerate(spin_vals))

    # 有限动量子空间态
    fin_n = ntuple(i -> n_tuple[M + i], N - M)
    fin_lam = ntuple(i -> lambda_tuple[M + i], N - M)
    fin_states = _collect_fin_subspace_states(fin_n, fin_lam, group_elements, spin, etas[1], n_base)
    n_fin = length(fin_states)

    fin_state_to_rowbase = Dict{Tuple, Int}()
    for (k, st) in enumerate(fin_states)
        fin_state_to_rowbase[st] = (k - 1) * dim_kappa + 1
    end

    # X 矩阵行维度
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

        # 有限动量粒子相位（螺旋度 + e^{∓iπs} + 内禀宇称）
        fin_phase = ComplexF64(1.0, 0.0)
        for i in (M + 1):N
            phase_i, _ = _total_particle_phase(n_tuple[i], g, g_idx, n_base,
                                                lambda_tuple[i], spin, eta_val, sign)
            fin_phase *= phase_i
        end
        # 零动量粒子内禀宇称
        if parity == -1
            fin_phase *= Float64(eta_val)^M
        end
        abs(fin_phase) < 1e-14 && continue

        # 变换有限动量/螺旋度并规范排序
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
            # M == N: 无有限动量粒子
            fin_row_base = 1
            found_p_fin = Int[]
        end

        # 遍历自旋指标
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
                    # σ' 排序为降序规范形
                    sigmap_canon, p_spin = spin_canon_info[sigmap]
                    σ_canon_idx = canon_spin_to_idx[sigmap_canon]

                    # 合成全置换 p_total ∈ S_M × S_{N-M} ⊂ S_N
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
                    # 非对称化基: 自旋不排序，σ' 各自保留独立行
                    # p_total: ZM 自旋部分为恒等置换
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

                # 遍历 S_N 和 Γ 指标
                for a in 1:dim_kappa
                    for b in 1:dim_kappa  # b' — 输出 S_N 指标
                        R_ba = R_p[b, a]
                        abs(R_ba) < 1e-14 && continue

                        for nu in 1:dim_Gamma
                            # C 的行索引: (a-1)*dimΓ*nσ + (nu-1)*nσ + sigma_idx
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

# 多物种版本
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

    # 多物种自旋组态
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    nσ = length(spin_tuples)

    # ZM 全局位置 → 本地 ZM 索引
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

    # 每 ZM 粒子（本地序）的物种标签
    zm_particle_species = Int[]
    for (k, mk) in enumerate(zero_counts)
        for _ in 1:mk
            push!(zm_particle_species, k)
        end
    end

    # FM 全局位置 → 本地 FM 索引
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

    # 每物种 Wigner D 矩阵
    wigner_Ds_by_species = Dict{Int, Vector{Matrix{ComplexF64}}}()
    spin_to_dj_by_species = Dict{Int, Dict{Float64, Int}}()
    for k in 1:K
        if zero_counts[k] > 0 && spins[k] != 0.0
            wigner_Ds_by_species[k] = [
                _wigner_D_for_element(Rational{Int}(Int(2*spins[k]), 2), g_idx, n_base)
                for g_idx in 1:nG]
            vals = _spin_values_float(spins[k])
            spin_to_dj_by_species[k] = Dict(v => i for (i, v) in enumerate(vals))
        end
    end

    # 有限动量子空间态（多物种版）
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

    # FM 物种块大小（用于 _is_within_species_permutation 过滤）
    fin_species_for_is = Int[species[k] - zero_counts[k] for k in 1:K]
    fin_species_for_is = Int[n for n in fin_species_for_is if n > 0]

    # X 矩阵行维度
    row_dim = nσ * n_fin * dim_kappa
    X = zeros(ComplexF64, row_dim, n_r)
    n_r == 0 && return X

    # 每粒子 spin / eta（用于相位计算）
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

        # 有限动量粒子相位（每粒子用自己的 spin/eta）
        fin_phase = ComplexF64(1.0, 0.0)
        for (fm_local, gp) in enumerate(fm_global_positions)
            sp_k = fm_particle_species[fm_local]
            phase_i, _ = _total_particle_phase(n_tuple[gp], g, g_idx, n_base,
                                                lambda_tuple[gp], spins[sp_k],
                                                etas[sp_k], sign)
            fin_phase *= phase_i
        end
        # 零动量粒子内禀宇称因子 ∏_k η_k^{M_k}
        if parity == -1
            for k in 1:K
                fin_phase *= Float64(etas[k])^zero_counts[k]
            end
        end
        abs(fin_phase) < 1e-14 && continue

        # 变换有限动量/螺旋度并按物种内部排序
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

            # 查找稳定子置换: canon_fin[p_fin[i]] == trans_fin[i]
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

        # 构造全局置换 p_total: ZM 恒等，FM 经 found_p_fin 映射
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

        # 遍历自旋指标
        for sigma_idx in 1:nσ
            sigma = spin_tuples[sigma_idx]

            for sigmap_idx in 1:nσ
                sigmap = spin_tuples[sigmap_idx]

                # D-product: 每 ZM 粒子用所属物种的 Wigner D
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

# ============ 规范基重叠矩阵 S ============

"""
    build_S_matrix(subspace_states::Vector, N::Int, kappa::String, species_type::Symbol) -> Matrix{Float64}

构造规范基的重叠矩阵 S。

    S_{(n',λ'),b'; (n',λ'),b''} = Σ_{t ∈ Stab({n'}, λ')} δ(t) × R^{[κ]}_{b'b''}(t)

其中:
- Stab({n'}, λ') = {t ∈ S_N : n'_{t_i}=n'_i 且 λ'_{t_i}=λ'_i, ∀i}
- δ(t): 玻色子恒为 1，费米子为 sign(t)

S 为块对角矩阵，每块对应一个规范态 (n', λ')。
"""
function build_S_matrix(subspace_states::Vector, N::Int, kappa::String, species_type::Symbol)
    dim_kappa = get_SN_irrep_dim(N, kappa)
    n_states = length(subspace_states)
    S = zeros(Float64, n_states * dim_kappa, n_states * dim_kappa)
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
        S[rb:rb+dim_kappa-1, rb:rb+dim_kappa-1] .= blk
    end
    return S
end

# 多物种版本：稳定子限制在直积群 S_{N₁}×⋯×S_{Nₖ}
function build_S_matrix(subspace_states::Vector, species::Vector{Int}, κ_tuple,
                         particle_types::Vector{Symbol})
    dim_kappa = _kappa_tuple_dim(species, κ_tuple)
    n_states = length(subspace_states)
    N = sum(species)
    S = zeros(Float64, n_states * dim_kappa, n_states * dim_kappa)

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
        S[rb:rb+dim_kappa-1, rb:rb+dim_kappa-1] .= blk
    end
    return S
end

"""
    build_S_matrix_zero_momentum(M, spin_tuples, fin_subspace_states,
                                  N, kappa, species_type) -> Matrix{Float64}

构造零动量情形的重叠矩阵 S（非对称化自旋基底）。

公式（zero_momentum_new.md Eq.39）:
  S_{(σ',n',λ',a'), (σ,n,λ,a)} = Σ_{s∈S_M×S_{N-M}} δ(s) ×
    Π_i δ_{σ'_{s_i},σ_i} × Π_j δ_{n'_{s_{M+j}-M},n_j} × Π_j δ_{λ'_{s_{M+j}-M},λ_j}
    × R_{a'a}(s)

与 `build_X_matrix_zero_momentum` (canonical_spin=false) 的行索引约定一致。
"""
function build_S_matrix_zero_momentum(M::Int, spin_tuples::Vector,
                                       fin_subspace_states::Vector,
                                       N::Int, kappa::String, species_type::Symbol)
    dim_kappa = get_SN_irrep_dim(N, kappa)
    nσ = length(spin_tuples)
    n_fin = length(fin_subspace_states)
    fermion = (species_type == :fermion)

    S = zeros(Float64, nσ * n_fin * dim_kappa, nσ * n_fin * dim_kappa)

    # S_M × S_{N-M} 子群元素列表
    sm_snm = _SM_x_SNM_elements(N, M)

    for (σ_idx, σ) in enumerate(spin_tuples)
        for (fin_idx, (n_p, lam_p)) in enumerate(fin_subspace_states)
            Nfm = length(n_p)
            row_blk_idx = (σ_idx - 1) * n_fin + fin_idx
            row_rb = (row_blk_idx - 1) * dim_kappa + 1

            for (σp_idx, σp) in enumerate(spin_tuples)
                for (finp_idx, (np_p, lamp_p)) in enumerate(fin_subspace_states)
                    col_blk_idx = (σp_idx - 1) * n_fin + finp_idx
                    col_cb = (col_blk_idx - 1) * dim_kappa + 1

                    blk = zeros(Float64, dim_kappa, dim_kappa)

                    for (s_idx, s) in sm_snm
                        # 自旋匹配: σ'_{s_i} = σ_i, i=1..M
                        σ_ok = true
                        for i in 1:M
                            σp[s[i]] != σ[i] && (σ_ok = false; break)
                        end
                        σ_ok || continue

                        # 有限动量匹配: n'_{s_{M+j}-M} = n_j, λ'_{s_{M+j}-M} = λ_j, j=1..Nfm
                        fin_ok = true
                        for j in 1:Nfm
                            sj = s[M + j] - M  # 映射到 1..Nfm
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

# 多物种版本：零动量重叠矩阵
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

    # ZM / FM 全局位置 → 本地索引
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
            row_blk_idx = (σ_idx - 1) * n_fin + fin_idx
            row_rb = (row_blk_idx - 1) * dim_kappa + 1

            for (σp_idx, σp) in enumerate(spin_tuples)
                for (finp_idx, (np_p, lamp_p)) in enumerate(fin_subspace_states)
                    col_blk_idx = (σp_idx - 1) * n_fin + finp_idx
                    col_cb = (col_blk_idx - 1) * dim_kappa + 1

                    blk = zeros(Float64, dim_kappa, dim_kappa)

                    for (s_full, per_s_idx) in sm_snm
                        # ZM 自旋匹配: σ'_{s_full[gp]} = σ_{gp}
                        σ_ok = true
                        for gp in zm_global_positions
                            zm_local = zm_global_to_local[gp]
                            tgt_gp = s_full[gp]
                            tgt_zm_local = zm_global_to_local[tgt_gp]
                            σp[tgt_zm_local] != σ[zm_local] && (σ_ok = false; break)
                        end
                        σ_ok || continue

                        # FM 匹配: n'_{s_full[gp]} = n_{gp}, λ'_{...} = λ_{...}
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

# ============ 高层 API ============

"""
    subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                        d_total, species_type=:boson, spin, etas)

对单个子空间运行完整投影管道：I 矩阵 → Löwdin → X 矩阵。

# 参数
- `n_tuple`: 代表动量 (N 元组)
- `lambda_tuple`: 螺旋度组态 (N 元组)
- `kappa`: S_N 不可约表示标签，如 "[2]", "[1,1]"
- `Gamma`: 目标群不可约表示，如 "A1", "E", "T1+"
- `d_total`: 总动量 (用于确定对称群)
- `species_type`: `:boson` 或 `:fermion`
- `spin`: 单粒子自旋
- `etas`: 各物种单粒子内禀宇称，如 `[1.0]` 或 `[-1.0]`

# 返回
- `Z`: 非零本征值向量
- `X`: 系数矩阵 (子空间维度 × 非零本征值数)，列 r 对应 |Γ, r⟩（μ=1）
- `subspace_states`: 规范基态列表
- `I_evals`: I 矩阵全部本征值 (含零，用于检验幂等性)
"""
function subspace_projection(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                              kappa::String, Gamma::String;
                              d_total::Momentum=D000,
                              species_type::Symbol=:boson,
                              spin::Float64, etas::Vector{Float64}) where N
    # 检测零动量 + 非零自旋粒子 → 零动量管线
    M = count(n -> n == Momentum(0,0,0), n_tuple)
    if M > 0 && spin != 0.0
        return _subspace_projection_zero(M, n_tuple, lambda_tuple, kappa, Gamma;
                                          d_total=d_total, species_type=species_type,
                                          spin=spin, etas=etas)
    end

    # 确定对称群（费米子使用双覆盖）
    needs_double = (species_type == :fermion)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)

    nG = length(group_els)          # 全群大小 |G|
    n_base = needs_double ? nG ÷ 2 : nG  # 基础 O(3) 群大小

    # 1. I 矩阵
    I = build_I_matrix(n_tuple, lambda_tuple, kappa, Gamma,
                        group_els, irrep_mats, species_type, spin, etas, n_base)

    # 用于检验的完整本征值
    I_evals = eigen(Hermitian(I)).values

    # 2. Löwdin 正交化
    Z, C, _ = lowdin_orthogonalize(I)

    # 3. X 矩阵
    X = build_X_matrix(n_tuple, lambda_tuple, kappa, Gamma,
                        group_els, irrep_mats, species_type, spin, etas, n_base, Z, C)

    # 4. 子空间态列表 (与 build_X_matrix 内部调用保持一致)
    subspace_states = _collect_subspace_states(n_tuple, lambda_tuple, group_els,
                                               spin, etas[1], n_base)

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

# ============ 多物种 subspace_projection ============

"""
    _canonicalize_zm_ordering(n_tuple, lambda_tuple, species)

将每个物种内的粒子排序：零动量粒子在前，有限动量粒子在后，同时重排 helicity。
返回 (sorted_n, sorted_lam, zero_counts)。

ZM pipeline 函数要求族内 ZM 优先排序，此函数在 _subspace_projection_zero
内部自动调用，用户无需手动处理。
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

多物种投影管线：计算 I 矩阵 → Löwdin 正交化 → X 矩阵。
检测到零动量且对应物种自旋非零时自动切换 ZM 管线。
"""
function subspace_projection(n_tuple::NTuple{N,Momentum}, lambda_tuple::NTuple{N,Float64},
                             κ_tuple, Gamma::String;
                             d_total::Momentum=D000,
                             species::Vector{Int}, particle_types::Vector{Symbol},
                             spins::Vector{Float64}, etas::Vector{Float64}) where N
    # 检测零动量 + 非零自旋粒子 → 零动量管线
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

    # 有限动量管线
    needs_double = any(pt -> pt == :fermion, particle_types)
    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
    irrep_mats = irrep_matrices(Gamma; group=group_name)

    nG = length(group_els)
    n_base = needs_double ? nG ÷ 2 : nG

    I = build_I_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                        group_els, irrep_mats, species, particle_types, spins, etas, n_base)
    I_evals = eigen(Hermitian(I)).values
    Z, C, _ = lowdin_orthogonalize(I)

    X = build_X_matrix(n_tuple, lambda_tuple, κ_tuple, Gamma,
                        group_els, irrep_mats, species, particle_types, spins, etas, n_base, Z, C)

    subspace_states = _collect_subspace_states(n_tuple, lambda_tuple, group_els,
                                                species, spins, etas, n_base)

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

"""
    _subspace_projection_zero(n_tuple, lambda_tuple, κ_tuple, Gamma; ...)

多物种零动量粒子投影管线。
自动对输入进行 ZM 优先规范化排序，然后调用
build_I_matrix_zero_momentum → Löwdin → build_X_matrix_zero_momentum。
"""

"""
    _subspace_projection_zero(M, n_tuple, lambda_tuple, kappa, Gamma; ...)

零动量粒子投影管线：build_I_matrix_zero_momentum → Löwdin → build_X_matrix_zero_momentum。

约定 R_st(0)=I（零动量方向无定义，取恒等旋转），因此零动量粒子的自旋投影 σ
充当其"螺旋度"标签。返回的 subspace_states 是标准 (n_full, λ_full) 二元组格式，
与 build_V_hel 完全兼容。
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
    I_evals = eigen(Hermitian(I)).values
    Z, C, _ = lowdin_orthogonalize(I)

    # 非对称化自旋基: 所有 (2j+1)^M 个自旋组态，与 build_V_hel 一致
    X = build_X_matrix_zero_momentum(M, j, n_tuple, lambda_tuple, kappa, Gamma,
                                      group_els, irrep_mats, species_type, spin,
                                      etas, n_base, Z, C; canonical_spin=false)

    # 构造标准 (n_full, λ_full) 格式，与 build_V_hel 兼容
    # R_st(0)=I → wigner_D(j,0)=I → 旋转系数退化为 δ_{σ',σ}
    # 零动量粒子 λ 标签取所有自旋投影值（非对称化），与正则极化基底一致
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

# ============ 多物种 _subspace_projection_zero ============

function _subspace_projection_zero(n_tuple::NTuple{N,Momentum},
                                   lambda_tuple::NTuple{N,Float64},
                                   κ_tuple, Gamma::String;
                                   d_total::Momentum,
                                   species::Vector{Int}, particle_types::Vector{Symbol},
                                   spins::Vector{Float64}, etas::Vector{Float64}) where N
    # 规范排序: ZM 粒子优先于 FM 粒子（按物种）
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
    I_evals = eigen(Hermitian(I)).values
    Z, C, _ = lowdin_orthogonalize(I)

    X = build_X_matrix_zero_momentum(zero_counts, sorted_n, sorted_lam, κ_tuple, Gamma,
                                      group_els, irrep_mats, species, particle_types,
                                      spins, etas, n_base, Z, C)

    # 构造标准 (n_full, λ_full) 格式的子空间态
    spin_tuples = _multi_spin_tuples(zero_counts, spins)
    Nfm = N - M_total
    if Nfm > 0
        fin_n = ntuple(i -> sorted_n[M_total + i], Nfm)
        fin_lam = ntuple(i -> sorted_lam[M_total + i], Nfm)
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

    zero_mom = Momentum(0, 0, 0)
    zero_n = ntuple(_ -> zero_mom, M_total)
    subspace_states = []
    for σ in spin_tuples
        for (fn, fl) in fin_states
            full_n = (zero_n..., fn...)
            full_lam = (σ..., fl...)
            push!(subspace_states, (full_n, full_lam))
        end
    end

    return (Z=Z, X=X, subspace_states=subspace_states, I_evals=I_evals)
end

"""
    project_interaction(V::AbstractMatrix, X_left::AbstractMatrix, X_right::AbstractMatrix)
        -> Matrix

相互作用矩阵投影: V^Γ = X_left^† × V × X_right
"""
function project_interaction(V::AbstractMatrix, X_left::AbstractMatrix, X_right::AbstractMatrix)
    return X_left' * V * X_right
end

"""
    project_V(X_left, X_right, subspace_states_left, subspace_states_right,
              per_spin_left, per_spin_right, L, V_can_func, extra_args...)
        -> Matrix

完整投影流程: V_can → V_hel → X_left^† × V_hel × X_right

# 参数
- `X_left, X_right`: `subspace_projection` 或 `build_X_matrix` 返回的 X 矩阵
- `subspace_states_left/right`: 子空间基态列表 (每项 `(n_tuple, λ_tuple)`)
- `per_spin_left/right`: 每粒子自旋
- `L`: 有限体积尺寸
- `V_can_func(n'_tuple, σ'_tuple, n_tuple, σ_tuple, extra_args...)`: 正则极化表象下的 V 矩阵元
- `extra_args...`: 透传给 V_can_func

自动乘 (2π ħc/L)^{d/2} 因子, d = 3(N_α + N_β) - 6。
"""
function project_V(X_left::AbstractMatrix, X_right::AbstractMatrix,
                   subspace_states_left::Vector,
                   subspace_states_right::Vector,
                   per_spin_left::AbstractVector{<:Real},
                   per_spin_right::AbstractVector{<:Real},
                   L::Real,
                   V_can_func::Function, extra_args...)
    # Fourier 因子: (2π ħc/L)^{d/2}, d = 3(N_α + N_β) - 6
    # N_α, N_β 从 subspace_states 中第一个态的动量元组长度获取
    N_α = length(first(subspace_states_left)[1])
    N_β = length(first(subspace_states_right)[1])
    d = 3 * (N_α + N_β) - 6
    fv_factor = (2π * ħc / L)^(d / 2)

    V_hel = build_V_hel(subspace_states_left, subspace_states_right,
                        per_spin_left, per_spin_right,
                        V_can_func, extra_args...)
    return fv_factor * (X_left' * V_hel * X_right)
end
