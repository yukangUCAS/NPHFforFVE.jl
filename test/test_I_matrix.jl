using NPHFforFVE
using LinearAlgebra
using StaticArrays
using Dates

const M = NPHFforFVE.Momentum

# ============ 状态计数估计 ============

const MAX_REPS = 10000  # 代表态数量上限，超过则跳过

# ============ 辅助函数 ============

function _permutation_sign(p::Vector{Int})
    inv_count = 0
    n = length(p)
    for i in 1:n
        for j in i+1:n
            if p[i] > p[j]
                inv_count += 1
            end
        end
    end
    return iseven(inv_count) ? 1 : -1
end

# ======================================================================
# Y matrix (single rep, helicity basis):
#   Y_{(λ',b),(λ,a)} = Σ_{s∈S_N} δ(mom) × δ(λ', s·λ) × R_{ba}(s)
#   其中 (s·λ)_j = λ_{s^{-1}(j)}, δ(mom): rep[p[i]] == rep[i]
# rank(Y) × orbit_sz = 未投影维数
# ======================================================================
function compute_Y_matrix(rep::NTuple{N,M}, kappa::String, spec_type::Symbol, spin::Float64) where N
    n_hel_vals = Int(2 * spin + 1)
    hel_vals = [Float64(-spin + i) for i in 0:(n_hel_vals-1)]
    hel_tuples = vec(collect(Iterators.product(ntuple(_ -> hel_vals, N)...)))
    n_hel_all = length(hel_tuples)
    dim_kappa = NPHFforFVE.get_SN_irrep_dim(N, kappa)
    Y_size = n_hel_all * dim_kappa
    Y = zeros(Float64, Y_size, Y_size)

    hel_to_idx = Dict(t => i for (i, t) in enumerate(hel_tuples))
    all_perms = NPHFforFVE.SN_ELEMENTS[N]
    fermion = (spec_type == :fermion)

    for (s_idx, p) in enumerate(all_perms)
        # 动量 Kronecker deltas
        ok = true
        for i in 1:N
            if rep[p[i]] != rep[i]
                ok = false
                break
            end
        end
        ok || continue

        # 逆置换: pinv[j] = 到达位置 j 的粒子编号
        pinv = Vector{Int}(undef, N)
        for i in 1:N
            pinv[p[i]] = i
        end

        delta_s = fermion ? Float64(_permutation_sign(p)) : 1.0
        R_s = NPHFforFVE.get_SN_irrep_matrix(N, kappa, s_idx)

        for (lam_idx, lam) in enumerate(hel_tuples)
            # λ' = s·λ: λ'_j = λ_{s^{-1}(j)} = lam[pinv[j]]
            lam_p = ntuple(j -> lam[pinv[j]], N)
            lam_p_idx = hel_to_idx[lam_p]

            for a in 1:dim_kappa
                col_idx = (lam_idx - 1) * dim_kappa + a
                for b in 1:dim_kappa
                    row_idx = (lam_p_idx - 1) * dim_kappa + b
                    Y[row_idx, col_idx] += delta_s * R_s[b, a]
                end
            end
        end
    end

    return Y
end

"""
    get_irrep_names(group_name::Symbol, needs_double::Bool)

返回给定群的不可约表示名称列表。
"""
function get_irrep_names(group_name::Symbol, needs_double::Bool)
    if group_name == :Oh
        return NPHFforFVE.OH_IRREP_NAMES
    elseif group_name == :Oh2
        return NPHFforFVE.OH2_IRREP_NAMES
    elseif group_name == :C4v
        return ["A1", "A2", "B1", "B2", "E"]
    elseif group_name == :C4v2
        return ["A1", "A2", "B1", "B2", "E", "G1", "G2"]
    elseif group_name == :C3v
        return ["A1", "A2", "E"]
    elseif group_name == :C3v2
        return ["A1", "A2", "E", "F1", "F2", "G"]
    elseif group_name == :C2v
        return ["A1", "A2", "B1", "B2"]
    elseif group_name == :C2v2
        return ["A1", "A2", "B1", "B2", "G"]
    else
        return String[]
    end
end

"""
    check_I_matrix(I::Matrix{ComplexF64})

检验 I 矩阵：
  - 半正定性：无负本征值
  - 等值性：所有非零本征值为同一正整数

返回 (pass::Bool, message::String, nonzero_vals::Vector{Float64})
"""
function check_I_matrix(I::Matrix{ComplexF64})
    evals = real.(eigen(Hermitian(I)).values)
    nonzero = [v for v in evals if abs(v) > 1e-9]

    # 全零是合法的（该表示不出现在子空间中）
    if isempty(nonzero)
        return true, "OK (zero)", Float64[]
    end

    # 检验负本征值
    for v in nonzero
        if v < -1e-8
            return false, "NEGATIVE eigenvalue $v", nonzero
        end
    end

    # 检验所有非零本征值相等
    ref = nonzero[1]
    for v in nonzero
        if abs(v - ref) > 1e-8
            return false, "UNEQUAL nonzero eigenvalues: $(round.(nonzero, digits=4))", nonzero
        end
    end

    # 检验为正整数
    nearest_int = round(ref)
    if abs(ref - nearest_int) > 1e-8
        return false, "NON-INTEGER eigenvalue: $ref", nonzero
    end

    return true, "OK ($nearest_int)", nonzero
end

# ============ 失败终止 ============

function fail_and_exit(label::String, N, spec_type, spin_val, Ncut, d_total, rep, kappa, hel, Gamma, msg, nonzero)
    _p()
    _p("========================================")
    _p("  FAILURE DETECTED — 脚本终止")
    _p("========================================")
    _p("检验类型: $label")
    _p("N=$N  $spec_type  spin=$spin_val  Ncut=$Ncut  d=$d_total")
    _p("rep = $rep")
    _p("hel = $hel")
    _p("κ   = $kappa")
    _p("Γ   = $Gamma")
    _p("reason: $msg")
    if !isempty(nonzero)
        _p("nonzero evals: $(round.(nonzero, digits=6))")
    end
    _p("========================================")
    exit(1)
end

function fail_dim_and_exit(N, spec_type, spin_val, Ncut, d_total, rep, kappa, i_matrix_sum, expected_dim)
    _p()
    _p("========================================")
    _p("  FAILURE DETECTED — 脚本终止")
    _p("========================================")
    _p("检验类型: Criterion 2 (维度匹配)")
    _p("N=$N  $spec_type  spin=$spin_val  Ncut=$Ncut  d=$d_total")
    _p("rep = $rep")
    _p("κ   = $kappa")
    _p("Σ r×dim(Γ) = $i_matrix_sum")
    _p("expected   = $expected_dim  (l × orbit_sz, l=nonzeros(Y))")
    _p("========================================")
    exit(1)
end

# ============ 测试运行器 ============

function run_test(max_ncut_user::Union{Dict{Int,Int},Nothing}=nothing)
    c1_count    = 0   # criterion 1: I 矩阵检验数
    c2_count    = 0   # criterion 2: (rep, κ) 维数匹配检验数
    skip_states = 0   # 因 n_states > 50000 跳过
    skip_reps   = 0   # 因 n_reps > MAX_REPS 跳过

    default_max_ncut = Dict(2 => 16, 3 => 12, 4 => 8)

    for N in [2, 3, 4]
        t_N_start = time()
        _p()
        _p("========== N=$N ==========")
        _p("[$(Dates.format(now(), "HH:MM:SS"))] 开始")

        max_ncut = max_ncut_user !== nothing ? max_ncut_user[N] : default_max_ncut[N]
        for spec_type in [:boson, :fermion]
            spin_vals = (spec_type == :fermion) ? [1//2] : [0, 1]

            for spin_val in spin_vals
                spin = Float64(spin_val)

                Ncut = max_ncut
                d_total_list = (N == 2) ? [M(0,0,0), M(0,0,1), M(0,1,1), M(1,1,1)] : [M(0,0,0)]
                for d_total in d_total_list
                    # ---- 群信息 ----
                    needs_double = (spec_type == :fermion)
                    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
                    nG = length(group_els)
                    n_base = needs_double ? nG ÷ 2 : nG

                    irrep_names = get_irrep_names(group_name, needs_double)
                    isempty(irrep_names) && continue

                    kappa_names = NPHFforFVE.get_SN_irrep_names(N)

                    # ---- 检查状态数量 ----
                    n_states = count_momentum_states(N, Ncut=Ncut,
                        particle_type=spec_type, d=d_total)
                    n_states > 50000 && begin
                        skip_states += 1
                        _p("  SKIP N=$N $spec_type spin=$(spin_val) d=$d_total: $n_states states (>50000)")
                        continue
                    end

                    # ---- 生成代表态 ----
                    reps = find_representatives(N, Ncut=Ncut,
                        particle_type=spec_type, d=d_total)

                    n_reps = length(reps)
                    n_reps > MAX_REPS && begin
                        skip_reps += 1
                        _p("  SKIP N=$N $spec_type spin=$(spin_val) d=$d_total: $n_reps reps (>$MAX_REPS)")
                        continue
                    end

                    elapsed = round(time() - t_N_start, digits=1)
                    _p("N=$N $spec_type spin=$(spin_val) d=$d_total: $n_reps reps, $n_states states  [$(elapsed)s]")

                    # ---- 遍历每个代表态 ----
                    etas = [1.0]

                    for rep in reps
                        # ZM 粒子数
                        M_zm = count(n -> n == M(0,0,0), rep)
                        zm_mask = [n == M(0,0,0) for n in rep]

                        # 轨道
                        orbit_vec = group_orbit(rep, d=d_total,
                            particle_type=spec_type, species=[N])
                        orbit_sz = length(orbit_vec)

                        # 生成螺旋度组态
                        local hel_configs
                        if spin == 0
                            hel_configs = [ntuple(_ -> 0.0, N)]
                        elseif M_zm == 0
                            h = helicity_representatives(rep, species=[N],
                                particle_types=[spec_type], spins=[spin_val], d=d_total)
                            hel_configs = [Tuple(Float64.(x)) for x in h]
                        else
                            zm = [n == M(0,0,0) for n in rep]
                            fm_idxs = findall(x -> !x, zm)
                            N_fm = length(fm_idxs)
                            if N_fm == 0
                                hel_configs = [ntuple(_ -> 0.0, N)]
                            else
                                fm_rep = Tuple(rep[i] for i in fm_idxs)
                                h_fm = helicity_representatives(fm_rep, species=[N_fm],
                                    particle_types=[spec_type], spins=[spin_val], d=d_total)
                                fm_hels = [Tuple(Float64.(x)) for x in h_fm]
                                hel_configs = [ntuple(i -> zm[i] ? 0.0 : fm_hel[findfirst(==(i), fm_idxs)], N)
                                              for fm_hel in fm_hels]
                            end
                        end

                        # ---- 对每个 κ ----
                        for kappa in kappa_names

                            # Line 10: 未投影维数 = rank(Y_old) × orbit_sz
                            Y = compute_Y_matrix(rep, kappa, spec_type, spin)
                            y_evals = real.(eigen(Hermitian(Y)).values)
                            expected_dim = count(v -> abs(v) > 1e-9, y_evals) * orbit_sz

                            # Line 12: 从 I 矩阵累加 Σ_{hel,Γ} r × dim(Γ)
                            i_matrix_sum = 0
                            # 缓存打印所需信息，避免二次构造 I 矩阵
                            local print_data = Tuple{typeof(hel_configs[1]), String, Int, Int, Int}[]

                            for hel in hel_configs
                                for Gamma in irrep_names
                                    irrep_mats = try
                                        irrep_matrices(Gamma; group=group_name)
                                    catch
                                        nothing
                                    end
                                    irrep_mats === nothing && continue

                                    # ZM 重排（与 build_I_matrix_zero_momentum 一致，ZM 粒子在前）
                                    local rep_use, hel_use
                                    if M_zm > 0 && spin != 0
                                        zm_reorder = vcat(findall(zm_mask), findall(x -> !x, zm_mask))
                                        rep_use = ntuple(i -> rep[zm_reorder[i]], N)
                                        hel_use = ntuple(i -> hel[zm_reorder[i]], N)
                                    else
                                        rep_use = rep
                                        hel_use = hel
                                    end

                                    local I
                                    try
                                        if M_zm > 0 && spin != 0
                                            I = NPHFforFVE.build_I_matrix_zero_momentum(
                                                M_zm, Rational{Int}(spin_val), rep_use, hel_use, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas, n_base)
                                        else
                                            I = NPHFforFVE.build_I_matrix(rep_use, hel_use, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas, n_base)
                                        end
                                    catch e
                                        fail_and_exit("build_I_matrix EXCEPTION",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, kappa, hel, Gamma, "EXCEPTION: $e", Float64[])
                                    end

                                    # ---- Criterion 1 ----
                                    c1_count += 1
                                    ok, msg, nonzero = check_I_matrix(I)
                                    if !ok
                                        fail_and_exit("Criterion 1 (半正定性)",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, kappa, hel, Gamma, msg, nonzero)
                                    end

                                    # r = 非零本征值个数 = 重数
                                    r = length(nonzero)
                                    dim_Gamma = size(irrep_mats[1], 1)
                                    i_matrix_sum += r * dim_Gamma
                                    push!(print_data, (hel, string(Gamma), size(I,1), r, dim_Gamma))
                                end  # Gamma
                            end  # hel

                            # ---- Criterion 2 ----
                            c2_count += 1
                            matched = abs(i_matrix_sum - expected_dim) < 1e-8

                            # 维数打印: ZM 情况全部打印, 非 ZM 只打前几个
                            local do_print
                            if M_zm > 0 && spin != 0
                                do_print = true   # ZM 情况全部打印
                            else
                                do_print = c2_count <= 3  # 非 ZM 前 3 个
                            end
                            if do_print
                                _p("  --- dims: rep=$rep  κ=$kappa  orbit_sz=$orbit_sz  M_zm=$M_zm ---")
                                _p("    Y: $(size(Y,1))x$(size(Y,2))  rank(Y)=$(expected_dim)")
                                _p("    hel_configs: $(length(hel_configs))")
                                _p("    I_matrices:")
                                for (hel, Gamma, sz, r, dim_Gamma) in print_data
                                    _p("      hel=$hel  Γ=$Gamma  I:$(sz)x$(sz)  r=$r  dimΓ=$dim_Gamma  r*dimΓ=$(r*dim_Gamma)")
                                end
                                _p("    Σ r*dimΓ = $i_matrix_sum  |  expected = $expected_dim  $(matched ? "OK" : "MISMATCH")")
                            end

                            if !matched
                                fail_dim_and_exit(N, spec_type, spin_val, Ncut, d_total,
                                                  rep, kappa, i_matrix_sum, expected_dim)
                            end
                        end  # kappa
                    end  # rep
                end  # d_total
            end  # spin_val
        end      # spec_type

        t_N_elapsed = round(time() - t_N_start, digits=1)
        _p("[$(Dates.format(now(), "HH:MM:SS"))] N=$N 完成, 耗时 $(t_N_elapsed)s, 累计 I矩阵:$(c1_count) 维度:$(c2_count)")
    end          # N

    # ======== 全部通过 ========
    _p()
    _p("========================================")
    _p("         最终检验成果")
    _p("========================================")
    _p("参数范围:")
    n2  = max_ncut_user !== nothing ? max_ncut_user[2] : 16
    n3  = max_ncut_user !== nothing ? max_ncut_user[3] : 12
    n4  = max_ncut_user !== nothing ? max_ncut_user[4] : 8
    _p("  N=2 (Ncut=$n2), N=3 (Ncut=$n3), N=4 (Ncut=$n4)")
    _p("  boson (s=0,1) / fermion (s=1/2)")
    _p("  总动量: D000, D001, D011, D111")
    _p()
    _p("Criterion 1 (半正定性):")
    _p("  检验 I 矩阵数: $c1_count, 全部通过")
    _p()
    _p("Criterion 2 (维数匹配):")
    _p("  检验 (rep,κ) 数: $c2_count, 全部通过")
    _p()
    if skip_states > 0 || skip_reps > 0
        _p("跳过:")
        skip_states > 0 && _p("  因 n_states > 50000: $skip_states")
        skip_reps   > 0 && _p("  因 n_reps   > $MAX_REPS: $skip_reps")
        _p()
    end
    _p("所有测试通过。")
    _p()
end

# ============ 多物种测试 ============

function run_multi_species_test(max_ncut_user::Union{Dict{Int,Int},Nothing}=nothing)
    c1_count    = 0
    c2_count    = 0
    skip_states = 0
    skip_reps   = 0

    default_max_ncut = Dict(2 => 8, 3 => 5, 4 => 6)

    # 测试系统定义: (label, N, species, particle_types, spin_vals)
    systems = [
        # N=2, species=[1,1]
        ("N=2_F-F_[1,1]",    2, [1,1], [:fermion, :fermion], [1//2, 1//2]),
        ("N=2_B-F_[1,1]",    2, [1,1], [:boson,   :fermion], [0,    1//2]),
        ("N=2_B-B_01_[1,1]", 2, [1,1], [:boson,   :boson],   [0,    1]),
        ("N=2_B-B_11_[1,1]", 2, [1,1], [:boson,   :boson],   [1,    1]),
        # N=3, species=[2,1]
        ("N=3_F-F_[2,1]",    3, [2,1], [:fermion, :fermion], [1//2, 1//2]),
        ("N=3_B-F_[2,1]",    3, [2,1], [:boson,   :fermion], [0,    1//2]),
        ("N=3_F-B_[2,1]",    3, [2,1], [:fermion, :boson],   [1//2, 0]),
        ("N=3_B-B_01_[2,1]", 3, [2,1], [:boson,   :boson],   [0,    1]),
        ("N=3_B-B_10_[2,1]", 3, [2,1], [:boson,   :boson],   [1,    0]),
        # N=4, species=[2,2]
        ("N=4_F-F_[2,2]",    4, [2,2], [:fermion, :fermion], [1//2, 1//2]),
        ("N=4_B-B_[2,2]",    4, [2,2], [:boson,   :boson],   [1,    1]),
        # N=4, species=[3,1]
        ("N=4_F-F_[3,1]",    4, [3,1], [:fermion, :fermion], [1//2, 1//2]),
        ("N=4_B-B_[3,1]",    4, [3,1], [:boson,   :boson],   [1,    1]),
    ]

    _p()
    _p("========================================")
    _p("  多物种测试")
    _p("========================================")

    for sys in systems
        label, N, species, particle_types, spin_vals = sys
        spins = Float64.(spin_vals)
        K = length(species)
        etas = ones(Float64, K)

        max_ncut = max_ncut_user !== nothing ? max_ncut_user[N] : default_max_ncut[N]
        isempty(NPHFforFVE.get_SN_irrep_names(maximum(species))) && continue

        _p()
        _p("----- $label  Ncut=$max_ncut -----")

        Ncut = max_ncut
        d_total_list = (N == 2) ? [M(0,0,0), M(0,0,1), M(0,1,1), M(1,1,1)] : [M(0,0,0)]
        for d_total in d_total_list
            # 群信息
            needs_double = any(pt == :fermion for pt in particle_types)
            group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
            nG = length(group_els)
            n_base = needs_double ? nG ÷ 2 : nG

            irrep_names = get_irrep_names(group_name, needs_double)
            isempty(irrep_names) && continue

            # κ 组合: 各物种 SN irrep 的直积
            kappa_options = [NPHFforFVE.get_SN_irrep_names(species[k]) for k in 1:K]
            kappa_combos = vec(collect(Iterators.product(kappa_options...)))

            # 状态数检查
            n_states = count_momentum_states(N, Ncut=Ncut,
                species=species, particle_types=particle_types)
            n_states > 50000 && begin
                skip_states += 1
                _p("  SKIP $label d=$d_total: $n_states states (>50000)")
                continue
            end

            # 代表态
            reps = find_representatives(N, Ncut=Ncut,
                species=species, particle_types=particle_types)

            n_reps = length(reps)
            n_reps > MAX_REPS && begin
                skip_reps += 1
                _p("  SKIP $label d=$d_total: $n_reps reps (>$MAX_REPS)")
                continue
            end

            _p("  $label d=$d_total: $n_reps reps, $n_states states")

            # 粒子→物种映射
            spec_of = Vector{Int}(undef, N)
            off = 1
            for (k, sz) in enumerate(species)
                for i in off:(off+sz-1)
                    spec_of[i] = k
                end
                off += sz
            end

            for rep in reps
                # 轨道大小
                orbit_sz = length(group_orbit(rep, d=d_total,
                    species=species, particle_types=particle_types))

                # ZM 信息
                zm_mask = [n == M(0,0,0) for n in rep]
                M_zm = count(zm_mask)

                # 各物种 ZM 粒子数
                zero_counts = zeros(Int, K)
                for i in 1:N
                    if zm_mask[i]
                        zero_counts[spec_of[i]] += 1
                    end
                end

                # 螺旋度组态
                local hel_configs
                if M_zm == 0
                    h = helicity_representatives(rep; species=species,
                        particle_types=particle_types, spins=spins, d=d_total)
                    hel_configs = [Tuple(Float64.(x)) for x in h]
                elseif M_zm == N
                    hel_configs = [ntuple(_ -> 0.0, N)]
                else
                    # 混合 ZM+FM: 尝试全系统调用，失败则提取 FM
                    h = try
                        helicity_representatives(rep; species=species,
                            particle_types=particle_types, spins=spins, d=d_total)
                    catch
                        Vector{Vector{Float64}}()
                    end
                    if !isempty(h)
                        hel_configs = [Tuple(Float64.(x)) for x in h]
                    else
                        fm_idxs = findall(x -> !x, zm_mask)
                        fm_rep = Tuple(rep[i] for i in fm_idxs)
                        fm_spec_of = spec_of[fm_idxs]

                        # 按物种分组 FM 粒子
                        fm_species_vec = Int[]
                        fm_ptypes_vec = Symbol[]
                        fm_spins_vec = Float64[]
                        curr_s = fm_spec_of[1]
                        cnt = 1
                        for i in 2:length(fm_spec_of)
                            if fm_spec_of[i] == curr_s
                                cnt += 1
                            else
                                push!(fm_species_vec, cnt)
                                push!(fm_ptypes_vec, particle_types[curr_s])
                                push!(fm_spins_vec, spins[curr_s])
                                curr_s = fm_spec_of[i]
                                cnt = 1
                            end
                        end
                        push!(fm_species_vec, cnt)
                        push!(fm_ptypes_vec, particle_types[curr_s])
                        push!(fm_spins_vec, spins[curr_s])

                        h_fm = helicity_representatives(fm_rep; species=fm_species_vec,
                            particle_types=fm_ptypes_vec, spins=fm_spins_vec, d=d_total)
                        fm_hels = [Tuple(Float64.(x)) for x in h_fm]
                        hel_configs = [ntuple(i -> zm_mask[i] ? 0.0 : fm_hel[findfirst(==(i), fm_idxs)], N)
                                      for fm_hel in fm_hels]
                    end
                end

                isempty(hel_configs) && continue

                # 对每个 κ 组合
                for κ_tuple in kappa_combos

                    # Criterion 2: l_total = ∏_k l_k
                    l_total = 1
                    off_k = 1
                    for k in 1:K
                        Nk = species[k]
                        sub_rep = ntuple(ii -> rep[off_k+ii-1], Nk)
                        Yk = compute_Y_matrix(sub_rep, κ_tuple[k], particle_types[k], spins[k])
                        yk_evals = real.(eigen(Hermitian(Yk)).values)
                        l_total *= count(v -> abs(v) > 1e-9, yk_evals)
                        off_k += Nk
                    end
                    expected_dim = l_total * orbit_sz

                    i_matrix_sum = 0

                    for hel in hel_configs
                    for Gamma in irrep_names
                        irrep_mats = try
                            irrep_matrices(Gamma; group=group_name)
                        catch
                            nothing
                        end
                        irrep_mats === nothing && continue

                        use_zm = any(zero_counts .> 0) && any(s -> s != 0.0, spins)

                        # ZM 重排（每物种块内 ZM 粒子在前，与 build_I_matrix_zero_momentum 一致）
                        local rep_use, hel_use
                        if use_zm
                            zm_reorder = Int[]
                            off = 0
                            for k in 1:K
                                Nk = species[k]
                                block_zm = Int[]; block_fm = Int[]
                                for i in off+1:off+Nk
                                    if rep[i] == M(0,0,0)
                                        push!(block_zm, i)
                                    else
                                        push!(block_fm, i)
                                    end
                                end
                                append!(zm_reorder, block_zm); append!(zm_reorder, block_fm)
                                off += Nk
                            end
                            rep_use = ntuple(i -> rep[zm_reorder[i]], N)
                            hel_use = ntuple(i -> hel[zm_reorder[i]], N)
                        else
                            rep_use = rep
                            hel_use = hel
                        end

                        local I
                        try
                            if use_zm
                                I = NPHFforFVE.build_I_matrix_zero_momentum(
                                    zero_counts, rep_use, hel_use, κ_tuple, Gamma,
                                    group_els, irrep_mats, species, particle_types,
                                    spins, etas, n_base)
                            else
                                I = NPHFforFVE.build_I_matrix(rep_use, hel_use, κ_tuple, Gamma,
                                    group_els, irrep_mats, species, particle_types,
                                    spins, etas, n_base)
                            end
                        catch e
                            fail_and_exit("build_I_matrix EXCEPTION",
                                N, :multi, spin_vals, Ncut, d_total,
                                rep, κ_tuple, hel, Gamma, "EXCEPTION: $e", Float64[])
                        end

                        # Criterion 1
                        c1_count += 1
                        ok, msg, nonzero = check_I_matrix(I)
                        if !ok
                            fail_and_exit("Criterion 1 (半正定性)",
                                N, :multi, spin_vals, Ncut, d_total,
                                rep, κ_tuple, hel, Gamma, msg, nonzero)
                        end

                        r = length(nonzero)
                        dim_Gamma = size(irrep_mats[1], 1)
                        i_matrix_sum += r * dim_Gamma
                    end  # Gamma
                    end  # hel

                    # Criterion 2
                    c2_count += 1
                    if abs(i_matrix_sum - expected_dim) > 1e-8
                        fail_dim_and_exit(N, :multi, spin_vals, Ncut, d_total,
                                          rep, κ_tuple, i_matrix_sum, expected_dim)
                    end
                end  # κ_tuple
            end  # rep
        end  # d_total
    end  # system

    _p()
    _p("========================================")
    _p("  多物种测试完成")
    _p("========================================")
    _p("Criterion 1 (半正定性): $c1_count 通过")
    _p("Criterion 2 (维数匹配): $c2_count 通过")
    if skip_states > 0 || skip_reps > 0
        _p("跳过: states=$skip_states, reps=$skip_reps")
    end
    _p()
    return (c1_count, c2_count, skip_states, skip_reps)
end

# ============ 入口 ============

# 同时写入终端和文件的 print 函数
const _LOG_IO = Ref{IO}(devnull)
_p(args...; kwargs...) = (println(args...; kwargs...); println(_LOG_IO[], args...; kwargs...); flush(_LOG_IO[]))

const OUTPUT_FILE = "test_I_matrix_output.txt"

# 用法: julia test/test_I_matrix.jl [Ncut]           → 所有 N 相同
#       julia test/test_I_matrix.jl [Ncut2] [Ncut3] [Ncut4]  → 分别指定 N=2,3,4 的 Ncut
function parse_ncut_args()
    if isempty(ARGS)
        return nothing  # 使用默认值
    end

    vals = Int[]
    for arg in ARGS
        n = tryparse(Int, arg)
        n === nothing && error("无法解析 '$arg' 为整数")
        n <= 1 && error("Ncut 必须 ≥ 2")
        push!(vals, n)
    end

    if length(vals) == 1
        println("Ncut = $vals  (所有 N 相同)")
        return Dict(2 => vals[1], 3 => vals[1], 4 => vals[1])
    elseif length(vals) == 3
        println("Ncut: N=2→$(vals[1]), N=3→$(vals[2]), N=4→$(vals[3])")
        return Dict(2 => vals[1], 3 => vals[2], 4 => vals[3])
    else
        error("请提供 1 个或 3 个 Ncut 值 (对应 N=2,3,4), 收到了 $(length(vals)) 个")
    end
end

max_ncut_user = parse_ncut_args()

println("输出同时写入: $OUTPUT_FILE")
println()

_LOG_IO[] = open(OUTPUT_FILE, "w")
try
    @time run_test(max_ncut_user)
    println()
    @time run_multi_species_test(max_ncut_user)
finally
    close(_LOG_IO[])
end

println("结果已保存至: $OUTPUT_FILE")
