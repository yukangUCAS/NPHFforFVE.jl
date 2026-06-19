using NPHFforFVE
using LinearAlgebra
using StaticArrays
using Dates

const M = NPHFforFVE.Momentum

# ============ 辅助函数 ============

"""
    check_X_matrix(X::Matrix{ComplexF64}, S::Matrix{Float64}) -> (Bool, Float64, String)

检验 X† S X = I。
返回 (pass, max_deviation, message)。
"""
function check_X_matrix(X::Matrix{ComplexF64}, S::Matrix{Float64})
    n_r = size(X, 2)
    if n_r == 0
        return true, 0.0, "OK (empty X)"
    end

    prod = X' * S * X
    expected = Matrix{Float64}(I, n_r, n_r)
    diff = prod - expected
    maxdev = maximum(abs.(diff))

    if maxdev < 1e-10
        return true, maxdev, "OK"
    elseif maxdev < 1e-6
        return true, maxdev, "OK (tol 1e-6)"
    else
        return false, maxdev, "FAIL: max|X'SX - I| = $maxdev"
    end
end

"""
    check_S_psd(S::Matrix{Float64}) -> (Bool, Float64, String)

检验 S 矩阵半正定性。
返回 (pass, min_eigenvalue, message)。
"""
function check_S_psd(S::Matrix{Float64})
    evals = eigen(Symmetric(S)).values
    min_ev = minimum(evals)
    if min_ev < -1e-10
        return false, min_ev, "FAIL: S 含负特征值 λ_min = $min_ev"
    end
    n_pos = count(x -> x > 1e-10, evals)
    return true, min_ev, "OK (非零特征值: $n_pos)"
end

# ============ 失败终止 ============

function fail_and_exit(label::String, N, spec_type, spin_val, Ncut, d_total,
                        rep, hel, kappa, Gamma, msg)
    println()
    println("========================================")
    println("  FAILURE DETECTED — 脚本终止")
    println("========================================")
    _p("检验类型: $label")
    _p("N=$N  $spec_type  spin=$spin_val  Ncut=$Ncut  d=$d_total")
    _p("rep = $rep")
    _p("hel = $hel")
    _p("κ   = $kappa")
    _p("Γ   = $Gamma")
    _p("reason: $msg")
    _p("========================================")
    exit(1)
end

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

# ============ 单物种测试 ============

function run_test(max_ncut_user::Union{Dict{Int,Int},Nothing}=nothing)
    n_tested   = 0
    n_ok       = 0
    n_empty    = 0
    skip_states = 0
    skip_reps   = 0
    max_dev     = 0.0

    default_max_ncut = Dict(2 => 8, 3 => 5, 4 => 6)

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
                for d_total in [M(0,0,0), M(0,0,1), M(0,1,1), M(1,1,1)]
                    needs_double = (spec_type == :fermion)
                    group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
                    nG = length(group_els)
                    n_base = needs_double ? nG ÷ 2 : nG

                    irrep_names = get_irrep_names(group_name, needs_double)
                    isempty(irrep_names) && continue

                    kappa_names = NPHFforFVE.get_SN_irrep_names(N)

                    n_total_states = count_momentum_states(N, Ncut=Ncut,
                        particle_type=spec_type, d=d_total)
                    n_total_states > 50000 && begin
                        skip_states += 1
                        continue
                    end

                    reps = find_representatives(N, Ncut=Ncut,
                        particle_type=spec_type, d=d_total)
                    length(reps) > 10000 && begin
                        skip_reps += 1
                        continue
                    end

                    etas = [1.0]

                    for rep in reps
                        # ZM 检测
                        zm_mask = [n == M(0,0,0) for n in rep]
                        M_zm = count(zm_mask)

                        # 螺旋度组态
                        local hel_configs
                        if spin == 0
                            hel_configs = [ntuple(_ -> 0.0, N)]
                        elseif M_zm == 0
                            h = helicity_representatives(rep, species=[N],
                                particle_types=[spec_type], spins=[spin_val], d=d_total)
                            hel_configs = [Tuple(Float64.(x)) for x in h]
                        elseif M_zm == N
                            hel_configs = [ntuple(_ -> 0.0, N)]
                        else
                            # 混合 ZM+FM: 提取 FM 部分求螺旋度
                            h = try
                                helicity_representatives(rep, species=[N],
                                    particle_types=[spec_type], spins=[spin_val], d=d_total)
                            catch
                                Vector{Vector{Float64}}()
                            end
                            if !isempty(h)
                                hel_configs = [Tuple(Float64.(x)) for x in h]
                            else
                                fm_idxs = findall(x -> !x, zm_mask)
                                fm_rep = Tuple(rep[i] for i in fm_idxs)
                                h_fm = helicity_representatives(fm_rep, species=[length(fm_idxs)],
                                    particle_types=[spec_type], spins=[spin_val], d=d_total)
                                fm_hels = [Tuple(Float64.(x)) for x in h_fm]
                                hel_configs = [ntuple(i -> zm_mask[i] ? 0.0 : fm_hel[findfirst(==(i), fm_idxs)], N)
                                              for fm_hel in fm_hels]
                            end
                        end

                        for hel in hel_configs
                            for kappa in kappa_names

                                # ZM 路由: 有 ZM 粒子且 spin≠0 时使用 ZM pipeline
                                use_zm = M_zm > 0 && spin != 0.0
                                local j_zm = use_zm ? Rational{Int}(Int(2*spin), 2) : 0//1

                                for Gamma in irrep_names
                                    irrep_mats = try
                                        irrep_matrices(Gamma; group=group_name)
                                    catch
                                        nothing
                                    end
                                    irrep_mats === nothing && continue

                                    # ZM 重排（与 build_X_matrix_zero_momentum 一致，ZM 粒子在前）
                                    if use_zm
                                        zm_reorder = vcat(findall(zm_mask), findall(x -> !x, zm_mask))
                                        rep_zm = ntuple(i -> rep[zm_reorder[i]], N)
                                        hel_zm = ntuple(i -> hel[zm_reorder[i]], N)
                                    else
                                        rep_zm = rep
                                        hel_zm = hel
                                    end

                                    # ---- I 矩阵 ----
                                    local I
                                    try
                                        if use_zm
                                            I = NPHFforFVE.build_I_matrix_zero_momentum(
                                                M_zm, j_zm, rep_zm, hel_zm, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas, n_base)
                                        else
                                            I = NPHFforFVE.build_I_matrix(rep_zm, hel_zm, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas, n_base)
                                        end
                                    catch e
                                        fail_and_exit("build_I_matrix EXCEPTION",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, hel, kappa, Gamma, "EXCEPTION: $e")
                                    end

                                    # Löwdin 正交化
                                    Z, C, _ = NPHFforFVE.lowdin_orthogonalize(I)

                                    # ---- X 矩阵 ----
                                    local X
                                    try
                                        if use_zm
                                            X = NPHFforFVE.build_X_matrix_zero_momentum(
                                                M_zm, j_zm, rep_zm, hel_zm, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas,
                                                n_base, Z, C)
                                        else
                                            X = NPHFforFVE.build_X_matrix(rep_zm, hel_zm, kappa, Gamma,
                                                group_els, irrep_mats, spec_type, spin, etas,
                                                n_base, Z, C)
                                        end
                                    catch e
                                        fail_and_exit("build_X_matrix EXCEPTION",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, hel, kappa, Gamma, "EXCEPTION: $e")
                                    end

                                    # ---- S 矩阵 ----
                                    local S
                                    if use_zm
                                        spin_tuples = NPHFforFVE._spin_tuples(j_zm, M_zm)
                                        fin_n = ntuple(i -> rep_zm[M_zm + i], N - M_zm)
                                        fin_lam = ntuple(i -> hel_zm[M_zm + i], N - M_zm)
                                        fin_states = NPHFforFVE._collect_fin_subspace_states(
                                            fin_n, fin_lam, group_els, spin, etas[1], n_base)
                                        S = NPHFforFVE.build_S_matrix_zero_momentum(
                                            M_zm, spin_tuples, fin_states, N, kappa, spec_type)
                                    else
                                        subspace_states = NPHFforFVE._collect_subspace_states(
                                            rep, hel, group_els, spin, etas[1], n_base)
                                        S = NPHFforFVE.build_S_matrix(subspace_states, N, kappa, spec_type)
                                    end

                                    # ---- 检验 S 半正定性 ----
                                    s_ok, s_min_ev, s_msg = check_S_psd(S)
                                    if !s_ok
                                        fail_and_exit("S 半正定性失败",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, hel, kappa, Gamma, s_msg)
                                    end

                                    # ---- 检验 X† S X = I ----
                                    n_tested += 1
                                    ok, dev, msg = check_X_matrix(X, S)
                                    if dev > max_dev
                                        max_dev = dev
                                    end

                                    if !ok
                                        fail_and_exit("X† S X ≠ I",
                                            N, spec_type, spin_val, Ncut, d_total,
                                            rep, hel, kappa, Gamma, msg)
                                    end

                                    if size(X, 2) == 0
                                        n_empty += 1
                                    else
                                        n_ok += 1
                                    end
                                end  # Gamma
                            end  # kappa
                        end  # hel
                    end  # rep
                end  # d_total
            end  # spin_val
        end  # spec_type

        t_elapsed = round(time() - t_N_start, digits=1)
        _p("[$(Dates.format(now(), "HH:MM:SS"))] N=$N 完成, 耗时 $(t_elapsed)s, 累计检验:$(n_tested) 非空:$(n_ok) 空:$(n_empty)")
    end  # N

    # ======== 全部通过 ========
    _p()
    _p("========================================")
    _p("         单物种最终检验成果")
    _p("========================================")
    _p("参数范围:")
    n2 = max_ncut_user !== nothing ? max_ncut_user[2] : 8
    n3 = max_ncut_user !== nothing ? max_ncut_user[3] : 5
    n4 = max_ncut_user !== nothing ? max_ncut_user[4] : 6
    _p("  N=2 (Ncut≤$n2), N=3 (Ncut≤$n3), N=4 (Ncut≤$n4)")
    _p("  boson (s=0,1) / fermion (s=1/2)")
    _p("  总动量: D000, D001, D011, D111")
    _p()
    _p("检验 X 矩阵数: $n_tested  (非空: $n_ok, 空: $n_empty)")
    _p("最大偏差 max|X'SX - I|: $(max_dev)")
    _p()
    _p("检验标准 X† S X = I:  全部通过")
    _p()
    if skip_states > 0 || skip_reps > 0
        _p("跳过:")
        skip_states > 0 && _p("  因 n_states > 50000: $skip_states")
        skip_reps   > 0 && _p("  因 n_reps   > 10000: $skip_reps")
        _p()
    end
    _p("所有测试通过。")
    _p()
    return (n_tested, n_ok, n_empty, max_dev, skip_states, skip_reps)
end

# ============ 多物种测试 ============

function run_multi_species_X_test(max_ncut_user::Union{Dict{Int,Int},Nothing}=nothing)
    n_tested   = 0
    n_ok       = 0
    n_empty    = 0
    skip_states = 0
    skip_reps   = 0
    max_dev     = 0.0

    default_max_ncut = Dict(2 => 8, 3 => 5, 4 => 6)

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
    _p("  多物种 X 矩阵测试")
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
            needs_double = any(pt == :fermion for pt in particle_types)
            group_els, group_name = group_for_momentum(d_total; double_cover=needs_double)
            nG = length(group_els)
            n_base = needs_double ? nG ÷ 2 : nG

            irrep_names = get_irrep_names(group_name, needs_double)
            isempty(irrep_names) && continue

            kappa_options = [NPHFforFVE.get_SN_irrep_names(species[k]) for k in 1:K]
            kappa_combos = vec(collect(Iterators.product(kappa_options...)))

            n_states = count_momentum_states(N, Ncut=Ncut,
                species=species, particle_types=particle_types)
            n_states > 50000 && begin
                skip_states += 1
                _p("  SKIP $label d=$d_total: $n_states states (>50000)")
                continue
            end

            reps = find_representatives(N, Ncut=Ncut,
                species=species, particle_types=particle_types)
            length(reps) > 10000 && begin
                skip_reps += 1
                _p("  SKIP $label d=$d_total: $(length(reps)) reps (>10000)")
                continue
            end

            _p("  $label d=$d_total: $(length(reps)) reps, $n_states states")

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
                zm_mask = [n == M(0,0,0) for n in rep]
                M_zm = count(zm_mask)

                zero_counts = zeros(Int, K)
                for i in 1:N
                    if zm_mask[i]
                        zero_counts[spec_of[i]] += 1
                    end
                end

                # 螺旋度组态 (与 test_I_matrix.jl 一致)
                local hel_configs
                if M_zm == 0
                    h = helicity_representatives(rep; species=species,
                        particle_types=particle_types, spins=spins, d=d_total)
                    hel_configs = [Tuple(Float64.(x)) for x in h]
                elseif M_zm == N
                    hel_configs = [ntuple(_ -> 0.0, N)]
                else
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

                for κ_tuple in kappa_combos
                    for hel in hel_configs
                    for Gamma in irrep_names
                        irrep_mats = try
                            irrep_matrices(Gamma; group=group_name)
                        catch
                            nothing
                        end
                        irrep_mats === nothing && continue

                        # ZM 路由
                        use_zm = any(zero_counts .> 0) && any(s -> s != 0.0, spins)

                        # ZM 重排（每物种块内 ZM 粒子在前，与 build_*_zero_momentum 一致）
                        local rep_use, hel_use
                        if use_zm
                            zm_reorder = Int[]
                            off_zm = 0
                            for k in 1:K
                                Nk = species[k]
                                block_zm = Int[]; block_fm = Int[]
                                for i in off_zm+1:off_zm+Nk
                                    if rep[i] == M(0,0,0)
                                        push!(block_zm, i)
                                    else
                                        push!(block_fm, i)
                                    end
                                end
                                append!(zm_reorder, block_zm); append!(zm_reorder, block_fm)
                                off_zm += Nk
                            end
                            rep_use = ntuple(i -> rep[zm_reorder[i]], N)
                            hel_use = ntuple(i -> hel[zm_reorder[i]], N)
                        else
                            rep_use = rep
                            hel_use = hel
                        end

                        # ---- I 矩阵 ----
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
                                rep_use, hel_use, κ_tuple, Gamma, "EXCEPTION: $e")
                        end

                        # Löwdin 正交化
                        Z, C, _ = NPHFforFVE.lowdin_orthogonalize(I)

                        # ---- X 矩阵 ----
                        local X
                        try
                            if use_zm
                                X = NPHFforFVE.build_X_matrix_zero_momentum(
                                    zero_counts, rep_use, hel_use, κ_tuple, Gamma,
                                    group_els, irrep_mats, species, particle_types,
                                    spins, etas, n_base, Z, C)
                            else
                                X = NPHFforFVE.build_X_matrix(rep_use, hel_use, κ_tuple, Gamma,
                                    group_els, irrep_mats, species, particle_types,
                                    spins, etas, n_base, Z, C)
                            end
                        catch e
                            fail_and_exit("build_X_matrix EXCEPTION",
                                N, :multi, spin_vals, Ncut, d_total,
                                rep_use, hel_use, κ_tuple, Gamma, "EXCEPTION: $e")
                        end

                        # ---- S 矩阵 ----
                        local S
                        if use_zm
                            spin_tuples = NPHFforFVE._multi_spin_tuples(zero_counts, spins)
                            fin_n_list = Momentum[n for n in rep_use if !iszero(n)]
                            fin_lam_list = Float64[hel_use[i] for i in 1:N if !iszero(rep_use[i])]
                            fin_n = Tuple(fin_n_list)
                            fin_lam = Tuple(fin_lam_list)
                            fin_species = Int[species[k] - zero_counts[k] for k in 1:K]
                            fin_species = Int[n for n in fin_species if n > 0]
                            fin_spins = Float64[spins[k] for k in 1:K if species[k] > zero_counts[k]]
                            fin_etas = Float64[etas[k] for k in 1:K if species[k] > zero_counts[k]]
                            fin_states = NPHFforFVE._collect_fin_subspace_states(
                                fin_n, fin_lam, group_els, fin_species, fin_spins, fin_etas, n_base)
                            S = NPHFforFVE.build_S_matrix_zero_momentum(
                                zero_counts, spin_tuples, fin_states, species, κ_tuple, particle_types)
                        else
                            subspace_states = NPHFforFVE._collect_subspace_states(
                                rep_use, hel_use, group_els, species, spins, etas, n_base)
                            S = NPHFforFVE.build_S_matrix(subspace_states, species, κ_tuple, particle_types)
                        end

                        # ---- 检验 S 半正定性 ----
                        s_ok, s_min_ev, s_msg = check_S_psd(S)
                        if !s_ok
                            fail_and_exit("S 半正定性失败",
                                N, :multi, spin_vals, Ncut, d_total,
                                rep, hel, κ_tuple, Gamma, s_msg)
                        end

                        # ---- 检验 X† S X = I ----
                        n_tested += 1
                        ok, dev, msg = check_X_matrix(X, S)
                        if dev > max_dev
                            max_dev = dev
                        end

                        if !ok
                            fail_and_exit("X† S X ≠ I",
                                N, :multi, spin_vals, Ncut, d_total,
                                rep, hel, κ_tuple, Gamma, msg)
                        end

                        if size(X, 2) == 0
                            n_empty += 1
                        else
                            n_ok += 1
                        end
                    end  # Gamma
                    end  # hel
                end  # κ_tuple
            end  # rep
        end  # d_total
    end  # system

    _p()
    _p("========================================")
    _p("  多物种 X 矩阵测试完成")
    _p("========================================")
    _p("检验 X 矩阵数: $n_tested  (非空: $n_ok, 空: $n_empty)")
    _p("最大偏差 max|X'SX - I|: $(max_dev)")
    _p()
    _p("检验标准 X† S X = I:  全部通过")
    _p()
    if skip_states > 0 || skip_reps > 0
        _p("跳过:")
        skip_states > 0 && _p("  因 n_states > 50000: $skip_states")
        skip_reps   > 0 && _p("  因 n_reps   > 10000: $skip_reps")
        _p()
    end
    return (n_tested, n_ok, n_empty, max_dev, skip_states, skip_reps)
end

# ============ 入口 ============

const _LOG_IO = Ref{IO}(devnull)
_p(args...; kwargs...) = (println(args...; kwargs...); println(_LOG_IO[], args...; kwargs...); flush(_LOG_IO[]))

const OUTPUT_FILE = "test_X_matrix_output.txt"

function parse_ncut_args()
    if isempty(ARGS)
        return nothing
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
    @time run_multi_species_X_test(max_ncut_user)
finally
    close(_LOG_IO[])
end

println("结果已保存至: $OUTPUT_FILE")
