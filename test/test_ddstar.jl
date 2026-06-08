# ==========================================================================
# D D* (pseudoscalar-vector) 体系测试
#
# 粒子: D (s=0) + D* (s=1), 可区分粒子, species=[1,1]
# 同位旋 I=0, κ=("[1]","[1]") (S₁×S₁ 平凡)
# 作用: V(σ',σ) = C0 × δ_{σ'σ} × f(n₁')f(n₁) × f(n₂')f(n₂)
# 形状因子: dipole  f(n) = 1/(1 + |n|²/Λ_n²)²
#
# H_raw = T_diag + fv_factor * V_hel          (全空间, 可区分粒子 S=I)
# H_proj = T_proj + project_V(...)             (投影空间)
#
# 验证: eig(H_proj) ⊂ eig(H_raw)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

function test_ddstar()
    M = NPHFforFVE.Momentum

    # ---------- 物理参数 ----------
    m_D  = 1864.84
    m_Ds = 2008.5
    C0   = 5.0e-7           # MeV⁻²
    Λ    = 1000.0           # MeV (dipole cutoff)
    L0   = 48
    a    = 0.1              # fm
    L_phys = L0 * a
    ħc  = 197.327           # MeV·fm
    Ncut = 20

    pref_T  = (2π * ħc / L_phys)^2
    Λ_n²    = (Λ / (2π * ħc / L_phys))^2
    per_mass = [m_D, m_Ds]

    # ---------- dipole 形状因子 ----------
    ff(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²)^2

    # ---------- 体系定义 ----------
    species = [1, 1]
    particle_types = [:boson, :boson]
    spins = Float64[0.0, 1.0]
    etas  = Float64[1.0, 1.0]
    N = length(species)
    d_total = M(0,0,0)

    # ---------- V_can ----------
    function V_can_func(n_prime, σ_prime, n, σ, extra...)
        σ1p, σ2p = σ_prime; σ1, σ2 = σ
        σ1p == σ1 || return zero(ComplexF64)
        σ2p == σ2 || return zero(ComplexF64)
        n1p, n2p = n_prime; n1, n2 = n
        val = C0 * ff(n1p) * ff(n1) * ff(n2p) * ff(n2)
        return ComplexF64(val)
    end

    per_spin_r = Rational{Int}[]
    for k in 1:length(species)
        for _ in 1:species[k]
            push!(per_spin_r, Rational{Int}(Int(2*spins[k]), 2))
        end
    end

    # ---------- 群 ----------
    needs_double = any(pt -> pt == :fermion, particle_types)
    group_els, group_name = NPHFforFVE.group_for_momentum(d_total; double_cover=needs_double)
    irrep_names = NPHFforFVE.OH_IRREP_NAMES

    # ---------- 代表态 ----------
    reps = NPHFforFVE.find_representatives(N; Ncut=Ncut, d=d_total,
        species=species, particle_types=particle_types)

    println("物理参数:")
    println("  Ncut = $Ncut,  L = $L0,  a = $a fm,  L_phys = $L_phys fm")
    println("  m_D = $m_D MeV,  m_D* = $m_Ds MeV")
    println("  C₀ = $C0 MeV⁻²,  Λ = $Λ MeV  (dipole)")
    println("  pref_T = $(round(pref_T, digits=1)) MeV²")
    println("代表态数目: $(length(reps))")

    kappa_tuple = ("[1]", "[1]")

    # ================================================================
    # 收集投影分块 + 全空间态 (跳过 ZM+自旋)
    # ================================================================
    function has_zm_spin(rep)
        off = 0
        for k in 1:length(species)
            for i in 1:species[k]
                if iszero(rep[off + i]) && spins[k] != 0.0
                    return true
                end
            end
            off += species[k]
        end
        return false
    end

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []   # (X, states, rep, Gamma, n_r)

    for rep in reps
        has_zm_spin(rep) && continue

        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species, particle_types=particle_types,
                spins=spins, d=d_total)
        catch
            [ntuple(_ -> 0.0, N)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names
                irrep_mats = try
                    NPHFforFVE.irrep_matrices(Gamma; group=group_name)
                catch
                    nothing
                end
                irrep_mats === nothing && continue

                result = try
                    NPHFforFVE.subspace_projection(rep, hel_float,
                        kappa_tuple, Gamma;
                        d_total=d_total, species=species,
                        particle_types=particle_types,
                        spins=spins, etas=etas)
                catch
                    continue
                end

                n_r = size(result.X, 2)
                n_r == 0 && continue

                idx_list = Int[]
                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                    push!(idx_list, state_to_idx[st])
                end

                push!(proj_blocks, (X=result.X, states=result.subspace_states,
                                   rep=rep, Gamma=Gamma, n_r=n_r, indices=idx_list))
            end
        end
    end

    K = length(all_states)
    println("全子空间态数: K = $K")
    println("投影分块数: $(length(proj_blocks))")

    # ================================================================
    # 构建 T_diag (动能)
    # ================================================================
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass))
        T_diag[idx, idx] = T
    end

    # ================================================================
    # 构建 V_hel 和 H_raw (全空间，可区分粒子 S=I)
    # ================================================================
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                       per_spin_r, per_spin_r, V_can_func)

    dd = 3*(N + N) - 6
    fv_factor = (2π * ħc / L_phys)^(dd/2)
    H_raw = T_diag + fv_factor * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))

    println("H_raw 维数: $(size(H_raw))")
    println("本征值范围: [$(round(evals_full[1], digits=2)), $(round(evals_full[end], digits=2))] MeV")

    # ================================================================
    # 按不可约表示分组
    # ================================================================
    blocks_by_irrep = Dict{String, Vector}()
    for blk in proj_blocks
        Gamma = blk.Gamma
        if !haskey(blocks_by_irrep, Gamma)
            blocks_by_irrep[Gamma] = []
        end
        push!(blocks_by_irrep[Gamma], blk)
    end

    println("\n─── 本征值归属性检验 ───")
    println(rpad("  Γ", 8), rpad("dim", 6), rpad("matched/total", 16), "status")
    println("  " * "-"^40)

    all_ok = true

    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        total_cols = sum(blk.n_r for blk in blocks)

        # ---- 用 project_V 构造投影势能 ----
        V_proj = zeros(ComplexF64, total_cols, total_cols)
        row_start = 1
        for (i, blk_i) in enumerate(blocks)
            n_r_i = blk_i.n_r
            col_start = 1
            for (j, blk_j) in enumerate(blocks)
                n_r_j = blk_j.n_r
                sub_V = NPHFforFVE.project_V(
                    blk_i.X, blk_j.X,
                    blk_i.states, blk_j.states,
                    per_spin_r, per_spin_r,
                    L_phys, V_can_func)
                V_proj[row_start:row_start+n_r_i-1,
                       col_start:col_start+n_r_j-1] .= sub_V
                col_start += n_r_j
            end
            row_start += n_r_i
        end

        # ---- 投影动能 ----
        T_proj = zeros(ComplexF64, total_cols, total_cols)
        row = 1
        for blk in blocks
            T_rep = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                        for (n_, m_) in zip(blk.rep, per_mass))
            for i in 1:blk.n_r
                T_proj[row, row] = T_rep
                row += 1
            end
        end

        H_proj = T_proj + V_proj
        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))

        # ---- 检验归属 ----
        matched = 0
        for ep in evals_proj
            if minimum(abs.(ep .- evals_full)) < 1e-8
                matched += 1
            end
        end

        total_evals = length(evals_proj)
        ok = matched == total_evals
        all_ok = all_ok && ok

        status = ok ? "✓" : "✗ FAIL"
        extra = ok ? "" : "  unmatched=$(total_evals-matched)"
        println(rpad("  $Gamma", 8), rpad("$(total_cols)", 6),
                rpad("$matched/$total_evals", 16), status * extra)
    end

    println("  " * "-"^40)
    println(all_ok ? "\n全部通过 ✓" : "\n存在失败 ✗")

    # ---- 额外物理检验: 仅 T1+ 有非零相互作用 ----
    println("\n─── 物理检验: V_proj 是否仅 T1+ 非零 ───")
    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        if isempty(blocks); continue; end
        total_cols = sum(blk.n_r for blk in blocks)
        V_proj = zeros(ComplexF64, total_cols, total_cols)
        row_start = 1
        for (i, blk_i) in enumerate(blocks)
            n_r_i = blk_i.n_r
            col_start = 1
            for (j, blk_j) in enumerate(blocks)
                n_r_j = blk_j.n_r
                sub_V = NPHFforFVE.project_V(
                    blk_i.X, blk_j.X, blk_i.states, blk_j.states,
                    per_spin_r, per_spin_r, L_phys, V_can_func)
                V_proj[row_start:row_start+n_r_i-1,
                       col_start:col_start+n_r_j-1] .= sub_V
                col_start += n_r_j
            end
            row_start += n_r_i
        end
        maxV = maximum(abs.(V_proj))
        is_t1 = Gamma == "T1+"
        ok_v = is_t1 ? maxV > 1e-12 : maxV < 1e-12
        println("  $Gamma: max|V|=$maxV  $(ok_v ? "✓" : "✗ 异常!")")
        all_ok = all_ok && ok_v
    end

    println(all_ok ? "\n全部通过 ✓" : "\n存在失败 ✗")
    return all_ok
end

test_ddstar()
