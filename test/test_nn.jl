# ==========================================================================
# NN (nucleon-nucleon) 体系测试 — 总自旋投影相互作用
# 两自旋 1/2 全同费米子
#
# 作用: V(σ'₁,σ'₂; σ₁,σ₂) = C * (δ_{σ'₁,σ₁}δ_{σ'₂,σ₂} ± δ_{σ'₁,σ₂}δ_{σ'₂,σ₁})
#   I=1, κ=[2] → S=0 (反对称): sign = -
#   I=0, κ=[1,1] → S=1 (对称):  sign = +
#
# Pauli 原理: I + κ + S 全反对称
#   I=1 (对称) + κ=[2] (对称) → S=0 (反对称)
#   I=0 (反对称) + κ=[1,1] (反对称) → S=1 (对称)
#
# 双覆盖群 2O_h (96 元素), dipole 形状因子
# 检验: 子空间不变性 + 本征值归属性
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

# =========================== 物理常数 ===========================
const M_N    = 938.92  # 核子质量 [MeV]
const ħc_val = 197.327 # [MeV·fm]

# =========================== 相互作用 ===========================

function make_V_nn(C0, Λ_n²)
    return function(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, params)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        # I=1 (κ=[2], S=0): 反对称 → - ; I=0 (κ=[1,1], S=1): 对称 → +
        sign_ex = kapA == "[2]" ? -1.0 : +1.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        cutoff = 1.0
        for n_ in nA; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
        for n_ in nB; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
        ComplexF64(C0 * cutoff * sf)
    end
end

# =========================== 投影 + 检验 ===========================

function check_channel(ch, I_val, V_func, L_phys, label)
    per_spin, per_etas_arr = NPHFforFVE._expand_per_particle(ch)
    spin_val = Float64(only(unique(Rational{Int}.(per_spin))))
    etas_v = Float64.(per_etas_arr)
    per_spin_r = Rational{Int}.(per_spin)
    per_mass = NPHFforFVE._expand_per_particle_mass(ch)

    subs = get_isospin_subchannels(ch, I_val)
    sub = subs[1]
    kappa_str = I_val == 1//1 ? "[2]" : "[1,1]"
    expected_S = I_val == 1//1 ? "S=0" : "S=1"

    V_adapter = NPHFforFVE._V_adapter(
        V_func, sub.κ, sub.κ, sub.r, sub.r, sub.a, sub.a, nothing)

    println("I=$I_val, κ=$kappa_str ($expected_S)  ——  $label")
    sign_label = I_val == 1//1 ? "-" : "+"
    println("  V = C * (δ_{σ'₁,σ₁}δ_{σ'₂,σ₂} $sign_label δ_{σ'₁,σ₂}δ_{σ'₂,σ₁})")

    n_pass = 0; n_fail = 0
    for irrep in NPHFforFVE.OH2_IRREP_NAMES
        projs = NPHFforFVE._get_channel_proj_list(
            ch, 20, NPHFforFVE.Momentum(0,0,0), sub.κ, irrep, spin_val, etas_v)
        proj_dim = sum(p.n_r for p in projs; init=0)
        proj_dim == 0 && continue

        all_states = []
        state_to_idx = Dict()
        for p in projs
            for st in p.states
                key = st
                if !haskey(state_to_idx, key)
                    push!(all_states, st)
                    state_to_idx[key] = length(all_states)
                end
            end
        end
        K = length(all_states)

        X_big = zeros(ComplexF64, K, proj_dim); col = 1
        for p in projs
            idx = [state_to_idx[st] for st in p.states]
            X_big[idx, col:col+p.n_r-1] .= p.X
            col += p.n_r
        end

        V_adapt(nα, σα, nβ, σβ, extra...) = V_adapter(nα, σα, nβ, σβ, extra...)
        V_hel = build_V_hel(all_states, all_states, per_spin_r, per_spin_r, V_adapt)
        V_proj = X_big' * V_hel * X_big

        dv = norm(V_hel * X_big - X_big * V_proj)

        N_α = length(all_states[1][1])
        dd = 3 * (N_α + N_α) - 6
        fv_factor = (2π / L_phys)^(dd / 2)
        pref_T = (2π * ħc_val / L_phys)^2

        T_diag = zeros(ComplexF64, K, K)
        for (idx, (n_tup, _)) in enumerate(all_states)
            T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                    for (n_, m_) in zip(n_tup, per_mass))
            T_diag[idx, idx] = T
        end

        H_raw = T_diag + fv_factor * V_hel
        H_proj = X_big' * H_raw * X_big

        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_gen  = sort(real.(eigvals(H_raw, Matrix(1.0I, K, K))))
        evals_gen  = evals_gen[isfinite.(evals_gen) .&& evals_gen .> 100]
        matched = count(ep -> minimum(abs.(ep .- evals_gen)) < 1e-8, evals_proj)

        ok = dv < 1e-10 && matched == length(evals_proj)
        status = ok ? "✓" : "✗ FAIL"
        if ok; n_pass += 1; else; n_fail += 1; end

        detail = ok ? "" : "  dv=$(round(dv,digits=1)) attrib=$(matched)/$(length(evals_proj))"
        println("  $irrep: dim=$proj_dim, K=$K  $status$detail")
    end
    println("  通过: $n_pass, 失败: $n_fail")
    return n_fail == 0
end

# =========================== 主程序 ===========================

function main()
    L0 = 48; a = 0.1; L_phys = L0 * a
    C0 = 15.0e-6; Λ = 1000.0; Λ_n² = (Λ / (2π * ħc_val / L_phys))^2

    ch = FockChannel("NN", [2], [:fermion], [M_N],
                     [1//2], [1//2], [1.0], relativistic)

    println("NN 体系 — 总自旋投影相互作用")
    println("  L=$L0, a=$a fm, Ncut=20")
    println("  C=$(C0) MeV⁻², Λ=$Λ MeV")
    println("  m_N=$(M_N) MeV")

    V_func = make_V_nn(C0, Λ_n²)

    all_pass = true
    ok1 = check_channel(ch, 1//1, V_func, L_phys, "V = V_S (总自旋投影)")
    ok2 = check_channel(ch, 0//1, V_func, L_phys, "V = V_S (总自旋投影)")

    println("\n$(repeat("=", 60))")
    all_pass = ok1 && ok2
    println(all_pass ? "全部测试通过 ✓" : "存在失败测试 ✗")
end

main()
