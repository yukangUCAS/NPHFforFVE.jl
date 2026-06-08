# ==========================================================================
# D*D* (vector-vector) 体系测试 — 两自旋 1 全同玻色子
#
# 物理: isospin-1/2
#   I=1 (isospin 对称) + κ=[2] (空间对称) → 自旋反对称: S=1
#   I=0 (isospin 反对称) + κ=[1,1] (空间反对称) → 自旋对称: S=0,2
#
# 作用: V(σ',σ) = C0 * cutoff * (δ_{11'}δ_{22'} ± δ_{12'}δ_{21'})
#   I=1: - (反对称), I=0: + (对称)
#
# 验证: GEP H_raw v = λ S v  本征值归属性 (同 ππ 方式)
#   S = direct + exchange (κ=[2]) / direct - exchange (κ=[1,1])
#   ZM λ₁=λ₂ (κ=[1,1]) 排除 (S=0)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

function test_dd()
    M = NPHFforFVE.Momentum
    m_Ds = 2008.5; C0 = 1.0; Λ = 1000.0
    L0 = 48; a = 0.1; L_phys = L0 * a
    ħc = 197.327; Λ_n² = (Λ / (2π * ħc / L_phys))^2
    Ncut = 20

    ch = FockChannel("D*D*", [2], [:boson], [m_Ds], [1], [1//2], [1.0], relativistic)
    per_spin, per_etas_arr = NPHFforFVE._expand_per_particle(ch)
    spin_val = Float64(only(unique(Rational{Int}.(per_spin))))
    etas_v = Float64.(per_etas_arr)
    per_spin_r = Rational{Int}.(per_spin)
    per_mass = NPHFforFVE._expand_per_particle_mass(ch)

    # I=1, κ=[2] (对称): 自旋对称   δ_{11'}δ_{22'} + δ_{12'}δ_{21'}  (S=0,2)
    # I=0, κ=[1,1] (反对称): 自旋反对称 δ_{11'}δ_{22'} - δ_{12'}δ_{21'}  (S=1)
    function make_V_dd(I_val)
        sign_spin = I_val == 1//1 ? +1.0 : -1.0
        return function(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, params)
            direct   = (sp[1]==s[1] && sp[2]==s[2]) ? 1.0 : 0.0
            exchange = (sp[1]==s[2] && sp[2]==s[1]) ? 1.0 : 0.0
            spin_factor = direct + sign_spin * exchange
            spin_factor == 0.0 && return 0.0
            cutoff = 1.0
            for n_ in nA; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
            for n_ in nB; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
            ComplexF64(C0 * cutoff * spin_factor)
        end
    end

    single_valued_irreps = ["A1+", "A2+", "E+", "T1+", "T2+",
                            "A1-", "A2-", "E-", "T1-", "T2-"]
    pref_T = (2π * ħc / L_phys)^2
    N_α = 2; dd = 3*(N_α+N_α)-6; fv_factor = (2π/L_phys)^(dd/2)

    all_ok = true

    for (I_val, kappa_label) in [(1//1, "[2]"), (0//1, "[1,1]")]
        subs = get_isospin_subchannels(ch, I_val)
        sub = subs[1]
        sign_ex = I_val == 1//1 ? +1.0 : -1.0
        is_κ_sym = I_val == 1//1

        println("\nI=$I_val, κ=$kappa_label, S = direct $(sign_ex > 0 ? "+" : "-") exchange")
        println("  自旋: $(I_val == 1//1 ? "对称 (S=0,2)" : "反对称 (S=1)")")

        V_contact = make_V_dd(I_val)
        V_adapter = NPHFforFVE._V_adapter(
            V_contact, sub.κ, sub.κ, sub.r, sub.r, sub.a, sub.a, (C0=C0, Λ_n²=Λ_n²))

        for irrep in single_valued_irreps
            projs = NPHFforFVE._get_channel_proj_list(
                ch, Ncut, M(0,0,0), sub.κ, irrep, spin_val, etas_v)
            proj_dim = sum(p.n_r for p in projs; init=0)
            proj_dim == 0 && continue

            # ---- all_states + S 矩阵 ----
            all_states = []
            state_to_idx = Dict()
            for p in projs
                for st in p.states
                    n_tup, lam_tup = st
                    is_zm = all(n -> n == M(0,0,0), n_tup)
                    if !is_κ_sym && is_zm && lam_tup[1] == lam_tup[2]
                        continue  # κ=[1,1]: ZM λ₁=λ₂ S=0, 排除
                    end
                    key = st
                    if !haskey(state_to_idx, key)
                        push!(all_states, st)
                        state_to_idx[key] = length(all_states)
                    end
                end
            end
            K = length(all_states)

            S_mat = zeros(Float64, K, K)
            for (i, (nA, lamA)) in enumerate(all_states)
                for (j, (nB, lamB)) in enumerate(all_states)
                    direct = (all(nA[k]==nB[k] for k in 1:2) &&
                              all(lamA[k]==lamB[k] for k in 1:2)) ? 1.0 : 0.0
                    exchange = (all(k -> nA[k]==nB[3-k], 1:2) &&
                                all(k -> lamA[k]==lamB[3-k], 1:2)) ? 1.0 : 0.0
                    S_mat[i,j] = direct + sign_ex * exchange
                end
            end

            # ---- T_raw = T_diag * S ----
            T_diag = zeros(ComplexF64, K, K)
            for (idx, (n_tup, _)) in enumerate(all_states)
                T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                        for (n_, m_) in zip(n_tup, per_mass))
                T_diag[idx, idx] = T
            end
            T_raw = T_diag * S_mat

            # ---- V_hel ----
            V_can_func(nα, σα, nβ, σβ, extra...) =
                V_contact(nα, nβ, σα, σβ, sub.κ, sub.κ, sub.r, sub.r, sub.a, sub.a, (C0=C0, Λ_n²=Λ_n²))
            V_hel = build_V_hel(all_states, all_states, per_spin_r, per_spin_r, V_can_func)

            H_raw = T_raw + fv_factor * V_hel

            # ---- X_big ----
            X_big = zeros(ComplexF64, K, proj_dim); col = 1
            for p in projs
                is_zm = all(n -> n == M(0,0,0), p.rep)
                if is_zm && !is_κ_sym
                    keep = [ist for (ist, st) in enumerate(p.states)
                            if st[2][1] != st[2][2]]
                    X_use = p.X[keep, :]; st_use = p.states[keep]
                else
                    X_use = p.X; st_use = p.states
                end
                idx = [state_to_idx[st] for st in st_use]
                X_big[idx, col:col+p.n_r-1] .= X_use
                col += p.n_r
            end

            # ---- GEP 本征值归属性 ----
            H_proj = X_big' * H_raw * X_big
            XSX = X_big' * S_mat * X_big

            evals_proj = sort(real.(eigvals(Hermitian(H_proj))))

            # 非对称化自旋基底可能导致 S 奇异，正则化处理
            S_eig = eigvals(S_mat)
            if minimum(abs.(S_eig)) < 1e-10
                S_reg = S_mat + 1e-12 * Matrix(1.0I, K, K)
            else
                S_reg = S_mat
            end
            evals_gen = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_reg))))
            evals_gen = evals_gen[isfinite.(evals_gen) .&& evals_gen .> 100]
            matched = count(ep -> minimum(abs.(ep .- evals_gen)) < 1e-8, evals_proj)

            s_ok = norm(XSX - I) < 1e-10
            ok = s_ok && matched == length(evals_proj)
            status = ok ? "✓" : "✗ FAIL"
            if ok; all_ok = all_ok && true; else; all_ok = false; end

            extra = ok ? "" : "  S=$(round(norm(XSX-I),digits=1)) attrib=$matched/$(length(evals_proj))"
            println("  $irrep: dim=$proj_dim, K=$K  $status$extra")
        end
    end

    println(all_ok ? "\n全部通过 ✓" : "\n存在失败 ✗")
    return all_ok
end

test_dd()
