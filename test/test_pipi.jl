# ==========================================================================
# ππ (pion-pion) 体系测试 — 两自旋 0 全同玻色子
#
# 物理: I=2, κ=[2] (isospin+spatial 全对称)
# 作用: 接触势 V = C0 * cutoff （无自旋结构）
#
# 关键点:
# 1. 全同玻色子: 重叠矩阵 S = direct + exchange ≠ I
#    零动量态 (n₁=n₂=0) 的 S[i,i] = 2
# 2. 动能矩阵非对角: T_raw = T_diag * S
# 3. 必须解广义本征值问题 H_raw v = λ S v
# 4. 投影后 X†SX = I (S-正交化), 本征值直接比较
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

function test_pipi()
M = NPHFforFVE.Momentum
m_π = 140.0; C0 = 0.5; Λ = 1000.0
L0 = 48; a = 0.1; L_phys = L0 * a
ħc = 197.327; Λ_n² = (Λ / (2π * ħc / L_phys))^2

ch = FockChannel("ππ", [2], [:boson], [m_π], [0], [1], [1.0], relativistic)
per_spin, per_etas_arr = NPHFforFVE._expand_per_particle(ch)
spin_val = Float64(only(unique(Rational{Int}.(per_spin))))
etas_v = Float64.(per_etas_arr)
per_spin_r = Rational{Int}.(per_spin)
per_mass = NPHFforFVE._expand_per_particle_mass(ch)

subs = get_isospin_subchannels(ch, 2//1)
sub = subs[1]
println("I=2, κ=$(sub.κ)")

function V_contact(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, params)
    cutoff = 1.0
    for n_ in nA; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
    for n_ in nB; cutoff /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
    ComplexF64(C0 * cutoff)
end

# 自旋 0 只使用单值不可约表示 (整数自旋, 无 G1±/G2±/H±)
single_valued_irreps = ["A1+", "A2+", "E+", "T1+", "T2+",
                        "A1-", "A2-", "E-", "T1-", "T2-"]

all_ok = true
for irrep in single_valued_irreps
    projs = NPHFforFVE._get_channel_proj_list(ch, 4, M(0,0,0), sub.κ, irrep, spin_val, etas_v)
    proj_dim = sum(p.n_r for p in projs; init=0)
    proj_dim == 0 && continue

    all_states = []; state_to_idx = Dict()
    for p in projs
        for st in p.states
            key = st; haskey(state_to_idx, key) || (push!(all_states, st); state_to_idx[key] = length(all_states))
        end
    end
    K = length(all_states)

    # 重叠矩阵 S = direct + exchange
    S_mat = zeros(Float64, K, K)
    for (i, (nA, lamA)) in enumerate(all_states)
        for (j, (nB, lamB)) in enumerate(all_states)
            direct   = (all(nA[k]==nB[k] for k in 1:2) && all(lamA[k]==lamB[k] for k in 1:2)) ? 1.0 : 0.0
            exchange = (all(k -> nA[k]==nB[3-k], 1:2) && all(k -> lamA[k]==lamB[3-k], 1:2)) ? 1.0 : 0.0
            S_mat[i,j] = direct + exchange
        end
    end

    # 动能 T_raw = T_diag * S
    pref_T = (2π * ħc / L_phys)^2
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_))) for (n_, m_) in zip(n_tup, per_mass))
        T_diag[idx, idx] = T
    end
    T_raw = T_diag * S_mat

    # V_hel
    V_adapter = NPHFforFVE._V_adapter(V_contact, sub.κ, sub.κ, sub.r, sub.r, sub.a, sub.a, nothing)
    V_adapt(nα, σα, nβ, σβ, extra...) = V_adapter(nα, σα, nβ, σβ, extra...)
    V_hel = build_V_hel(all_states, all_states, per_spin_r, per_spin_r, V_adapt)

    N_α = 2; dd = 3*(N_α+N_α)-6; fv_factor = (2π/L_phys)^(dd/2)
    H_raw = T_raw + fv_factor * V_hel

    # 投影矩阵 + 检验
    X_big = zeros(ComplexF64, K, proj_dim); col = 1
    for p in projs; idx = [state_to_idx[st] for st in p.states]; X_big[idx, col:col+p.n_r-1] .= p.X; col += p.n_r; end

    # S-正交性
    XSX = X_big' * S_mat * X_big
    s_ok = norm(XSX - I) < 1e-10

    # X†HX = H_proj
    H_proj = X_big' * H_raw * X_big
    evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
    evals_gen  = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_mat))))
    matched = count(ep -> minimum(abs.(ep .- evals_gen)) < 1e-8, evals_proj)

    ok = s_ok && matched == length(evals_proj)
    status = ok ? "✓" : "✗ FAIL"
    if ok; all_ok = all_ok && true; else; all_ok = false; end
    extra = ok ? "" : "  X†SX=$(round(norm(XSX-I),digits=1)) attrib=$matched/$(length(evals_proj))"
    println("  $irrep: dim=$proj_dim, K=$K  $status$extra")
end

println(all_ok ? "\n全部通过 ✓" : "\n存在失败 ✗")
return all_ok
end

test_pipi()
