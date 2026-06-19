# ==========================================================================
# 3π → 3π 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)  ← 主体代码
#   evals_full = eig(H_raw, S)                                 ← 底层参考
#   验证: eig(H_proj) ⊂ evals_full
#
# π: s=0, I=1, boson, η=-1, m=139.57 MeV
# 全同玻色子 × 3
#
# 同位旋分解: (I=1)³ = 7⊕5⊕5⊕3⊕3⊕3⊕1
#   I=3 → κ="[3]"        接触相互作用
#   I=2 → κ="[2,1]"      2×2 p 波
#   I=0 → κ="[1,1,1]"    p 波三重积
#   I=1 → κ="[3]"⊕"[2,1]"  接触 ⊕ 2×2 p 波 ⊕ 耦合
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

# ============================================================================
# 共用参数与辅助量
# ============================================================================
const M_3pi    = NPHFforFVE.Momentum
const m_π_3pi  = 139.57
const Λ_3pi    = 1000.0
const L0_3pi   = 48
const a_3pi    = 0.1
const L_phys_3pi = L0_3pi * a_3pi
const ħc_3pi   = 197.327
const Ncut_3pi = 10

const pref_T_3pi   = (2π * ħc_3pi / L_phys_3pi)^2
const Λ_n²_3pi     = (Λ_3pi / (2π * ħc_3pi / L_phys_3pi))^2
const fv_factor_3pi = (2π * ħc_3pi / L_phys_3pi)^6
const mom_factor_3pi = (2π * ħc_3pi / L_phys_3pi)^3

const d_3pi = M_3pi(0,0,0)
const OH = NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES

const species_3pi    = [3]
const particle_3pi   = [:boson]
const N_α_3pi        = 3
const per_mass_3pi   = [m_π_3pi, m_π_3pi, m_π_3pi]
const per_spin_r_3pi = Rational{Int}[0//1, 0//1, 0//1]
const etas_3pi       = [-1.0]
const spin_val_3pi   = 0.0

ff_3pi(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_3pi)^2

function distinct_levels(evals::Vector{Float64}, n::Int; tol::Float64=1e-8)
    distinct = Float64[]
    for e in evals
        if isempty(distinct) || e - distinct[end] > tol
            push!(distinct, e)
        end
        length(distinct) >= n && break
    end
    return distinct
end

# ============================================================================
# 通用辅助: 构造参考谱 (底层全空间 H_raw, S)
#           枚举所有不可约表示中的态, 求解 eig(H_raw, S)
# ============================================================================
function _build_reference(d, Ncut, species, particle_types, N_α,
                          per_mass, per_spin_r, kappa, V_can_func,
                          fv_factor, pref_T, irrep_names,
                          spin_val, etas)

    reps = NPHFforFVE.find_representatives(N_α; Ncut=Ncut, d=d,
        species=species, particle_types=particle_types)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species, particle_types=particle_types,
                spins=[spin_val], d=d)
        catch
            [ntuple(_ -> 0.0, N_α)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa, Gamma;
                    d_total=d, species_type=:boson,
                    spin=spin_val, etas=etas)
                n_r = size(result.X, 2)
                n_r == 0 && continue

                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                end

                push!(proj_blocks, (X=result.X, states=result.subspace_states,
                                    Gamma=Gamma, n_r=n_r))
            end
        end
    end

    K = length(all_states)
    dim_κ = NPHFforFVE.get_SN_irrep_dim(N_α, kappa)

    S_big = NPHFforFVE.build_S_matrix(all_states, N_α, kappa, :boson)

    T_diag = zeros(ComplexF64, K * dim_κ, K * dim_κ)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass))
        rng = (idx-1)*dim_κ+1 : idx*dim_κ
        T_diag[rng, rng] = T .* S_big[rng, rng]
    end

    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                   per_spin_r, per_spin_r, V_can_func)

    H_raw = T_diag + fv_factor * V_hel

    # 广义本征值 Hψ = E S ψ (处理 S 半正定性)
    S_eig = eigen(Hermitian(S_big))
    tol_S = 1e-12
    good_S = S_eig.values .> tol_S
    if all(good_S)
        evals_full = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_big))))
    else
        U = S_eig.vectors[:, good_S]
        H_red = U' * H_raw * U
        S_red = U' * S_big * U
        evals_full = sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    end
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, all_states=all_states,
            state_to_idx=state_to_idx, proj_blocks=proj_blocks,
            K=K, dim_κ=dim_κ)
end


# ============================================================================
# 通用辅助: 用 build_hamiltonian_block 验证各不可约表示
# ============================================================================
function _verify_with_hamiltonian(sys, V_func, params, evals_full, free_Ts, label)
    println("  自由能级 (前10非简并): $(round.(free_Ts, digits=4))")
    println()
    all_ok = true
    for Gamma in sys.selected_irreps
        H_proj = NPHFforFVE.build_hamiltonian_block(sys, Gamma, V_func, params)
        dim_total = size(H_proj, 1)
        dim_total == 0 && continue

        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_proj = evals_proj[isfinite.(evals_proj)]

        di = distinct_levels(evals_proj, 5)
        matched = count(ep -> minimum(abs.(ep .- evals_full)) < 1e-8, evals_proj)
        ok = matched == length(evals_proj)
        all_ok = all_ok && ok
        status = ok ? "✓" : "✗"
        println("  $Gamma: dim=$dim_total, matched=$matched/$(length(evals_proj))  $status  [$(round.(di, digits=4))]")
        @test ok
    end
    println()
    println(all_ok ? "全部通过 ✓" : "存在失败 ✗")
    return all_ok
end


# ============================================================================
# I=3: κ="[3]", dim_κ=1, 接触相互作用
# V = Cs * ff(n'_π1)ff(n'_π2)ff(n'_π3) * ff(n_π1)ff(n_π2)ff(n_π3)
# ============================================================================
function test_3pi_I3()
    Cs = 2.0e-14

    function V_can_func(np, sp, n, s, extra...)
        n1p, n2p, n3p = np; n1, n2, n3 = n
        ComplexF64(Cs * ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) *
                          ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3))
    end

    # ---- 参考谱 (底层全空间) ----
    ref = _build_reference(d_3pi, Ncut_3pi, species_3pi, particle_3pi, N_α_3pi,
        per_mass_3pi, per_spin_r_3pi, "[3]", V_can_func,
        fv_factor_3pi, pref_T_3pi, OH, spin_val_3pi, etas_3pi)

    K = ref.K
    println("3π → 3π  I=3  κ=[3]  Ncut=$Ncut_3pi")
    println("K = $K states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_3pi * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_3pi))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    # ---- 主体: FockSystem + build_hamiltonian_block ----
    ch = FockChannel("3pi", [3], [:boson], [m_π_3pi], [0//1], [1//1], [-1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_3pi, [ch], L0_3pi, a_3pi, 3//1, OH)
    params = (Cs=Cs,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        n1p, n2p, n3p = nA; n1, n2, n3 = nB
        ComplexF64(p.Cs * ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) *
                          ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3))
    end

    return _verify_with_hamiltonian(sys, V_func, params, ref.evals_full, free_Ts_d, "I=3")
end


# ============================================================================
# I=2: κ="[2,1]", dim_κ=2, p 波相互作用 (2×2 矩阵)
# V_{ab} 取 [2,1] 表示的标准矩阵元
# ============================================================================
function test_3pi_I2()
    C0 = 1.0e-18

    function V_can_func(np, sp, n, s, extra...)
        n1p, n2p, n3p = np; n1, n2, n3 = n
        dp11 = Float64(n3p[1]*n3[1]+n3p[2]*n3[2]+n3p[3]*n3[3])
        dp12 = Float64(n3p[1]*(n2[1]-n1[1])+n3p[2]*(n2[2]-n1[2])+n3p[3]*(n2[3]-n1[3]))
        dp21 = Float64((n2p[1]-n1p[1])*n3[1]+(n2p[2]-n1p[2])*n3[2]+(n2p[3]-n1p[3])*n3[3])
        dp22 = Float64((n2p[1]-n1p[1])*(n2[1]-n1[1])+(n2p[2]-n1p[2])*(n2[2]-n1[2])+(n2p[3]-n1p[3])*(n2[3]-n1[3]))
        ff_all = ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) * ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3)
        fac = C0 * pref_T_3pi * ff_all
        ComplexF64[ fac * dp11                 fac * (sqrt(3)/3) * dp12;
                    fac * (sqrt(3)/3) * dp21   fac * (1/3) * dp22 ]
    end

    # ---- 参考谱 ----
    ref = _build_reference(d_3pi, Ncut_3pi, species_3pi, particle_3pi, N_α_3pi,
        per_mass_3pi, per_spin_r_3pi, "[2,1]", V_can_func,
        fv_factor_3pi, pref_T_3pi, OH, spin_val_3pi, etas_3pi)

    println("3π → 3π  I=2  κ=[2,1]  Ncut=$Ncut_3pi")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_3pi * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_3pi))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    # ---- 主体: FockSystem + V_func (标量, 由 aA/aB 决定矩阵元) ----
    ch = FockChannel("3pi", [3], [:boson], [m_π_3pi], [0//1], [1//1], [-1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_3pi, [ch], L0_3pi, a_3pi, 2//1, OH)
    params = (C0=C0,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        n1p, n2p, n3p = nA; n1, n2, n3 = nB
        ff_all = ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) * ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3)
        fac = p.C0 * (2π * ħc_3pi / L_phys)^2 * ff_all
        dp11 = Float64(n3p[1]*n3[1]+n3p[2]*n3[2]+n3p[3]*n3[3])
        dp12 = Float64(n3p[1]*(n2[1]-n1[1])+n3p[2]*(n2[2]-n1[2])+n3p[3]*(n2[3]-n1[3]))
        dp21 = Float64((n2p[1]-n1p[1])*n3[1]+(n2p[2]-n1p[2])*n3[2]+(n2p[3]-n1p[3])*n3[3])
        dp22 = Float64((n2p[1]-n1p[1])*(n2[1]-n1[1])+(n2p[2]-n1p[2])*(n2[2]-n1[2])+(n2p[3]-n1p[3])*(n2[3]-n1[3]))
        ComplexF64[ fac * dp11                 fac * (sqrt(3)/3) * dp12;
                    fac * (sqrt(3)/3) * dp21   fac * (1/3) * dp22 ]
    end

    return _verify_with_hamiltonian(sys, V_func, params, ref.evals_full, free_Ts_d, "I=2")
end


# ============================================================================
# I=0: κ="[1,1,1]", dim_κ=1, p 波三重积相互作用
# V = C0 * (p_1·p_2×p_3)(k_1·k_2×k_3) * ff * ff
# ============================================================================
function test_3pi_I0()
    C0 = 1.0e-27

    function V_can_func(np, sp, n, s, extra...)
        n1p, n2p, n3p = np; n1, n2, n3 = n
        tp_out = n1p[1]*(n2p[2]*n3p[3]-n2p[3]*n3p[2]) +
                 n1p[2]*(n2p[3]*n3p[1]-n2p[1]*n3p[3]) +
                 n1p[3]*(n2p[1]*n3p[2]-n2p[2]*n3p[1])
        tp_in  = n1[1]*(n2[2]*n3[3]-n2[3]*n3[2]) +
                 n1[2]*(n2[3]*n3[1]-n2[1]*n3[3]) +
                 n1[3]*(n2[1]*n3[2]-n2[2]*n3[1])
        ComplexF64(C0 * mom_factor_3pi^2 * tp_out * tp_in *
                   ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) *
                   ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3))
    end

    # ---- 参考谱 ----
    ref = _build_reference(d_3pi, Ncut_3pi, species_3pi, particle_3pi, N_α_3pi,
        per_mass_3pi, per_spin_r_3pi, "[1,1,1]", V_can_func,
        fv_factor_3pi, pref_T_3pi, OH, spin_val_3pi, etas_3pi)

    println("3π → 3π  I=0  κ=[1,1,1]  Ncut=$Ncut_3pi")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_3pi * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_3pi))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    # ---- 主体 ----
    ch = FockChannel("3pi", [3], [:boson], [m_π_3pi], [0//1], [1//1], [-1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_3pi, [ch], L0_3pi, a_3pi, 0//1, OH)
    params = (C0=C0,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        n1p, n2p, n3p = nA; n1, n2, n3 = nB
        tp_out = n1p[1]*(n2p[2]*n3p[3]-n2p[3]*n3p[2]) +
                 n1p[2]*(n2p[3]*n3p[1]-n2p[1]*n3p[3]) +
                 n1p[3]*(n2p[1]*n3p[2]-n2p[2]*n3p[1])
        tp_in  = n1[1]*(n2[2]*n3[3]-n2[3]*n3[2]) +
                 n1[2]*(n2[3]*n3[1]-n2[1]*n3[3]) +
                 n1[3]*(n2[1]*n3[2]-n2[2]*n1[3])
        ComplexF64(p.C0 * mom_factor_3pi^2 * tp_out * tp_in *
                   ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) *
                   ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3))
    end

    return _verify_with_hamiltonian(sys, V_func, params, ref.evals_full, free_Ts_d, "I=0")
end


# ============================================================================
# I=1 coupled: κ="[3]" ⊕ κ="[2,1]", dim_total=3
#
# [3] 块: 接触 Cs * ff * ff
# [2,1] 块: 2×2 p 波
# 耦合块: [3] ↔ [2,1], 形状因子 f₁, f₂ 由动量差构造
#
# 参考谱需要手动拼接 S_big (3K×3K) 和 V_can (3×3)
# ============================================================================
function test_3pi_I1_coupled()
    Cs = 2.0e-14
    Cm = 1.0e-18
    Cc = 1.0e-18

    dim_s = 1; dim_m_κ = 2; dim_total = 3

    # ========== V_can (3×3) 供参考谱用 ==========
    function V_can_func(np, sp, n, s, extra...)
        n1p, n2p, n3p = np; n1, n2, n3 = n
        ff_all = ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) * ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3)
        pref = pref_T_3pi

        # [3] contact
        v11 = Cs * ff_all

        # [2,1] 2×2 p-wave
        v22 = Cm*pref*Float64(n3p[1]*n3[1]+n3p[2]*n3[2]+n3p[3]*n3[3])*ff_all
        v23 = Cm*pref*(sqrt(3)/3)*Float64(n3p[1]*(n2[1]-n1[1])+n3p[2]*(n2[2]-n1[2])+n3p[3]*(n2[3]-n1[3]))*ff_all
        v32 = Cm*pref*(sqrt(3)/3)*Float64((n2p[1]-n1p[1])*n3[1]+(n2p[2]-n1p[2])*n3[2]+(n2p[3]-n1p[3])*n3[3])*ff_all
        v33 = Cm*pref*(1/3)*Float64((n2p[1]-n1p[1])*(n2[1]-n1[1])+(n2p[2]-n1p[2])*(n2[2]-n1[2])+(n2p[3]-n1p[3])*(n2[3]-n1[3]))*ff_all

        # 耦合 [3] ↔ [2,1]
        sq1  = Float64(sum(abs2,n1)); sq2  = Float64(sum(abs2,n2)); sq3  = Float64(sum(abs2,n3))
        sq1p = Float64(sum(abs2,n1p)); sq2p = Float64(sum(abs2,n2p)); sq3p = Float64(sum(abs2,n3p))
        f1_k = (1/sqrt(6))*(2*sq3-sq1-sq2); f2_k = (1/sqrt(2))*(sq2-sq1)
        f1_p = (1/sqrt(6))*(2*sq3p-sq1p-sq2p); f2_p = (1/sqrt(2))*(sq2p-sq1p)
        v12 = Cc*pref*f1_k*ff_all; v13 = Cc*pref*f2_k*ff_all
        v21 = Cc*pref*f1_p*ff_all; v31 = Cc*pref*f2_p*ff_all

        ComplexF64[ v11  v12  v13; v21  v22  v23; v31  v32  v33 ]
    end

    # ========== 参考谱: 手动枚举, 拼接 [3] 和 [2,1] ==========
    reps = NPHFforFVE.find_representatives(N_α_3pi; Ncut=Ncut_3pi, d=d_3pi,
        species=species_3pi, particle_types=particle_3pi)

    all_states = []; state_to_idx = Dict(); proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_3pi, particle_types=particle_3pi,
                spins=[spin_val_3pi], d=d_3pi)
        catch
            [ntuple(_ -> 0.0, N_α_3pi)]
        end
        for hel in h_reps
            hel_float = Tuple(Float64.(hel))
            for kap in ("[3]", "[2,1]")
                for Gamma in OH
                    result = NPHFforFVE.subspace_projection(rep, hel_float,
                        kap, Gamma; d_total=d_3pi, species_type=:boson,
                        spin=spin_val_3pi, etas=etas_3pi)
                    size(result.X,2)==0 && continue
                    for st in result.subspace_states
                        if !haskey(state_to_idx, st)
                            push!(all_states, st); state_to_idx[st] = length(all_states)
                        end
                    end
                    push!(proj_blocks, (X=result.X, states=result.subspace_states,
                                        Gamma=Gamma, n_r=size(result.X,2), kap=kap))
                end
            end
        end
    end

    K = length(all_states)

    # 拼接 3K×3K S_big
    S_s = NPHFforFVE.build_S_matrix(all_states, N_α_3pi, "[3]", :boson)
    S_m = NPHFforFVE.build_S_matrix(all_states, N_α_3pi, "[2,1]", :boson)
    S_big = zeros(Float64, K*dim_total, K*dim_total)
    for k in 1:K
        S_big[(k-1)*dim_total+1,(k-1)*dim_total+1] = S_s[k,k]
        mr = (k-1)*dim_total+2:k*dim_total
        ms = (k-1)*dim_m_κ+1:k*dim_m_κ
        S_big[mr,mr] .= S_m[ms,ms]
    end

    # T_diag
    T_diag = zeros(ComplexF64, K*dim_total, K*dim_total)
    for (idx,(n_tup,_)) in enumerate(all_states)
        Tk = sum(sqrt(m_^2 + pref_T_3pi*Float64(sum(abs2,n_)))
                for (n_,m_) in zip(n_tup, per_mass_3pi))
        rng = (idx-1)*dim_total+1:idx*dim_total
        T_diag[rng,rng] = Tk .* S_big[rng,rng]
    end

    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                   per_spin_r_3pi, per_spin_r_3pi, V_can_func)
    H_raw = T_diag + fv_factor_3pi * V_hel

    S_eig = eigen(Hermitian(S_big))
    good_S = S_eig.values .> 1e-12
    U = S_eig.vectors[:, good_S]
    H_red = U'*H_raw*U; S_red = U'*S_big*U
    evals_full = sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    evals_full = evals_full[isfinite.(evals_full)]

    println("3π → 3π  I=1 coupled  κ=[3]⊕[2,1]  Ncut=$Ncut_3pi")
    println("K = $K states,  reference eigenvalues = $(length(evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_3pi * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_3pi))
        for (nt, _) in all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    # ========== 主体: FockSystem + V_func ==========
    ch = FockChannel("3pi", [3], [:boson], [m_π_3pi], [0//1], [1//1], [-1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_3pi, [ch], L0_3pi, a_3pi, 1//1, OH)
    params = (Cs=Cs, Cm=Cm, Cc=Cc)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        n1p, n2p, n3p = nA; n1, n2, n3 = nB
        ff_all = ff_3pi(n1p)*ff_3pi(n2p)*ff_3pi(n3p) * ff_3pi(n1)*ff_3pi(n2)*ff_3pi(n3)
        pref = (2π * ħc_3pi / L_phys)^2

        if kapA == "[3]" && kapB == "[3]"
            return p.Cs * ff_all

        elseif kapA == "[2,1]" && kapB == "[2,1]"
            dp11 = Float64(n3p[1]*n3[1]+n3p[2]*n3[2]+n3p[3]*n3[3])
            dp12 = Float64(n3p[1]*(n2[1]-n1[1])+n3p[2]*(n2[2]-n1[2])+n3p[3]*(n2[3]-n1[3]))
            dp21 = Float64((n2p[1]-n1p[1])*n3[1]+(n2p[2]-n1p[2])*n3[2]+(n2p[3]-n1p[3])*n3[3])
            dp22 = Float64((n2p[1]-n1p[1])*(n2[1]-n1[1])+(n2p[2]-n1p[2])*(n2[2]-n1[2])+(n2p[3]-n1p[3])*(n2[3]-n1[3]))
            return ComplexF64[ p.Cm*pref*dp11*ff_all           p.Cm*pref*(sqrt(3)/3)*dp12*ff_all;
                              p.Cm*pref*(sqrt(3)/3)*dp21*ff_all  p.Cm*pref*(1/3)*dp22*ff_all ]

        elseif kapA == "[3]" && kapB == "[2,1]"
            sq1 = Float64(sum(abs2,n1)); sq2 = Float64(sum(abs2,n2)); sq3 = Float64(sum(abs2,n3))
            f1 = p.Cc*pref*(1/sqrt(6))*(2*sq3-sq1-sq2)*ff_all
            f2 = p.Cc*pref*(1/sqrt(2))*(sq2-sq1)*ff_all
            return ComplexF64[f1  f2]

        elseif kapA == "[2,1]" && kapB == "[3]"
            sq1p = Float64(sum(abs2,n1p)); sq2p = Float64(sum(abs2,n2p)); sq3p = Float64(sum(abs2,n3p))
            f1 = p.Cc*pref*(1/sqrt(6))*(2*sq3p-sq1p-sq2p)*ff_all
            f2 = p.Cc*pref*(1/sqrt(2))*(sq2p-sq1p)*ff_all
            return ComplexF64[f1; f2]
        end

        return zero(ComplexF64)
    end

    return _verify_with_hamiltonian(sys, V_func, params, evals_full, free_Ts_d, "I=1 coupled")
end


# ============================================================================
# 编排
# ============================================================================
function test_3pi()
    ok3 = test_3pi_I3()
    println("============================================================")
    println()
    ok2 = test_3pi_I2()
    println("============================================================")
    println()
    ok0 = test_3pi_I0()
    println("============================================================")
    println()
    ok1c = test_3pi_I1_coupled()
    return ok3 && ok2 && ok0 && ok1c
end

test_3pi()
