# ==========================================================================
# D*D* → D*D* 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw, S)
#   验证: eig(H_proj) ⊂ evals_full
#
# D*: s=1, I=1/2, boson, η=+1, m=2008.5 MeV
# 全同玻色子 × 2
#
# I=1 (isospin 对称) + κ=[2] (空间对称) → 自旋对称 (S=0,2): sign_spin = +1
# I=0 (isospin 反对称) + κ=[1,1] (空间反对称) → 自旋反对称 (S=1): sign_spin = -1
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_dsds      = NPHFforFVE.Momentum
const m_Ds_dsds   = 2008.5
const Λ_dsds      = 1000.0
const L0_dsds     = 48
const a_dsds      = 0.1
const L_phys_dsds = L0_dsds * a_dsds
const ħc_dsds     = 197.327
const Ncut_dsds   = 20
const C0_dsds     = 1.0

const pref_T_dsds   = (2π * ħc_dsds / L_phys_dsds)^2
const Λ_n²_dsds     = (Λ_dsds / (2π * ħc_dsds / L_phys_dsds))^2
const d_dsds        = M_dsds(0,0,0)

const species_dsds    = [2]
const particle_dsds   = [:boson]
const N_α_dsds        = 2
const per_mass_dsds   = [m_Ds_dsds, m_Ds_dsds]
const per_spin_r_dsds = Rational{Int}[1//1, 1//1]
const spin_val_dsds   = 1.0
const etas_dsds       = [1.0]

ff_dsds(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_dsds)^2

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
# 构造参考谱
# ============================================================================
function _build_dstardstar_reference(kappa, sign_spin)
    function V_can_func(np, sp, n, s, extra...)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_spin * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in np; ffall *= ff_dsds(n_); end
        for n_ in n;  ffall *= ff_dsds(n_); end
        ComplexF64(C0_dsds * ffall * sf)
    end

    reps = NPHFforFVE.find_representatives(N_α_dsds; Ncut=Ncut_dsds, d=d_dsds,
        species=species_dsds, particle_types=particle_dsds)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        M_zm = count(n -> n == d_dsds, rep)
        h_reps = if M_zm > 0
            # ZM 粒子自旋非零，螺旋度无定义，直接走 ZM 管线
            [ntuple(_ -> 0.0, N_α_dsds)]
        else
            NPHFforFVE.helicity_representatives(rep;
                species=species_dsds, particle_types=particle_dsds,
                spins=[spin_val_dsds], d=d_dsds)
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa, Gamma;
                    d_total=d_dsds, species_type=:boson,
                    spin=spin_val_dsds, etas=etas_dsds)
                size(result.X, 2) == 0 && continue

                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                end

                push!(proj_blocks, (X=result.X, states=result.subspace_states,
                                    Gamma=Gamma, n_r=size(result.X,2)))
            end
        end
    end

    K = length(all_states)

    # S matrix: 按 n_tuple 分组，ZM 组用 build_S_matrix_zero_momentum
    S_big = zeros(Float64, K, K)
    n_tuple_to_idxs = Dict{Tuple, Vector{Int}}()
    for (idx, (n_tup, _)) in enumerate(all_states)
        idxs = get!(Vector{Int}, n_tuple_to_idxs, n_tup)
        push!(idxs, idx)
    end

    for (n_tup, idxs) in n_tuple_to_idxs
        M_zm = count(n -> n == M_dsds(0,0,0), n_tup)
        if M_zm == N_α_dsds
            # 全 ZM: 用 build_S_matrix_zero_momentum 包含完整交换耦合
            spin_tuples = [all_states[idx][2] for idx in idxs]
            S_grp = NPHFforFVE.build_S_matrix_zero_momentum(M_zm, spin_tuples,
                [(Tuple{}(), Tuple{}())], N_α_dsds, kappa, :boson)
        else
            S_grp = NPHFforFVE.build_S_matrix(all_states[idxs], N_α_dsds, kappa, :boson)
        end
        S_big[idxs, idxs] .= S_grp
    end

    # T_diag * S
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T_dsds * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_dsds))
        T_diag[idx, idx] = T
    end

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_dsds, per_spin_r_dsds, V_can_func)
    dd = 3 * (N_α_dsds + N_α_dsds) - 6
    fv_factor = (2π * ħc_dsds / L_phys_dsds)^(dd / 2)
    H_raw = T_diag * S_big + fv_factor * V_hel

    # 广义本征值
    S_eig = eigen(Hermitian(S_big))
    good_S = S_eig.values .> 1e-12
    if all(good_S)
        evals_full = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_big))))
    else
        U = S_eig.vectors[:, good_S]
        H_red = U' * H_raw * U
        S_red = U' * S_big * U
        evals_full = sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    end
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, K=K, all_states=all_states)
end

# ============================================================================
# 验证主体代码
# ============================================================================
function _verify_dstardstar(sys, V_func, params, ref, free_Ts, label)
    # 自由能级 (全局，不区分 irrep)
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

        matched = count(ep -> minimum(abs.(ep .- ref.evals_full)) < 1e-6, evals_proj)
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
# I=1: κ=[2], 自旋对称 sign_spin = +1 (S=0,2)
# ============================================================================
function test_dstardstar_I1()
    kappa = "[2]"
    sign_spin = +1.0

    ref = _build_dstardstar_reference(kappa, sign_spin)
    println("D*D* → D*D*  I=1  κ=[2]  S=0,2  Ncut=$Ncut_dsds")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    # 全局自由动能 (从参考态提取唯一 n_tuple)
    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_dsds * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_dsds))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    println()

    ch = FockChannel("D*D*", [2], [:boson], [m_Ds_dsds], [1//1], [1//2], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_dsds, [ch], L0_dsds, a_dsds, 1//1,
                     NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES)
    params = (C0=C0_dsds,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_spin * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_dsds)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_dsds)^2; end
        ComplexF64(p.C0 * ffall * sf)
    end

    return _verify_dstardstar(sys, V_func, params, ref, free_Ts_d, "I=1")
end

# ============================================================================
# I=0: κ=[1,1], 自旋反对称 sign_spin = -1 (S=1)
# ============================================================================
function test_dstardstar_I0()
    kappa = "[1,1]"
    sign_spin = -1.0

    ref = _build_dstardstar_reference(kappa, sign_spin)
    println("D*D* → D*D*  I=0  κ=[1,1]  S=1  Ncut=$Ncut_dsds")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_dsds * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_dsds))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    println()

    ch = FockChannel("D*D*", [2], [:boson], [m_Ds_dsds], [1//1], [1//2], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_dsds, [ch], L0_dsds, a_dsds, 0//1,
                     NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES)
    params = (C0=C0_dsds,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_spin * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_dsds)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_dsds)^2; end
        ComplexF64(p.C0 * ffall * sf)
    end

    return _verify_dstardstar(sys, V_func, params, ref, free_Ts_d, "I=0")
end

# ============================================================================
# 编排
# ============================================================================
function test_dstardstar()
    ok1 = test_dstardstar_I1()
    println(repeat("=", 60))
    println()
    ok0 = test_dstardstar_I0()
    return ok1 && ok0
end

test_dstardstar()
