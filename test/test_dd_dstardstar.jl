# ==========================================================================
# DD + D*D* 耦合道体系测试 (I=1, κ="[2]")
#
# DD:   两个全同 s=0 玻色子, I=1 → κ="[2]" (空间对称)
# D*D*: 两个全同 s=1 玻色子, I=1 → κ="[2]" (空间对称, 自旋对称 S=0,2)
#
# 相互作用:
#   ⟨DD|V|DD⟩ = C_DD
#   ⟨DD|V|D*D*,σ₁,σ₂⟩ = C_mix × (−1)^{σ₁} × δ_{σ₁,−σ₂}
#   ⟨D*D*|V|D*D*⟩ = C_22 × (δ_{σ₁σ₁'}δ_{σ₂σ₂'} + δ_{σ₁σ₂'}δ_{σ₂σ₁'})
#
# 主体代码: FockSystem + build_hamiltonian_block
# 参考代码: H_raw = T·S + fv·V_hel, GEP(H_raw, S)
# 验证: eig(H_proj) ⊂ eig(H_raw, S)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_cc      = NPHFforFVE.Momentum
const m_D       = 1864.84
const m_Ds      = 2008.5
const Λ_cc      = 1000.0
const L0_cc     = 48
const a_cc      = 0.1
const L_phys_cc = L0_cc * a_cc
const ħc_cc     = 197.327
const Ncut_cc   = 20

const C_DD_cc  = 2.0e-6
const C_mix_cc = 2.0e-6
const C_22_cc  = 2.0e-6

const pref_T_cc   = (2π * ħc_cc / L_phys_cc)^2
const Λ_n²_cc     = (Λ_cc / (2π * ħc_cc / L_phys_cc))^2
const fv_factor_cc = (2π * ħc_cc / L_phys_cc)^3

const d_cc = M_cc(0,0,0)
const OH   = NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES
const kappa_sym = "[2]"

ff_cc(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_cc)^2

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
# 构造参考谱 (H_raw = T·S + fv·V_hel, GEP)
# ============================================================================
function _build_reference()
    # ---- DD 道 (species=[2], s=0) ----
    reps = NPHFforFVE.find_representatives(2; Ncut=Ncut_cc, d=d_cc,
        species=[2], particle_types=[:boson])

    all_states_DD = []
    sd_DD = Dict()
    proj_DD = []

    for rep in reps
        hel_tuple = (0.0, 0.0)
        for Gamma in OH
            result = NPHFforFVE.subspace_projection(rep, hel_tuple,
                kappa_sym, Gamma; d_total=d_cc,
                species_type=:boson, spin=0.0, etas=[1.0])
            size(result.X, 2) == 0 && continue
            for st in result.subspace_states
                if !haskey(sd_DD, st)
                    push!(all_states_DD, st)
                    sd_DD[st] = length(all_states_DD)
                end
            end
            push!(proj_DD, (X=result.X, states=result.subspace_states,
                            n_r=size(result.X, 2), Gamma=Gamma))
        end
    end
    K_DD = length(all_states_DD)

    # ---- D*D* 道 (species=[2], s=1) ----
    all_states_DsDs = []
    sd_DsDs = Dict()
    proj_DsDs = []

    for rep in reps
        M_zm = count(n -> n == d_cc, rep)
        h_reps = if M_zm > 0
            # ZM 粒子自旋非零，螺旋度无定义，直接走 ZM 管线
            [ntuple(_ -> 0.0, 2)]
        else
            NPHFforFVE.helicity_representatives(rep;
                species=[2], particle_types=[:boson],
                spins=[1.0], d=d_cc)
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))
            for Gamma in OH
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa_sym, Gamma; d_total=d_cc,
                    species_type=:boson, spin=1.0, etas=[1.0])
                size(result.X, 2) == 0 && continue
                for st in result.subspace_states
                    if !haskey(sd_DsDs, st)
                        push!(all_states_DsDs, st)
                        sd_DsDs[st] = length(all_states_DsDs)
                    end
                end
                push!(proj_DsDs, (X=result.X, states=result.subspace_states,
                                  n_r=size(result.X, 2), Gamma=Gamma))
            end
        end
    end
    K_DsDs = length(all_states_DsDs)
    K = K_DD + K_DsDs

    # ---- S 矩阵 (按 n_tuple 分组，ZM 用 build_S_matrix_zero_momentum) ----
    S_mat = zeros(Float64, K, K)

    # DD 道 (κ="[2]", boson)
    n_to_idxs_DD = Dict{Tuple, Vector{Int}}()
    for (idx, (n_tup, _)) in enumerate(all_states_DD)
        idxs = get!(Vector{Int}, n_to_idxs_DD, n_tup)
        push!(idxs, idx)
    end
    for (n_tup, idxs) in n_to_idxs_DD
        M_zm = count(n -> n == d_cc, n_tup)
        if M_zm == 2
            spin_tuples = [all_states_DD[idx][2] for idx in idxs]
            S_DD = NPHFforFVE.build_S_matrix_zero_momentum(2, spin_tuples,
                [(Tuple{}(), Tuple{}())], 2, kappa_sym, :boson)
        else
            S_DD = NPHFforFVE.build_S_matrix(all_states_DD[idxs], 2, kappa_sym, :boson)
        end
        S_mat[idxs, idxs] .= S_DD
    end

    # D*D* 道 (κ="[2]", boson)
    n_to_idxs_DsDs = Dict{Tuple, Vector{Int}}()
    for (idx, (n_tup, _)) in enumerate(all_states_DsDs)
        idxs = get!(Vector{Int}, n_to_idxs_DsDs, n_tup)
        push!(idxs, K_DD + idx)
    end
    for (n_tup, idxs) in n_to_idxs_DsDs
        M_zm = count(n -> n == d_cc, n_tup)
        if M_zm == 2
            spin_tuples = [all_states_DsDs[idx - K_DD][2] for idx in idxs]
            S_DsDs = NPHFforFVE.build_S_matrix_zero_momentum(2, spin_tuples,
                [(Tuple{}(), Tuple{}())], 2, kappa_sym, :boson)
        else
            S_DsDs = NPHFforFVE.build_S_matrix(all_states_DsDs[idxs .- K_DD], 2, kappa_sym, :boson)
        end
        S_mat[idxs, idxs] .= S_DsDs
    end

    # ---- T_diag ----
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states_DD)
        T_diag[idx, idx] = sum(sqrt(m_D^2 + pref_T_cc * Float64(sum(abs2, n_)))
                               for n_ in n_tup)
    end
    for (idx, (n_tup, _)) in enumerate(all_states_DsDs)
        T_diag[K_DD+idx, K_DD+idx] = sum(sqrt(m_Ds^2 + pref_T_cc * Float64(sum(abs2, n_)))
                                         for n_ in n_tup)
    end

    # ---- V_can 函数 ----
    function V_DD_can(np, sp, n, s, extra...)
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C_DD_cc * ff_cc(n1p)*ff_cc(n2p)*ff_cc(n1)*ff_cc(n2))
    end

    function V_DsDs_can(np, sp, n, s, extra...)
        direct   = (sp[1]==s[1] && sp[2]==s[2]) ? 1.0 : 0.0
        exchange = (sp[1]==s[2] && sp[2]==s[1]) ? 1.0 : 0.0
        spin_factor = direct + exchange
        spin_factor == 0.0 && return zero(ComplexF64)
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C_22_cc * spin_factor * ff_cc(n1p)*ff_cc(n2p)*ff_cc(n1)*ff_cc(n2))
    end

    function V_cross_can(np, sp, n, s, extra...)
        σ₁, σ₂ = Int.(s)
        σ₁ == -σ₂ || return zero(ComplexF64)
        sign_f = σ₁ == 0 ? 1.0 : -1.0
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C_mix_cc * sign_f * ff_cc(n1p)*ff_cc(n2p)*ff_cc(n1)*ff_cc(n2))
    end

    # ---- V_hel ----
    per_spin_DD   = Rational{Int}[0//1, 0//1]
    per_spin_DsDs = Rational{Int}[1//1, 1//1]

    V_hel = zeros(ComplexF64, K, K)

    V_hel[1:K_DD, 1:K_DD] .= NPHFforFVE.build_V_hel(
        all_states_DD, all_states_DD, per_spin_DD, per_spin_DD, V_DD_can)

    V_hel[K_DD+1:K, K_DD+1:K] .= NPHFforFVE.build_V_hel(
        all_states_DsDs, all_states_DsDs, per_spin_DsDs, per_spin_DsDs, V_DsDs_can)

    V_hel[1:K_DD, K_DD+1:K] .= NPHFforFVE.build_V_hel(
        all_states_DD, all_states_DsDs, per_spin_DD, per_spin_DsDs, V_cross_can)
    V_hel[K_DD+1:K, 1:K_DD] .= V_hel[1:K_DD, K_DD+1:K]'

    # ---- H_raw = T_diag * S_mat + fv * V_hel ----
    H_raw = T_diag * S_mat + fv_factor_cc * V_hel

    # ---- GEP (处理 S 半正定性) ----
    S_eig = eigen(Hermitian(S_mat))
    good_S = S_eig.values .> 1e-12
    evals_full = if all(good_S)
        sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_mat))))
    else
        U = S_eig.vectors[:, good_S]
        H_red = U' * H_raw * U
        S_red = U' * S_mat * U
        sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    end
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full = evals_full,
            all_states_DD = all_states_DD, sd_DD = sd_DD, proj_DD = proj_DD, K_DD = K_DD,
            all_states_DsDs = all_states_DsDs, sd_DsDs = sd_DsDs, proj_DsDs = proj_DsDs, K_DsDs = K_DsDs,
            K = K)
end

# ============================================================================
# 测试主体
# ============================================================================
function test_dd_dstardstar()
    println("DD + D*D* 耦合道  I=1  κ=[2]  Ncut=$Ncut_cc")
    println()

    ref = _build_reference()
    println("K = $(ref.K) (DD:$(ref.K_DD), D*D*:$(ref.K_DsDs)),  ref eigenvalues = $(length(ref.evals_full))")

    # 全局自由动能 (DD + D*D* 所有唯一 n_tuple，按各道质量)
    free_Ts = Float64[]
    for (nt, _) in ref.all_states_DD
        push!(free_Ts, sum(sqrt(m_D^2 + pref_T_cc * Float64(sum(abs2, n_))) for n_ in nt))
    end
    for (nt, _) in ref.all_states_DsDs
        push!(free_Ts, sum(sqrt(m_Ds^2 + pref_T_cc * Float64(sum(abs2, n_))) for n_ in nt))
    end
    free_Ts_d = distinct_levels(sort(unique!(free_Ts)), 10)
    println("  自由能级 (前10非简并): $(round.(free_Ts_d, digits=4))")
    println()

    # ---- FockSystem ----
    ch_DD = FockChannel("DD", [2], [:boson], [m_D],
                         [0//1], [1//2], [1.0], NPHFforFVE.relativistic)
    ch_DsDs = FockChannel("D*D*", [2], [:boson], [m_Ds],
                           [1//1], [1//2], [1.0], NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_cc, [ch_DD, ch_DsDs], L0_cc, a_cc, 1//1, OH)
    params = (C_DD=C_DD_cc, C_mix=C_mix_cc, C_22=C_22_cc)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        ff_nA = prod(n_ -> 1.0 / (1.0 + Float64(sum(abs2, n_)) / (Λ_cc/(2π*ħc_cc/L_phys))^2)^2, nA)
        ff_nB = prod(n_ -> 1.0 / (1.0 + Float64(sum(abs2, n_)) / (Λ_cc/(2π*ħc_cc/L_phys))^2)^2, nB)
        ffall = ff_nA * ff_nB

        if chA == 1 && chB == 1
            # DD ↔ DD (channel 1)
            return ComplexF64(p.C_DD * ffall)
        elseif chA == 2 && chB == 2
            # D*D* ↔ D*D* (channel 2)
            direct   = (sp[1]==s[1] && sp[2]==s[2]) ? 1.0 : 0.0
            exchange = (sp[1]==s[2] && sp[2]==s[1]) ? 1.0 : 0.0
            spin_factor = direct + exchange
            spin_factor == 0.0 && return zero(ComplexF64)
            return ComplexF64(p.C_22 * spin_factor * ffall)
        else
            # DD ↔ D*D* (cross)
            σ = chA == 1 ? s : sp
            σ₁, σ₂ = Int(σ[1]), Int(σ[2])
            σ₁ == -σ₂ || return zero(ComplexF64)
            sign_f = σ₁ == 0 ? 1.0 : -1.0
            return ComplexF64(p.C_mix * sign_f * ffall)
        end
    end

    all_ok = true
    for Gamma in OH
        H_proj = NPHFforFVE.build_hamiltonian_block(sys, Gamma, V_func, params)
        dim = size(H_proj, 1)
        dim == 0 && continue

        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_proj = evals_proj[isfinite.(evals_proj)]

        di = distinct_levels(evals_proj, 5)

        matched = count(ep -> minimum(abs.(ep .- ref.evals_full)) < 1e-8, evals_proj)
        ok = matched == length(evals_proj)
        all_ok = all_ok && ok
        println("  $Gamma: dim=$dim, matched=$matched/$(length(evals_proj))  $(ok ? "✓" : "✗")  [$(round.(di, digits=4))]")
        @test ok
    end

    println()
    println(all_ok ? "全部通过 ✓" : "存在失败 ✗")
    return all_ok
end

test_dd_dstardstar()
