# ==========================================================================
# NN → NN 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw, S)
#   验证: eig(H_proj) ⊂ evals_full
#
# N: s=1/2, I=1/2, fermion, η=+1, m=938.92 MeV
# 全同费米子 × 2
#
# I=1 (对称) + κ=[2] (对称) → S=0 (反对称): sign_ex = -1
# I=0 (反对称) + κ=[1,1] (反对称) → S=1 (对称): sign_ex = +1
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_nn      = NPHFforFVE.Momentum
const m_N_nn    = 938.92
const Λ_nn      = 1000.0
const L0_nn     = 48
const a_nn      = 0.1
const L_phys_nn = L0_nn * a_nn
const ħc_nn     = 197.327
const Ncut_nn   = 20

const pref_T_nn   = (2π * ħc_nn / L_phys_nn)^2
const Λ_n²_nn     = (Λ_nn / (2π * ħc_nn / L_phys_nn))^2
const d_nn        = M_nn(0,0,0)

const species_nn    = [2]
const particle_nn   = [:fermion]
const N_α_nn        = 2
const per_mass_nn   = [m_N_nn, m_N_nn]
const per_spin_r_nn = Rational{Int}[1//2, 1//2]
const spin_val_nn   = 0.5
const etas_nn       = [1.0]

ff_nn(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_nn)^2

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
# 通用辅助: 构造参考谱
# ============================================================================
function _build_nn_reference(kappa, sign_ex, C0, I_label)
    function V_can_func(np, sp, n, s, extra...)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in np; ffall *= ff_nn(n_); end
        for n_ in n;  ffall *= ff_nn(n_); end
        ComplexF64(C0 * ffall * sf)
    end

    reps = NPHFforFVE.find_representatives(N_α_nn; Ncut=Ncut_nn, d=d_nn,
        species=species_nn, particle_types=particle_nn)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        M_zm = count(n -> n == d_nn, rep)
        h_reps = if M_zm > 0
            [ntuple(_ -> 0.0, N_α_nn)]
        else
            NPHFforFVE.helicity_representatives(rep;
                species=species_nn, particle_types=particle_nn,
                spins=[spin_val_nn], d=d_nn)
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa, Gamma;
                    d_total=d_nn, species_type=:fermion,
                    spin=spin_val_nn, etas=etas_nn)
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
        M_zm = count(n -> n == d_nn, n_tup)
        if M_zm == N_α_nn
            spin_tuples = [all_states[idx][2] for idx in idxs]
            S_grp = NPHFforFVE.build_S_matrix_zero_momentum(M_zm, spin_tuples,
                [(Tuple{}(), Tuple{}())], N_α_nn, kappa, :fermion)
        else
            S_grp = NPHFforFVE.build_S_matrix(all_states[idxs], N_α_nn, kappa, :fermion)
        end
        S_big[idxs, idxs] .= S_grp
    end

    # T_diag * S
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T_nn * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_nn))
        T_diag[idx, idx] = T
    end

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_nn, per_spin_r_nn, V_can_func)
    dd = 3 * (N_α_nn + N_α_nn) - 6
    fv_factor = (2π * ħc_nn / L_phys_nn)^(dd / 2)
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
    evals_full = evals_full[isfinite.(evals_full) .&& evals_full .> 100]

    return (evals_full=evals_full, K=K, all_states=all_states)
end

# ============================================================================
# 通用辅助: 验证主体代码
# ============================================================================
function _verify_nn(sys, V_func, params, ref, free_Ts, label)
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
# I=1: κ=[2], S=0 (sign_ex = -1)
# ============================================================================
function test_nn_I1()
    C0 = 15.0e-6
    sign_ex = -1.0

    ref = _build_nn_reference("[2]", sign_ex, C0, "I=1")
    println("NN → NN  I=1  κ=[2]  S=0  Ncut=$Ncut_nn")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")
    println()

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_nn * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_nn))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    ch = FockChannel("NN", [2], [:fermion], [m_N_nn], [1//2], [1//2], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_nn, [ch], L0_nn, a_nn, 1//1,
                     NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES)
    params = (C0=C0,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_nn)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_nn)^2; end
        ComplexF64(p.C0 * ffall * sf)
    end

    return _verify_nn(sys, V_func, params, ref, free_Ts_d, "I=1")
end

# ============================================================================
# I=0: κ=[1,1], S=1 (sign_ex = +1)
# ============================================================================
function test_nn_I0()
    C0 = 15.0e-6
    sign_ex = +1.0

    ref = _build_nn_reference("[1,1]", sign_ex, C0, "I=0")
    println("NN → NN  I=0  κ=[1,1]  S=1  Ncut=$Ncut_nn")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")
    println()

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_nn * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_nn))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)

    ch = FockChannel("NN", [2], [:fermion], [m_N_nn], [1//2], [1//2], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_nn, [ch], L0_nn, a_nn, 0//1,
                     NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES)
    params = (C0=C0,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_nn)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_nn)^2; end
        ComplexF64(p.C0 * ffall * sf)
    end

    return _verify_nn(sys, V_func, params, ref, free_Ts_d, "I=0")
end

# ============================================================================
# 编排
# ============================================================================
function test_nn()
    ok1 = test_nn_I1()
    println(repeat("=", 60))
    println()
    ok0 = test_nn_I0()
    return ok1 && ok0
end

test_nn()
