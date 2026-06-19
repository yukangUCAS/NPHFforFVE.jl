# ==========================================================================
# ππ → ππ 运动系测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个 (d_total, Γ):
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   验证: eig(H_proj) ⊂ evals_full (参考谱)
#
# ππ: 自旋 0 全同玻色子, I=2, κ=[2]
# 接触势 V = C0 * cutoff (CM-frame boost)
# 运动系: D001 (C4v), D011 (C2v), D111 (C3v)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const SG          = NPHFforFVE.SymmetryGroup
const M_pp        = NPHFforFVE.Momentum
const m_π         = 140.0
const L0_pp       = 48
const a_pp        = 0.1
const L_phys_pp   = L0_pp * a_pp
const hc_pp       = 197.327
const C0_pp       = 0.5
const Lambda_pp   = 1000.0
const pv_pp       = 2π * hc_pp / L_phys_pp
const Lambda2_pp  = Lambda_pp^2
const per_mass_pp = [m_π, m_π]
const per_spin_pp = Rational{Int}[0//1, 0//1]
const spin_pp     = 0.0
const etas_pp     = Float64[1.0, 1.0]

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
# 参考谱构造 (含 ZM 态，与主代码一致)
# ============================================================================
function _build_pipi_moving_reference(d_total, irrep_names, Ncut)
    reps = NPHFforFVE.find_representatives(2; Ncut=Ncut, d=d_total,
        species=[2], particle_types=[:boson])

    all_states = []
    state_to_idx = Dict()

    for rep in reps
        # spin-0: 只有全零螺旋度
        hel_configs = [ntuple(_ -> 0.0, 2)]
        for hel in hel_configs, Gamma in irrep_names
            result = NPHFforFVE.subspace_projection(rep, hel, "[2]", Gamma;
                d_total=d_total, species_type=:boson, spin=spin_pp, etas=etas_pp)
            size(result.X, 2) == 0 && continue
            for st in result.subspace_states
                if !haskey(state_to_idx, st)
                    push!(all_states, st)
                    state_to_idx[st] = length(all_states)
                end
            end
        end
    end

    K = length(all_states)
    K == 0 && return Float64[]

    # 度规 S (全同玻色子: direct + exchange)
    S_mat = NPHFforFVE.build_S_matrix(all_states, 2, "[2]", :boson)

    # 动能 (CM-frame relativistic, 与主代码 _kinetic_energy_rep 一致)
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        p_mov = [pv_pp .* Float64.(n_) for n_ in n_tup]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass_pp, d_total, L_phys_pp)
        T = sum(sqrt(m^2 + Float64(sum(abs2, p_))) for (p_, m) in zip(p_cm, per_mass_pp))
        T_diag[idx, idx] = T
    end

    # V_func
    function V_can(np, sp, n, s, extra...)
        p_mov = [pv_pp .* Float64.(ni) for ni in np]
        k_mov = [pv_pp .* Float64.(ni) for ni in n]
        p_cm, fb = NPHFforFVE.boost_to_cm(p_mov, per_mass_pp, d_total, L_phys_pp)
        k_cm, fk = NPHFforFVE.boost_to_cm(k_mov, per_mass_pp, d_total, L_phys_pp)
        cutoff = 1.0
        for pc in p_cm; cutoff /= (1.0 + Float64(sum(abs2, pc)) / Lambda2_pp)^2; end
        for pc in k_cm; cutoff /= (1.0 + Float64(sum(abs2, pc)) / Lambda2_pp)^2; end
        ComplexF64(fb * C0_pp * cutoff * fk)
    end

    V_hel = NPHFforFVE.build_V_hel(all_states, all_states, per_spin_pp, per_spin_pp, V_can)
    dd = 3 * (2 + 2) - 6
    fv = (2π * hc_pp / L_phys_pp)^(dd / 2)
    H_raw = T_diag * S_mat + fv * V_hel

    S_evals = eigen(Hermitian(S_mat)).values
    good_S = S_evals .> 1e-12
    if all(good_S)
        evals_full = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_mat))))
    else
        U = eigen(Hermitian(S_mat)).vectors[:, good_S]
        H_red = U' * H_raw * U
        S_red = U' * S_mat * U
        evals_full = sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    end
    return (evals_full=evals_full[isfinite.(evals_full)], all_states=all_states)
end

# ============================================================================
# 主代码 V_func
# ============================================================================
function _make_V_func_pipi(d_total)
    return function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        pv = 2π * hc_pp / L_phys
        pA_mov = [pv .* Float64.(n) for n in nA]
        pB_mov = [pv .* Float64.(n) for n in nB]
        pA, facA = NPHFforFVE.boost_to_cm(pA_mov, per_mass_pp, d_total, L_phys)
        pB, facB = NPHFforFVE.boost_to_cm(pB_mov, per_mass_pp, d_total, L_phys)
        cutoff = 1.0
        for pc in pA; cutoff /= (1.0 + Float64(sum(abs2, pc)) / Lambda2_pp)^2; end
        for pc in pB; cutoff /= (1.0 + Float64(sum(abs2, pc)) / Lambda2_pp)^2; end
        ComplexF64(facA * p.C0 * cutoff * facB)
    end
end

# ============================================================================
# 验证函数
# ============================================================================
function _verify_pipi_moving(sys, V_func, params, ref, free_Ts, label)
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
        matched = count(ep -> minimum(abs.(ep .- ref)) < 1e-6, evals_proj)
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
# D001 (C4v)  Ncut=4
# ============================================================================
function _compute_pp_moving_free_Ts(all_states, d_total)
    free_Ts = Float64[]
    for (nt, _) in all_states
        p_mov = [pv_pp .* Float64.(n_) for n_ in nt]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass_pp, d_total, L_phys_pp)
        T = sum(sqrt(m^2 + Float64(sum(abs2, p_))) for (p_, m) in zip(p_cm, per_mass_pp))
        push!(free_Ts, T)
    end
    sort(unique!(free_Ts))
end

function test_pipi_D001()
    d_total = NPHFforFVE.D001
    irrep_names = SG.C4V_BOSONIC_NAMES
    Ncut = 4
    println("ππ  D001 (C4v)  I=2  Ncut=$Ncut")
    ref = _build_pipi_moving_reference(d_total, irrep_names, Ncut)
    free_Ts = distinct_levels(_compute_pp_moving_free_Ts(ref.all_states, d_total), 10)
    println("reference eigenvalues = $(length(ref.evals_full))")

    ch = FockChannel("pipi", [2], [:boson], [m_π], [0//1], [1//1], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(d_total, Ncut, [ch], L0_pp, a_pp, 2//1, irrep_names)
    params = (C0=C0_pp,)

    return _verify_pipi_moving(sys, _make_V_func_pipi(d_total), params,
                               ref.evals_full, free_Ts, "D001")
end

# ============================================================================
# D011 (C2v)  Ncut=4
# ============================================================================
function test_pipi_D011()
    d_total = NPHFforFVE.D011
    irrep_names = SG.C2V_BOSONIC_NAMES
    Ncut = 4
    println("\n$(repeat("=", 60))")
    println("ππ  D011 (C2v)  I=2  Ncut=$Ncut")
    ref = _build_pipi_moving_reference(d_total, irrep_names, Ncut)
    free_Ts = distinct_levels(_compute_pp_moving_free_Ts(ref.all_states, d_total), 10)
    println("reference eigenvalues = $(length(ref.evals_full))")

    ch = FockChannel("pipi", [2], [:boson], [m_π], [0//1], [1//1], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(d_total, Ncut, [ch], L0_pp, a_pp, 2//1, irrep_names)
    params = (C0=C0_pp,)

    return _verify_pipi_moving(sys, _make_V_func_pipi(d_total), params,
                               ref.evals_full, free_Ts, "D011")
end

# ============================================================================
# D111 (C3v)  Ncut=6
# ============================================================================
function test_pipi_D111()
    d_total = NPHFforFVE.D111
    irrep_names = SG.C3V_BOSONIC_NAMES
    Ncut = 6
    println("\n$(repeat("=", 60))")
    println("ππ  D111 (C3v)  I=2  Ncut=$Ncut")
    ref = _build_pipi_moving_reference(d_total, irrep_names, Ncut)
    free_Ts = distinct_levels(_compute_pp_moving_free_Ts(ref.all_states, d_total), 10)
    println("reference eigenvalues = $(length(ref.evals_full))")

    ch = FockChannel("pipi", [2], [:boson], [m_π], [0//1], [1//1], [1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(d_total, Ncut, [ch], L0_pp, a_pp, 2//1, irrep_names)
    params = (C0=C0_pp,)

    return _verify_pipi_moving(sys, _make_V_func_pipi(d_total), params,
                               ref.evals_full, free_Ts, "D111")
end

# ============================================================================
# 编排
# ============================================================================
function test_pipi_moving()
    ok1 = test_pipi_D001()
    ok2 = test_pipi_D011()
    ok3 = test_pipi_D111()
    return ok1 && ok2 && ok3
end

test_pipi_moving()
