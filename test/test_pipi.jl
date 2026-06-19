# ==========================================================================
# ππ → ππ 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw, S)
#   验证: eig(H_proj) ⊂ evals_full
#
# π: s=0, I=1, boson, η=-1, m=140.0 MeV
# 全同玻色子 × 2, I=2 → κ=[2]
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_pipi      = NPHFforFVE.Momentum
const m_π_pipi    = 140.0
const Λ_pipi      = 1000.0
const L0_pipi     = 48
const a_pipi      = 0.1
const L_phys_pipi = L0_pipi * a_pipi
const ħc_pipi     = 197.327
const Ncut_pipi   = 4

const pref_T_pipi  = (2π * ħc_pipi / L_phys_pipi)^2
const Λ_n²_pipi    = (Λ_pipi / (2π * ħc_pipi / L_phys_pipi))^2
const d_pipi = M_pipi(0,0,0)

const species_pipi    = [2]
const particle_pipi   = [:boson]
const N_α_pipi        = 2
const per_mass_pipi   = [m_π_pipi, m_π_pipi]
const per_spin_r_pipi = Rational{Int}[0//1, 0//1]
const spin_val_pipi   = 0.0
const etas_pipi       = [-1.0]

ff_pipi(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_pipi)^2

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
# 测试主体
# ============================================================================
function test_pipi()
    C0 = 0.5

    # ---- V_can 供参考谱用 ----
    function V_can_func(np, sp, n, s, extra...)
        ffall = 1.0
        for n_ in np; ffall *= ff_pipi(n_); end
        for n_ in n;  ffall *= ff_pipi(n_); end
        ComplexF64(C0 * ffall)
    end

    # ---- 参考谱: 底层全空间枚举 ----
    reps = NPHFforFVE.find_representatives(N_α_pipi; Ncut=Ncut_pipi, d=d_pipi,
        species=species_pipi, particle_types=particle_pipi)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_pipi, particle_types=particle_pipi,
                spins=[spin_val_pipi], d=d_pipi)
        catch
            [ntuple(_ -> 0.0, N_α_pipi)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    "[2]", Gamma;
                    d_total=d_pipi, species_type=:boson,
                    spin=spin_val_pipi, etas=etas_pipi)
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
    println("ππ → ππ  I=2  κ=[2]  Ncut=$Ncut_pipi")
    println("K = $K states")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_pipi * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_pipi))
        for (nt, _) in all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    println("自由能级 (前10非简并): $(round.(free_Ts_d, digits=4))")

    # S matrix
    S_big = NPHFforFVE.build_S_matrix(all_states, N_α_pipi, "[2]", :boson)

    # T_diag
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T_pipi * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_pipi))
        T_diag[idx, idx] = T
    end
    T_raw = T_diag * S_big

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_pipi, per_spin_r_pipi, V_can_func)

    # dd = 3*(N_α+N_α)-6 = 3*4-6 = 6 ... wait, for two-body it's 0
    # Actually: N_α=2, N_β=2, dd = 3*(2+2)-6 = 6, fv_factor = (2π/L)^3
    # But wait, the original used dd = 3*(N_α+N_α)-6 = 6, fv = (2π/L)^(6/2)
    # Let me use the same as original: dd = 3*(N_α+N_α)-6, which for 2-body is...
    # Actually the original code says: N_α = 2; dd = 3*(N_α+N_α)-6 = 3*4-6 = 6
    # fv_factor = (2π/L_phys)^(dd/2) = (2π/L_phys)^3
    dd = 3*(N_α_pipi + N_α_pipi) - 6
    fv_factor = (2π * ħc_pipi / L_phys_pipi)^(dd / 2)

    H_raw = T_raw + fv_factor * V_hel

    # 参考本征值
    S_eig = eigen(Hermitian(S_big))
    good_S = S_eig.values .> 1e-12
    U = S_eig.vectors[:, good_S]
    H_red = U' * H_raw * U
    S_red = U' * S_big * U
    evals_full = sort(real.(eigvals(Hermitian(H_red), Hermitian(S_red))))
    evals_full = evals_full[isfinite.(evals_full)]
    println("reference eigenvalues = $(length(evals_full))")
    println()

    # ---- 主体: FockSystem + build_hamiltonian_block ----
    ch = FockChannel("ππ", [2], [:boson], [m_π_pipi], [0//1], [1//1], [-1.0],
                     NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_pipi, [ch], L0_pipi, a_pipi, 2//1,
                     NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES)
    params = (C0=C0,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_pipi)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_pipi)^2; end
        ComplexF64(p.C0 * ffall)
    end

    all_ok = true
    for Gamma in sys.selected_irreps
        H_proj = NPHFforFVE.build_hamiltonian_block(sys, Gamma, V_func, params)
        dim_total = size(H_proj, 1)
        dim_total == 0 && continue

        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_proj = evals_proj[isfinite.(evals_proj)]

        di = distinct_levels(evals_proj, 5)
        matched = count(ep -> minimum(abs.(ep .- evals_full)) < 1e-6, evals_proj)
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

test_pipi()
