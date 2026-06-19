# ==========================================================================
# D D* → D D* 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw)
#   验证: eig(H_proj) ⊂ evals_full
#
# D: s=0, I=1/2, boson, η=+1, m=1864.84 MeV
# D*: s=1, I=1/2, boson, η=+1, m=2008.5 MeV
# 可区分粒子 → species=[1,1], κ=("[1]","[1]"), S=I
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_dds      = NPHFforFVE.Momentum
const m_D_dds    = 1864.84
const m_Ds_dds   = 2008.5
const Λ_dds      = 1000.0
const L0_dds     = 48
const a_dds      = 0.1
const L_phys_dds = L0_dds * a_dds
const ħc_dds     = 197.327
const Ncut_dds   = 20
const C0_dds     = 5.0e-7

const pref_T_dds    = (2π * ħc_dds / L_phys_dds)^2
const Λ_n²_dds      = (Λ_dds / (2π * ħc_dds / L_phys_dds))^2
const fv_factor_dds  = (2π * ħc_dds / L_phys_dds)^3
const d_dds = M_dds(0,0,0)

const species_dds    = [1, 1]
const particle_dds   = [:boson, :boson]
const N_α_dds        = 2
const per_mass_dds   = [m_D_dds, m_Ds_dds]
const per_spin_r_dds = Rational{Int}[0//1, 1//1]
const spins_dds      = Float64[0.0, 1.0]
const etas_dds       = Float64[1.0, 1.0]
const kappa_dds      = ("[1]", "[1]")

const irrep_names_dds = NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES

ff_dds(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_dds)^2

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
function _build_ddstar_reference()
    function V_can_func(np, sp, n, s, extra...)
        σ1p, σ2p = sp; σ1, σ2 = s
        σ1p == σ1 || return zero(ComplexF64)
        σ2p == σ2 || return zero(ComplexF64)
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C0_dds * ff_dds(n1p)*ff_dds(n1) * ff_dds(n2p)*ff_dds(n2))
    end

    reps = NPHFforFVE.find_representatives(N_α_dds; Ncut=Ncut_dds, d=d_dds,
        species=species_dds, particle_types=particle_dds)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_dds, particle_types=particle_dds,
                spins=spins_dds, d=d_dds)
        catch
            [ntuple(_ -> 0.0, N_α_dds)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names_dds
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa_dds, Gamma;
                    d_total=d_dds, species=species_dds,
                    particle_types=particle_dds,
                    spins=spins_dds, etas=etas_dds)
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

    # T_diag (S=I)
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T_dds * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_dds))
        T_diag[idx, idx] = T
    end

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_dds, per_spin_r_dds, V_can_func)
    H_raw = T_diag + fv_factor_dds * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, K=K, all_states=all_states)
end

# ============================================================================
# 测试主体
# ============================================================================
function test_ddstar()
    ref = _build_ddstar_reference()

    println("D D* → D D*  I=0  Ncut=$Ncut_dds")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_dds * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_dds))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    println("自由能级 (前10非简并): $(round.(free_Ts_d, digits=4))")
    println()

    ch = FockChannel("DD*", [1,1], [:boson, :boson],
                     [m_D_dds, m_Ds_dds], [0//1, 1//1], [1//2, 1//2],
                     [1.0, 1.0], NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_dds, [ch], L0_dds, a_dds, 0//1,
                     irrep_names_dds)
    params = (C0=C0_dds,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        σ1p, σ2p = sp; σ1, σ2 = s
        σ1p == σ1 || return zero(ComplexF64)
        σ2p == σ2 || return zero(ComplexF64)
        n1p, n2p = nA; n1, n2 = nB
        ffall = ff_dds(n1p)*ff_dds(n1) * ff_dds(n2p)*ff_dds(n2)
        ComplexF64(p.C0 * ffall)
    end

    all_ok = true
    for Gamma in irrep_names_dds
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

test_ddstar()
