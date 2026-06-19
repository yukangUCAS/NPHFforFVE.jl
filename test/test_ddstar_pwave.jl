# ==========================================================================
# D D* → D D* p-波 测试 (主体代码端到端验证)
#
# 使用 FockSystem + build_hamiltonian_block 验证用户调用的主体代码正确性。
# 对每个不可约表示 Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw)
#   验证: eig(H_proj) ⊂ evals_full
#
# D: s=0, I=1/2, boson, η=+1, m=1864.84 MeV
# D*: s=1, I=1/2, boson, η=+1, m=2008.50 MeV
# 可区分粒子 → species=[1,1], κ=("[1]","[1]"), S=I
#
# p-波作用: V ∝ p·p' × Y_{1,-σ'}(p̂') × Y_{1,-σ}(p̂)* × ff
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_pw      = NPHFforFVE.Momentum
const m_D_pw    = 1864.84
const m_Ds_pw   = 2008.50
const Λ_pw      = 1000.0
const L0_pw     = 48
const a_pw      = 0.1
const L_phys_pw = L0_pw * a_pw
const ħc_pw     = 197.327
const Ncut_pw   = 20
const C0_pw     = 1.5e-12

const pref_T_pw    = (2π * ħc_pw / L_phys_pw)^2
const Λ_n²_pw      = (Λ_pw / (2π * ħc_pw / L_phys_pw))^2
const fv_factor_pw  = (2π * ħc_pw / L_phys_pw)^3
const d_pw = M_pw(0,0,0)

const species_pw    = [1, 1]
const particle_pw   = [:boson, :boson]
const N_α_pw        = 2
const per_mass_pw   = [m_D_pw, m_Ds_pw]
const per_spin_r_pw = Rational{Int}[0//1, 1//1]
const spins_pw      = Float64[0.0, 1.0]
const etas_pw       = Float64[1.0, 1.0]
const kappa_pw      = ("[1]", "[1]")

const irrep_names_pw = NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES

ff_pw(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_pw)^2

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

# f_m(n) = |n| * Y_{1,m}(n̂)
function fm_poly(m::Int, n::Momentum)
    nx, ny, nz = Float64(n[1]), Float64(n[2]), Float64(n[3])
    if m == 0
        return sqrt(3/(4π)) * nz
    elseif m == 1
        return -sqrt(3/(8π)) * (nx + im*ny)
    elseif m == -1
        return sqrt(3/(8π)) * (nx - im*ny)
    else
        return 0.0
    end
end

# ============================================================================
# 构造参考谱
# ============================================================================
function _build_ddstar_pwave_reference()
    function V_can_func(np, sp, n, s, extra...)
        sp[1] == 0 && s[1] == 0 || return zero(ComplexF64)
        nDsp, nDs = np[2], n[2]
        σp, σ = Int(sp[2]), Int(s[2])
        val = pref_T_pw * fm_poly(-σp, nDsp) * conj(fm_poly(-σ, nDs))
        iszero(val) && return zero(ComplexF64)
        ffall = 1.0
        for n_ in np; ffall *= ff_pw(n_); end
        for n_ in n;  ffall *= ff_pw(n_); end
        ComplexF64(C0_pw * ffall * (-1.0)^(σp+σ) / 3 * val)
    end

    reps = NPHFforFVE.find_representatives(N_α_pw; Ncut=Ncut_pw, d=d_pw,
        species=species_pw, particle_types=particle_pw)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_pw, particle_types=particle_pw,
                spins=spins_pw, d=d_pw)
        catch
            [ntuple(_ -> 0.0, N_α_pw)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names_pw
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa_pw, Gamma;
                    d_total=d_pw, species=species_pw,
                    particle_types=particle_pw,
                    spins=spins_pw, etas=etas_pw)
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
        T = sum(sqrt(m_^2 + pref_T_pw * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_pw))
        T_diag[idx, idx] = T
    end

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_pw, per_spin_r_pw, V_can_func)
    H_raw = T_diag + fv_factor_pw * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, K=K, all_states=all_states)
end

# ============================================================================
# 测试主体
# ============================================================================
function test_ddstar_pwave()
    ref = _build_ddstar_pwave_reference()

    println("D D* → D D*  p-波  I=0  Ncut=$Ncut_pw")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_pw * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_pw))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    println("自由能级 (前10非简并): $(round.(free_Ts_d, digits=4))")
    println()

    ch = FockChannel("DD*", [1,1], [:boson, :boson],
                     [m_D_pw, m_Ds_pw], [0//1, 1//1], [1//2, 1//2],
                     [1.0, 1.0], NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_pw, [ch], L0_pw, a_pw, 0//1,
                     irrep_names_pw)
    params = (C0=C0_pw,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p)
        sp[1] == 0 && s[1] == 0 || return zero(ComplexF64)
        nDsp, nDs = nA[2], nB[2]
        σp, σ = Int(sp[2]), Int(s[2])
        pref = (2π * ħc_pw / L_phys)^2
        val = pref * fm_poly(-σp, nDsp) * conj(fm_poly(-σ, nDs))
        iszero(val) && return zero(ComplexF64)
        ffall = 1.0
        for n_ in nA; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_pw)^2; end
        for n_ in nB; ffall /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²_pw)^2; end
        ComplexF64(p.C0 * ffall * (-1.0)^(σp+σ) / 3 * val)
    end

    all_ok = true
    for Gamma in irrep_names_pw
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

test_ddstar_pwave()
