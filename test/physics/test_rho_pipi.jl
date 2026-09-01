# ==========================================================================
# ρ → ππ coupled channelstest (end-to-end validation of the main code)
#
# Use `FockSystem` and `build_hamiltonian_block` to validate the main public calculation path.
# For every irrep Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw)
#   Check: eig(H_proj) ⊂ evals_full
#
# ρ: s=1, η=-1, m=800 MeV, N=1
# ππ: s=0 each, η=+1, m=140 MeV, N=2, I=1
# Coupling: ⟨ππ,k|V|ρ,σ⟩ = g * (2πħc/L) * fm_poly(σ, k)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_rp      = NPHFforFVE.Momentum
const m_ρ_rp    = 800.0
const m_π_rp    = 140.0
const g_rp      = 1.43e-5
const L0_rp     = 48
const a_rp      = 0.1
const L_phys_rp = L0_rp * a_rp
const ħc_rp     = 197.327
const Ncut_rp   = 20

const pref_T_rp  = (2π * ħc_rp / L_phys_rp)^2
const d_rp       = M_rp(0,0,0)
const irrep_names_rp = NPHFforFVE.SymmetryGroup.OH_IRREP_NAMES

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

# f_m polynomial
function fm_rp(m::Int, n::Momentum)
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
# Construct the reference spectrum (each irrep is computed separately)
# ============================================================================
function _build_rho_pipi_reference_per_irrep(Gamma)
    ch_rho = FockChannel("rho", [1], [:boson], [m_ρ_rp],
                         [1//1], [1//1], [-1.0], relativistic)
    ch_pipi = FockChannel("pipi", [2], [:boson], [m_π_rp],
                          [0//1], [1//1], [1.0], relativistic)

    sub_rho = NPHFforFVE.get_isospin_subchannels(ch_rho, 1//1)[1]
    sub_pipi = NPHFforFVE.get_isospin_subchannels(ch_pipi, 1//1)[1]

    _, etas_rho = NPHFforFVE._expand_per_particle(ch_rho)
    _, etas_pipi = NPHFforFVE._expand_per_particle(ch_pipi)
    per_spin_rho = Rational{Int}[1//1]
    per_spin_pipi = Rational{Int}[0//1, 0//1]
    per_mass_rho = NPHFforFVE._expand_per_particle_mass(ch_rho)
    per_mass_pipi = NPHFforFVE._expand_per_particle_mass(ch_pipi)

    spin_rho = 1.0; spin_pipi = 0.0

    projs_rho = try
        NPHFforFVE._get_channel_proj_list(
            ch_rho, Ncut_rp, d_rp, sub_rho.κ, Gamma, spin_rho, Float64.(etas_rho))
    catch; []; end

    projs_pipi = try
        NPHFforFVE._get_channel_proj_list(
            ch_pipi, Ncut_rp, d_rp, sub_pipi.κ, Gamma, spin_pipi, Float64.(etas_pipi))
    catch; []; end

    dim_rho = sum(p.n_r for p in projs_rho; init=0)
    dim_pipi = sum(p.n_r for p in projs_pipi; init=0)
    dim_rho == 0 && dim_pipi == 0 && return nothing

    # collect states
    rho_states = []; rho_to_idx = Dict()
    for p in projs_rho
        for st in p.states
            haskey(rho_to_idx, st) || (push!(rho_states, st); rho_to_idx[st] = length(rho_states))
        end
    end
    Kr = length(rho_states)

    pipi_states = []; pipi_to_idx = Dict()
    for p in projs_pipi
        for st in p.states
            haskey(pipi_to_idx, st) || (push!(pipi_states, st); pipi_to_idx[st] = length(pipi_states))
        end
    end
    Kp = length(pipi_states)

    K = Kr + Kp
    all_states = [rho_states..., pipi_states...]

    # T_diag
    T_diag = zeros(ComplexF64, K, K)
    for (idx, st) in enumerate(all_states)
        n_tup, _ = st
        m_arr = idx <= Kr ? per_mass_rho : per_mass_pipi
        T = sum(sqrt(m_^2 + pref_T_rp * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, m_arr))
        T_diag[idx, idx] = T
    end

    # V_can for build_V_hel
    function V_can_func(np, sp, n, s, extra...)
        if length(n) == 1 && length(np) == 2
            k, σ = np[1], Int(s[1])
            return ComplexF64(g_rp * (2π * ħc_rp / L_phys_rp) * fm_rp(σ, k))
        elseif length(n) == 2 && length(np) == 1
            k, σ = n[1], Int(sp[1])
            return ComplexF64(g_rp * (2π * ħc_rp / L_phys_rp) * conj(fm_rp(σ, k)))
        end
        return zero(ComplexF64)
    end

    # V_hel full space
    V_hel = zeros(ComplexF64, K, K)
    if Kr > 0 && Kp > 0
        V_hel[Kr+1:end, 1:Kr] .= NPHFforFVE.build_V_hel(
            pipi_states, rho_states, per_spin_pipi, per_spin_rho, V_can_func)
        V_hel[1:Kr, Kr+1:end] .= NPHFforFVE.build_V_hel(
            rho_states, pipi_states, per_spin_rho, per_spin_pipi, V_can_func)
    end

    # FV factor: dd = 3*(N_α+N_β)-6
    d_cross = 3*(1+2)-6; fv_cross = (2π * ħc_rp / L_phys_rp)^(d_cross/2)
    d_pipi = 3*(2+2)-6; fv_pipi = (2π * ħc_rp / L_phys_rp)^(d_pipi/2)
    Fv = ones(ComplexF64, K, K)
    Fv[1:Kr, 1:Kr] .= 1.0
    Fv[Kr+1:end, Kr+1:end] .= fv_pipi
    Fv[1:Kr, Kr+1:end] .= fv_cross
    Fv[Kr+1:end, 1:Kr] .= fv_cross

    H_raw = T_diag + Fv .* V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, dim_rho=dim_rho, dim_pipi=dim_pipi, K=K, all_states=all_states)
end

# ============================================================================
# test
# ============================================================================
function test_rho_pipi()
    println("ρ–ππ coupled-channel system  I=1  Ncut=$Ncut_rp")
    println()

    ch_rho = FockChannel("rho", [1], [:boson], [m_ρ_rp],
                         [1//1], [1//1], [-1.0], relativistic)
    ch_pipi = FockChannel("pipi", [2], [:boson], [m_π_rp],
                          [0//1], [1//1], [1.0], relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_rp, [ch_rho, ch_pipi], L0_rp, a_rp, 1//1,
                     irrep_names_rp)
    params = (g=g_rp,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p)
        if length(nA) == 2 && length(nB) == 1
            # ππ (bra) ← ρ (ket)
            k, σ = nA[1], Int(s[1])
            return ComplexF64(p.g * (2π * ħc_rp / L_phys) * fm_rp(σ, k))
        elseif length(nA) == 1 && length(nB) == 2
            # ρ (bra) ← ππ (ket)
            k, σ = nB[1], Int(sp[1])
            return ComplexF64(p.g * (2π * ħc_rp / L_phys) * conj(fm_rp(σ, k)))
        end
        return zero(ComplexF64)
    end

    all_ok = true
    for Gamma in irrep_names_rp
        ref = _build_rho_pipi_reference_per_irrep(Gamma)
        ref === nothing && continue

        free_Ts_vec = Float64[]
        for (idx, (nt, _)) in enumerate(ref.all_states)
            m_arr = idx <= ref.dim_rho ? [m_ρ_rp] : [m_π_rp, m_π_rp]
            T = sum(sqrt(m_^2 + pref_T_rp * Float64(sum(abs2, n_)))
                    for (n_, m_) in zip(nt, m_arr))
            push!(free_Ts_vec, T)
        end
        free_Ts = sort(unique!(free_Ts_vec))
        free_Ts_d = distinct_levels(free_Ts, 10)

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
        println("  Free energy levels: $(round.(free_Ts_d, digits=4))")
        println("  $Gamma: dim=$dim_total (ρ=$(ref.dim_rho) ππ=$(ref.dim_pipi)) matched=$matched/$(length(evals_proj)) K=$(ref.K)  $status  [$(round.(di, digits=4))]")
        @test ok
    end

    println()
    println(all_ok ? "All checks passed ✓" : "Some checks failed ✗")
    return all_ok
end

test_rho_pipi()
