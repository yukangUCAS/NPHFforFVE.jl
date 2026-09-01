# ==========================================================================
# πN → πN test (end-to-end validation of the main code)
#
# Use `FockSystem` and `build_hamiltonian_block` to validate the main public calculation path.
# For every irrep Γ:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   evals_full = eig(H_raw)
#   Check: eig(H_proj) ⊂ evals_full
#
# π: s=0, I=1, boson, η=-1, m=139.57 MeV
# N: s=1/2, I=1/2, fermion, η=+1, m=938.92 MeV
# Distinguishable particles → species=[1,1], κ=("[1]","[1]"), S=I
#
# Only double-valued irreps (G1±, G2±, H±) contain states
# I=1/2 and I=3/2 have identical eigenvalues (isospin-independent interaction)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const M_piN      = NPHFforFVE.Momentum
const m_π_piN    = 139.57
const m_N_piN    = 938.92
const Λ_piN      = 1000.0
const L0_piN     = 48
const a_piN      = 0.1
const L_phys_piN = L0_piN * a_piN
const ħc_piN     = 197.327
const Ncut_piN   = 20
const C0_piN     = 1.0e-5

const pref_T_piN   = (2π * ħc_piN / L_phys_piN)^2
const Λ_n²_piN     = (Λ_piN / (2π * ħc_piN / L_phys_piN))^2
const fv_factor_piN = (2π * ħc_piN / L_phys_piN)^3
const d_piN = M_piN(0,0,0)

const species_piN    = [1, 1]
const particle_piN   = [:boson, :fermion]
const N_α_piN        = 2
const per_mass_piN   = [m_π_piN, m_N_piN]
const per_spin_r_piN = Rational{Int}[0//1, 1//2]
const spins_piN      = Float64[0.0, 0.5]
const etas_piN       = Float64[-1.0, 1.0]
const kappa_piN      = ("[1]", "[1]")

const irrep_names_piN = NPHFforFVE.OH2_IRREP_NAMES

ff_piN(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²_piN)^2

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
# Construct the reference spectrum (identical for I=1/2 and I=3/2; isospin-independent)
# ============================================================================
function _build_piN_reference()
    function V_can_func(np, sp, n, s, extra...)
        σ_π_p, σ_N_p = sp
        σ_π, σ_N = s
        σ_π_p == σ_π || return zero(ComplexF64)
        σ_N_p == σ_N || return zero(ComplexF64)
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C0_piN * ff_piN(n1p)*ff_piN(n1) * ff_piN(n2p)*ff_piN(n2))
    end

    reps = NPHFforFVE.find_representatives(N_α_piN; Ncut=Ncut_piN, d=d_piN,
        species=species_piN, particle_types=particle_piN)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_piN, particle_types=particle_piN,
                spins=spins_piN, d=d_piN)
        catch
            [ntuple(_ -> 0.0, N_α_piN)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names_piN
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa_piN, Gamma;
                    d_total=d_piN, species=species_piN,
                    particle_types=particle_piN,
                    spins=spins_piN, etas=etas_piN)
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

    # T_diag (S=I, distinguishable)
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T_piN * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass_piN))
        T_diag[idx, idx] = T
    end

    # V_hel + H_raw
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                     per_spin_r_piN, per_spin_r_piN, V_can_func)
    H_raw = T_diag + fv_factor_piN * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]

    return (evals_full=evals_full, K=K, all_states=all_states)
end

# ============================================================================
# Validate the main code
# ============================================================================
function _verify_piN(sys, V_func, params, ref, free_Ts, label)
    println("  Free energy levels (first 10 nondegenerate): $(round.(free_Ts, digits=4))")
    println()
    all_ok = true
    for Gamma in irrep_names_piN
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
    println(all_ok ? "All checks passed ✓" : "Some checks failed ✗")
    return all_ok
end

# ============================================================================
# I=1/2
# ============================================================================
function test_piN_I12(ref, free_Ts)
    println("πN → πN  I=1/2  Ncut=$Ncut_piN")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    ch = FockChannel("πN", [1,1], [:boson, :fermion],
                     [m_π_piN, m_N_piN], [0//1, 1//2], [1//1, 1//2],
                     [-1.0, 1.0], NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_piN, [ch], L0_piN, a_piN, 1//2,
                     irrep_names_piN)
    params = (C0=C0_piN,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p)
        σ_π_p, σ_N_p = sp
        σ_π, σ_N = s
        σ_π_p == σ_π || return zero(ComplexF64)
        σ_N_p == σ_N || return zero(ComplexF64)
        n1p, n2p = nA; n1, n2 = nB
        ffall = ff_piN(n1p)*ff_piN(n1) * ff_piN(n2p)*ff_piN(n2)
        ComplexF64(p.C0 * ffall)
    end

    return _verify_piN(sys, V_func, params, ref, free_Ts, "I=1/2")
end

# ============================================================================
# I=3/2
# ============================================================================
function test_piN_I32(ref, free_Ts)
    println("πN → πN  I=3/2  Ncut=$Ncut_piN")
    println("K = $(ref.K) states,  reference eigenvalues = $(length(ref.evals_full))")

    ch = FockChannel("πN", [1,1], [:boson, :fermion],
                     [m_π_piN, m_N_piN], [0//1, 1//2], [1//1, 1//2],
                     [-1.0, 1.0], NPHFforFVE.relativistic)
    sys = FockSystem(NPHFforFVE.D000, Ncut_piN, [ch], L0_piN, a_piN, 3//2,
                     irrep_names_piN)
    params = (C0=C0_piN,)

    function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p)
        σ_π_p, σ_N_p = sp
        σ_π, σ_N = s
        σ_π_p == σ_π || return zero(ComplexF64)
        σ_N_p == σ_N || return zero(ComplexF64)
        n1p, n2p = nA; n1, n2 = nB
        ffall = ff_piN(n1p)*ff_piN(n1) * ff_piN(n2p)*ff_piN(n2)
        ComplexF64(p.C0 * ffall)
    end

    return _verify_piN(sys, V_func, params, ref, free_Ts, "I=3/2")
end

# ============================================================================
# Run tests
# ============================================================================
function test_piN()
    ref = _build_piN_reference()
    free_Ts = sort(unique(Float64[
        sum(sqrt(m_^2 + pref_T_piN * Float64(sum(abs2, n_)))
            for (n_, m_) in zip(nt, per_mass_piN))
        for (nt, _) in ref.all_states]))
    free_Ts_d = distinct_levels(free_Ts, 10)
    ok12 = test_piN_I12(ref, free_Ts_d)
    println(repeat("=", 60))
    println()
    ok32 = test_piN_I32(ref, free_Ts_d)
    return ok12 && ok32
end

test_piN()
