# ==========================================================================
# NN moving-frame test (end-to-end validation of the main code)
#
# Use `FockSystem` and `build_hamiltonian_block` to validate the main public calculation path.
# For every `(d_total, Γ)`:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   Check: eig(H_proj) ⊂ evals_full (reference spectrum)
#
# NN: identical spin-1/2 fermions (N=2, even fermions → single cover)
# I=1 + κ=[2] → S=0 (singlet), I=0 + κ=[1,1] → S=1 (triplet)
# Moving frames: D001 (C4v), D011 (C2v), D111 (C3v)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const SG         = NPHFforFVE.SymmetryGroup
const M_nn       = NPHFforFVE.Momentum
const m_N        = 938.92
const L0_nn      = 48
const a_nn       = 0.1
const L_phys_nn  = L0_nn * a_nn
const hc_nn      = 197.327
const Ncut_nn    = 20
const C0_nn      = 15.0e-6
const Lambda_nn  = 1000.0
const pv_nn       = 2π * hc_nn / L_phys_nn
const Lambda2_nn = Lambda_nn^2
const per_mass_nn = [m_N, m_N]
const per_spin_nn = Rational{Int}[1//2, 1//2]
const spin_nn     = 0.5
const etas_nn     = Float64[1.0, 1.0]

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
# Reference-spectrum construction
# ============================================================================
function _build_nn_moving_reference(d_total, kappa, sign_ex, irrep_names)
    reps = NPHFforFVE.find_representatives(2; Ncut=Ncut_nn, d=d_total,
        species=[2], particle_types=[:fermion])

    all_states = []
    state_to_idx = Dict()

    for rep in reps
        zm = [iszero(rep[i]) for i in 1:2]
        hel_configs = if !any(zm)
            h = NPHFforFVE.helicity_representatives(rep;
                species=[2], particle_types=[:fermion],
                spins=Float64[spin_nn], d=d_total)
            [Tuple(Float64.(x)) for x in h]
        else
            N_fin = 2 - count(zm)
            fin = Tuple(rep[i] for i in 1:2 if !zm[i])
            h_fin = if N_fin > 0
                try NPHFforFVE.helicity_representatives(
                    fin; species=[N_fin], particle_types=[:fermion],
                    spins=Float64[spin_nn], d=d_total)
                catch; [()]; end
            else; [()]; end
            [Tuple(Float64[zm[i] ? 0.0 : Float64(h[count(!iszero, rep[1:i])])
                          for i in 1:2]) for h in h_fin]
        end

        for hel in hel_configs, Gamma in irrep_names
            result = NPHFforFVE.subspace_projection(rep, hel, kappa, Gamma;
                d_total=d_total, species_type=:fermion, spin=spin_nn, etas=etas_nn)
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

    # Metric S — grouped by `n_tuple`; ZM groups use `build_S_matrix_zero_momentum`
    S_mat = zeros(Float64, K, K)
    n_tuple_to_idxs = Dict{Tuple, Vector{Int}}()
    for (idx, (n_tup, _)) in enumerate(all_states)
        idxs = get!(Vector{Int}, n_tuple_to_idxs, n_tup)
        push!(idxs, idx)
    end

    for (n_tup, idxs) in n_tuple_to_idxs
        M_zm = count(n -> iszero(n), n_tup)
        if M_zm == 2
            spin_tuples = [all_states[idx][2] for idx in idxs]
            S_grp = NPHFforFVE.build_S_matrix_zero_momentum(2, spin_tuples,
                [(Tuple{}(), Tuple{}())], 2, kappa, :fermion)
        else
            S_grp = NPHFforFVE.build_S_matrix(all_states[idxs], 2, kappa, :fermion)
        end
        S_mat[idxs, idxs] .= S_grp
    end

    # Kinetic energy (CM-frame relativistic; consistent with `_kinetic_energy_rep`)
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        p_mov = [pv_nn .* Float64.(n_) for n_ in n_tup]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass_nn, d_total, L_phys_nn)
        T = sum(sqrt(m_N^2 + Float64(sum(abs2, p_))) for p_ in p_cm)
        T_diag[idx, idx] = T
    end

    # V_func (CM-frame boost + dipole ff)
    function V_can(np, sp, n, s, extra...)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        p_mov = [pv_nn .* Float64.(ni) for ni in np]
        k_mov = [pv_nn .* Float64.(ni) for ni in n]
        p_cm, fb = NPHFforFVE.boost_to_cm(p_mov, per_mass_nn, d_total, L_phys_nn)
        k_cm, fk = NPHFforFVE.boost_to_cm(k_mov, per_mass_nn, d_total, L_phys_nn)
        ff_bra = prod(1 / (1 + Float64(sum(abs2, pc)) / Lambda2_nn)^2 for pc in p_cm)
        ff_ket = prod(1 / (1 + Float64(sum(abs2, kc)) / Lambda2_nn)^2 for kc in k_cm)
        ComplexF64(fb * C0_nn * ff_bra * ff_ket * fk * sf)
    end

    V_hel = NPHFforFVE.build_V_hel(all_states, all_states, per_spin_nn, per_spin_nn, V_can)
    dd = 3 * (2 + 2) - 6
    fv = (2π * hc_nn / L_phys_nn)^(dd / 2)
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
# Main-code `V_func`
# ============================================================================
function _make_V_func_nn(d_total, sign_ex)
    return function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p)
        diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
        exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
        sf = diag + sign_ex * exch
        sf == 0.0 && return zero(ComplexF64)
        pv = 2π * hc_nn / L_phys
        p_mov = [pv .* Float64.(ni) for ni in nA]
        k_mov = [pv .* Float64.(ni) for ni in nB]
        p_cm, fb = NPHFforFVE.boost_to_cm(p_mov, per_mass_nn, d_total, L_phys)
        k_cm, fk = NPHFforFVE.boost_to_cm(k_mov, per_mass_nn, d_total, L_phys)
        ff_bra = prod(1 / (1 + Float64(sum(abs2, pc)) / Lambda2_nn)^2 for pc in p_cm)
        ff_ket = prod(1 / (1 + Float64(sum(abs2, kc)) / Lambda2_nn)^2 for kc in k_cm)
        ComplexF64(fb * p.C0 * ff_bra * ff_ket * fk * sf)
    end
end

# ============================================================================
# Validation function
# ============================================================================
function _verify_nn_moving(sys, V_func, params, ref, free_Ts, label)
    println("  Free energy levels (first 10 nondegenerate): $(round.(free_Ts, digits=4))")
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
    println(all_ok ? "All checks passed ✓" : "Some checks failed ✗")
    return all_ok
end

# ============================================================================
# D001 (C4v)
# ============================================================================
function _compute_nn_moving_free_Ts(all_states, d_total)
    free_Ts = Float64[]
    for (nt, _) in all_states
        p_mov = [pv_nn .* Float64.(n_) for n_ in nt]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass_nn, d_total, L_phys_nn)
        T = sum(sqrt(m_N^2 + Float64(sum(abs2, p_))) for p_ in p_cm)
        push!(free_Ts, T)
    end
    sort(unique!(free_Ts))
end

function test_nn_D001()
    d_total = NPHFforFVE.D001
    irrep_names = SG.C4V_BOSONIC_NAMES
    println("NN  D001 (C4v)  Ncut=$Ncut_nn")
    all_ok = true

    for (isospin, kap, sign_ex, label) in [(1//1, "[2]", -1.0, "I=1,S=0"),
                                            (0//1, "[1,1]", +1.0, "I=0,S=1")]
        println("\n  --- $label ---")
        ref = _build_nn_moving_reference(d_total, kap, sign_ex, irrep_names)
        free_Ts = distinct_levels(_compute_nn_moving_free_Ts(ref.all_states, d_total), 10)
        println("  reference eigenvalues = $(length(ref.evals_full))")

        ch = FockChannel("NN", [2], [:fermion], [m_N], [1//2], [1//2], [1.0],
                         NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_nn, [ch], L0_nn, a_nn, isospin, irrep_names)
        params = (C0=C0_nn,)

        ok = _verify_nn_moving(sys, _make_V_func_nn(d_total, sign_ex), params,
                               ref.evals_full, free_Ts, "$label D001")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# D011 (C2v)
# ============================================================================
function test_nn_D011()
    d_total = NPHFforFVE.D011
    irrep_names = SG.C2V_BOSONIC_NAMES
    println("\n$(repeat("=", 60))")
    println("NN  D011 (C2v)  Ncut=$Ncut_nn")
    all_ok = true

    for (isospin, kap, sign_ex, label) in [(1//1, "[2]", -1.0, "I=1,S=0"),
                                            (0//1, "[1,1]", +1.0, "I=0,S=1")]
        println("\n  --- $label ---")
        ref = _build_nn_moving_reference(d_total, kap, sign_ex, irrep_names)
        free_Ts = distinct_levels(_compute_nn_moving_free_Ts(ref.all_states, d_total), 10)
        println("  reference eigenvalues = $(length(ref.evals_full))")

        ch = FockChannel("NN", [2], [:fermion], [m_N], [1//2], [1//2], [1.0],
                         NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_nn, [ch], L0_nn, a_nn, isospin, irrep_names)
        params = (C0=C0_nn,)

        ok = _verify_nn_moving(sys, _make_V_func_nn(d_total, sign_ex), params,
                               ref.evals_full, free_Ts, "$label D011")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# D111 (C3v)
# ============================================================================
function test_nn_D111()
    d_total = NPHFforFVE.D111
    irrep_names = SG.C3V_BOSONIC_NAMES
    println("\n$(repeat("=", 60))")
    println("NN  D111 (C3v)  Ncut=$Ncut_nn")
    all_ok = true

    for (isospin, kap, sign_ex, label) in [(1//1, "[2]", -1.0, "I=1,S=0"),
                                            (0//1, "[1,1]", +1.0, "I=0,S=1")]
        println("\n  --- $label ---")
        ref = _build_nn_moving_reference(d_total, kap, sign_ex, irrep_names)
        free_Ts = distinct_levels(_compute_nn_moving_free_Ts(ref.all_states, d_total), 10)
        println("  reference eigenvalues = $(length(ref.evals_full))")

        ch = FockChannel("NN", [2], [:fermion], [m_N], [1//2], [1//2], [1.0],
                         NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_nn, [ch], L0_nn, a_nn, isospin, irrep_names)
        params = (C0=C0_nn,)

        ok = _verify_nn_moving(sys, _make_V_func_nn(d_total, sign_ex), params,
                               ref.evals_full, free_Ts, "$label D111")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# Run tests
# ============================================================================
function test_nn_moving()
    ok1 = test_nn_D001()
    ok2 = test_nn_D011()
    ok3 = test_nn_D111()
    return ok1 && ok2 && ok3
end

test_nn_moving()
