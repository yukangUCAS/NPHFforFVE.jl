# ==========================================================================
# πN → πN moving-frame test (end-to-end validation of the main code)
#
# Use `FockSystem` and `build_hamiltonian_block` to validate the main public calculation path.
# For every `(d_total, Γ)`:
#   H_proj = build_hamiltonian_block(sys, Γ, V_func, params)
#   Check: eig(H_proj) ⊂ evals_full (reference spectrum)
#
# π: s=0, I=1, boson,  η=+1, m=139.57 MeV
# N: s=1/2, I=1/2, fermion, η=+1, m=938.92 MeV
# Distinguishable particles → species=[1,1], κ=("[1]","[1]"), S=I
# Moving frames: D001 (C4v2), D011 (C2v2), D111 (C3v2)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

const SG            = NPHFforFVE.SymmetryGroup
const M_piN         = NPHFforFVE.Momentum
const m_π           = 139.57
const m_N_piN       = 938.92
const L0_piN        = 48
const a_piN         = 0.1
const L_phys_piN    = L0_piN * a_piN
const hc_piN        = 197.327
const Ncut_piN      = 20
const C0_piN        = 1.0e-5
const Lambda_piN    = 1000.0
const pv_piN        = 2π * hc_piN / L_phys_piN
const Lambda2_piN   = Lambda_piN^2
const per_mass_piN  = [m_π, m_N_piN]
const per_spin_piN  = Rational{Int}[0//1, 1//2]
const spin_piN      = Float64[0.0, 0.5]
const etas_piN      = Float64[1.0, 1.0]
const species_piN   = [1, 1]
const ptypes_piN    = [:boson, :fermion]
const kappa_piN     = ("[1]", "[1]")
const N_α_piN       = 2

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
# Reference-spectrum construction (including ZM-state handling — can occur only when N has nonzero spin)
# ============================================================================
function _has_zm_spin_piN(rep)
    for i in 1:N_α_piN
        if iszero(rep[i]) && spin_piN[i] != 0.0
            return true
        end
    end
    return false
end

function _gen_hel_labels_piN(rep, d_total)
    if !_has_zm_spin_piN(rep)
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species_piN, particle_types=ptypes_piN,
                spins=spin_piN, d=d_total)
        catch
            [ntuple(_ -> 0.0, N_α_piN)]
        end
        return [Tuple(Float64.(hel)) for hel in h_reps]
    end

    # ZM+spin: FM-only helicity reps + 0.0 placeholders at ZM positions
    N_fin = N_α_piN - count(n -> iszero(n), rep)
    fin_momenta = M_piN[]
    fin_species_vec = Int[]
    fin_types_vec = Symbol[]
    fin_spins_vec = Rational{Int}[]
    off = 0
    for (k, Nk) in enumerate(species_piN)
        fm_in_sp = 0
        for i in 1:Nk
            !iszero(rep[off + i]) && (fm_in_sp += 1)
        end
        if fm_in_sp > 0
            push!(fin_species_vec, fm_in_sp)
            push!(fin_types_vec, ptypes_piN[k])
            push!(fin_spins_vec, Rational{Int}(Int(2*spin_piN[k]), 2))
            for i in 1:Nk
                !iszero(rep[off + i]) && push!(fin_momenta, rep[off + i])
            end
        end
        off += Nk
    end

    h_fm = if N_fin > 0
        fin_rep = Tuple(fin_momenta)
        try
            NPHFforFVE.helicity_representatives(fin_rep;
                species=fin_species_vec, particle_types=fin_types_vec,
                spins=fin_spins_vec, d=d_total)
        catch
            [()]
        end
    else
        [()]
    end

    result = []
    for fin_lam in h_fm
        fin_lam_f = Float64[Float64(x) for x in fin_lam]
        full = Float64[]
        fm_idx = 1
        off2 = 0
        for (k, Nk) in enumerate(species_piN)
            for i in 1:Nk
                if iszero(rep[off2 + i])
                    push!(full, 0.0)
                else
                    push!(full, fin_lam_f[fm_idx])
                    fm_idx += 1
                end
            end
            off2 += Nk
        end
        push!(result, Tuple(full))
    end
    return result
end

function _build_piN_moving_reference(d_total, irrep_names)
    reps = NPHFforFVE.find_representatives(N_α_piN; Ncut=Ncut_piN, d=d_total,
        species=species_piN, particle_types=ptypes_piN)

    all_states = []
    state_to_idx = Dict()

    for rep in reps
        for hel_float in _gen_hel_labels_piN(rep, d_total)
            for Gamma in irrep_names
                result = NPHFforFVE.subspace_projection(rep, hel_float,
                    kappa_piN, Gamma;
                    d_total=d_total, species=species_piN,
                    particle_types=ptypes_piN,
                    spins=spin_piN, etas=etas_piN)
                size(result.X, 2) == 0 && continue
                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                end
            end
        end
    end

    K = length(all_states)
    K == 0 && return Float64[]

    # Kinetic energy (CM-frame relativistic)
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        p_mov = [pv_piN .* Float64.(n_) for n_ in n_tup]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass_piN, d_total, L_phys_piN)
        T = sum(sqrt(m^2 + Float64(sum(abs2, p_))) for (p_, m) in zip(p_cm, per_mass_piN))
        T_diag[idx, idx] = T
    end

    # V_func (canonical spin basis)
    function V_can(np, sp, n, s, extra...)
        sp[2] == s[2] || return zero(ComplexF64)   # N spin diagonal (π spin always 0)
        p_mov = [pv_piN .* Float64.(ni) for ni in np]
        k_mov = [pv_piN .* Float64.(ni) for ni in n]
        p_cm, fb = NPHFforFVE.boost_to_cm(p_mov, per_mass_piN, d_total, L_phys_piN)
        k_cm, fk = NPHFforFVE.boost_to_cm(k_mov, per_mass_piN, d_total, L_phys_piN)
        ff_bra = prod(1 / (1 + Float64(sum(abs2, pc)) / Lambda2_piN)^2 for pc in p_cm)
        ff_ket = prod(1 / (1 + Float64(sum(abs2, kc)) / Lambda2_piN)^2 for kc in k_cm)
        ComplexF64(fb * C0_piN * ff_bra * ff_ket * fk)
    end

    V_hel = NPHFforFVE.build_V_hel(all_states, all_states, per_spin_piN, per_spin_piN, V_can)
    dd = 3 * (2 + 2) - 6
    fv = (2π * hc_piN / L_phys_piN)^(dd / 2)
    H_raw = T_diag + fv * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]
    return (evals_full=evals_full, all_states=all_states)
end

# ============================================================================
# Main-code `V_func`
# ============================================================================
function _make_V_func_piN(d_total)
    return function V_func(nA, nB, sp, s, kapA, kapB, rA, rB, chA, chB, L_phys, p)
        sp[2] == s[2] || return zero(ComplexF64)   # N spin diagonal
        pv = 2π * hc_piN / L_phys
        p_mov = [pv .* Float64.(ni) for ni in nA]
        k_mov = [pv .* Float64.(ni) for ni in nB]
        p_cm, fb = NPHFforFVE.boost_to_cm(p_mov, per_mass_piN, d_total, L_phys)
        k_cm, fk = NPHFforFVE.boost_to_cm(k_mov, per_mass_piN, d_total, L_phys)
        ff_bra = prod(1 / (1 + Float64(sum(abs2, pc)) / Lambda2_piN)^2 for pc in p_cm)
        ff_ket = prod(1 / (1 + Float64(sum(abs2, kc)) / Lambda2_piN)^2 for kc in k_cm)
        ComplexF64(fb * p.C0 * ff_bra * ff_ket * fk)
    end
end

# ============================================================================
# Validation function
# ============================================================================
function _verify_piN_moving(sys, V_func, params, ref, free_Ts, label)
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
        matched = count(ep -> minimum(abs.(ep .- ref)) < 1e-8, evals_proj)
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
# D001 (C4v2)
# ============================================================================
function _compute_moving_free_Ts(all_states, d_total, pv, per_mass, L_phys)
    free_Ts = Float64[]
    for (nt, _) in all_states
        p_mov = [pv .* Float64.(n_) for n_ in nt]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass, d_total, L_phys)
        T = sum(sqrt(m^2 + Float64(sum(abs2, p_))) for (p_, m) in zip(p_cm, per_mass))
        push!(free_Ts, T)
    end
    sort(unique!(free_Ts))
end

function test_piN_D001()
    d_total = NPHFforFVE.D001
    irrep_names = SG.C4V_IRREP_NAMES
    println("πN  D001 (C4v2)  Ncut=$Ncut_piN")
    all_ok = true

    ref = _build_piN_moving_reference(d_total, irrep_names)
    free_Ts = distinct_levels(_compute_moving_free_Ts(ref.all_states, d_total,
        pv_piN, per_mass_piN, L_phys_piN), 10)
    println("  reference eigenvalues = $(length(ref.evals_full))")

    for (isospin, label) in [(1//2, "I=1/2"), (3//2, "I=3/2")]
        println("\n  --- $label ---")

        ch = FockChannel("piN", [1, 1], [:boson, :fermion],
                         [m_π, m_N_piN], [0//1, 1//2], [1//1, 1//2],
                         [1.0, 1.0], NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_piN, [ch], L0_piN, a_piN, isospin, irrep_names)
        params = (C0=C0_piN,)

        ok = _verify_piN_moving(sys, _make_V_func_piN(d_total), params,
                                ref.evals_full, free_Ts, "$label D001")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# D011 (C2v2)
# ============================================================================
function test_piN_D011()
    d_total = NPHFforFVE.D011
    irrep_names = SG.C2V_IRREP_NAMES
    println("\n$(repeat("=", 60))")
    println("πN  D011 (C2v2)  Ncut=$Ncut_piN")
    all_ok = true

    ref = _build_piN_moving_reference(d_total, irrep_names)
    free_Ts = distinct_levels(_compute_moving_free_Ts(ref.all_states, d_total,
        pv_piN, per_mass_piN, L_phys_piN), 10)
    println("  reference eigenvalues = $(length(ref.evals_full))")

    for (isospin, label) in [(1//2, "I=1/2"), (3//2, "I=3/2")]
        println("\n  --- $label ---")

        ch = FockChannel("piN", [1, 1], [:boson, :fermion],
                         [m_π, m_N_piN], [0//1, 1//2], [1//1, 1//2],
                         [1.0, 1.0], NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_piN, [ch], L0_piN, a_piN, isospin, irrep_names)
        params = (C0=C0_piN,)

        ok = _verify_piN_moving(sys, _make_V_func_piN(d_total), params,
                                ref.evals_full, free_Ts, "$label D011")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# D111 (C3v2)
# ============================================================================
function test_piN_D111()
    d_total = NPHFforFVE.D111
    irrep_names = SG.C3V_IRREP_NAMES
    println("\n$(repeat("=", 60))")
    println("πN  D111 (C3v2)  Ncut=$Ncut_piN")
    all_ok = true

    ref = _build_piN_moving_reference(d_total, irrep_names)
    free_Ts = distinct_levels(_compute_moving_free_Ts(ref.all_states, d_total,
        pv_piN, per_mass_piN, L_phys_piN), 10)
    println("  reference eigenvalues = $(length(ref.evals_full))")

    for (isospin, label) in [(1//2, "I=1/2"), (3//2, "I=3/2")]
        println("\n  --- $label ---")

        ch = FockChannel("piN", [1, 1], [:boson, :fermion],
                         [m_π, m_N_piN], [0//1, 1//2], [1//1, 1//2],
                         [1.0, 1.0], NPHFforFVE.relativistic)
        sys = FockSystem(d_total, Ncut_piN, [ch], L0_piN, a_piN, isospin, irrep_names)
        params = (C0=C0_piN,)

        ok = _verify_piN_moving(sys, _make_V_func_piN(d_total), params,
                                ref.evals_full, free_Ts, "$label D111")
        all_ok = all_ok && ok
    end
    return all_ok
end

# ============================================================================
# Run tests
# ============================================================================
function test_piN_moving()
    ok1 = test_piN_D001()
    ok2 = test_piN_D011()
    ok3 = test_piN_D111()
    return ok1 && ok2 && ok3
end

test_piN_moving()
