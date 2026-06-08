# ==========================================================================
# πN → πN 运动系 (d=D001, D011, D111) 本征值归属测试
#
# π: s=0, I=1, boson,  m=139.57 MeV
# N: s=1/2, I=1/2, fermion, m=938.92 MeV
# 可区分粒子 → species=[1,1], κ=("[1]","[1]"), S=I
#
# d=D001 → 小群 C4v (16 元素双覆盖 C4v2)
#   玻色子不可约表示: A1, A2, B1, B2, E
#   费米子不可约表示: G1, G2
#
# d=D011 → 小群 C2v (8 元素双覆盖 C2v2)
#   玻色子不可约表示: A1, A2, B1, B2
#   费米子不可约表示: G
#
# d=D111 → 小群 C3v (12 元素双覆盖 C3v2)
#   玻色子不可约表示: A1, A2, E
#   费米子不可约表示: F1, F2, G
#
# 作用: <πN,σ'|V|πN,σ> = C0 * δ_{σ',σ}   (正则自旋基)
# C0 = 1×10⁻⁵ MeV⁻²
# 形状因子: dipole, Λ=1000 MeV (CM 框架)
#
# 验证: eig(H_proj) ⊂ eig(H_raw)  for all irreps
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

function _run_d_test(d_total, irrep_names, d_label)
    M = NPHFforFVE.Momentum

    # ---------- 物理参数 ----------
    m_π   = 139.57
    m_N   = 938.92
    Λ     = 1000.0
    L0    = 48
    a     = 0.1
    L_phys = L0 * a
    ħc_val = 197.327
    Ncut  = 20

    C0 = 1.0e-5

    pref_T  = (2π * ħc_val / L_phys)^2
    pv = 2π * ħc_val / L_phys
    Λ_phys2 = Λ^2
    fv_factor = (2π * ħc_val / L_phys)^3

    ff_cm(p2) = 1.0 / (1.0 + p2 / Λ_phys2)^2

    species = [1, 1]
    particle_types = [:boson, :fermion]
    spins = Float64[0.0, 0.5]
    etas  = Float64[1.0, 1.0]
    N_α = 2

    per_spin_r = Rational{Int}[0//1, 1//2]
    per_mass = [m_π, m_N]
    kappa_tuple = ("[1]", "[1]")

    function has_zm_spin(rep)
        for i in 1:N_α
            if iszero(rep[i]) && spins[i] != 0.0
                return true
            end
        end
        return false
    end

    function V_can_func(np, sp, n, s, extra...)
        σ_π_p, σ_N_p = sp
        σ_π, σ_N = s
        σ_π_p == σ_π || return zero(ComplexF64)
        σ_N_p == σ_N || return zero(ComplexF64)

        p_mov = [pv .* Float64.(ni) for ni in np]
        k_mov = [pv .* Float64.(ni) for ni in n]

        p_cm, fac_bra = NPHFforFVE.boost_to_cm(p_mov, per_mass, d_total, L_phys)
        k_cm, fac_ket = NPHFforFVE.boost_to_cm(k_mov, per_mass, d_total, L_phys)

        ff_bra = prod(ff_cm(Float64(sum(abs2, pc))) for pc in p_cm)
        ff_ket = prod(ff_cm(Float64(sum(abs2, kc))) for kc in k_cm)

        ComplexF64(fac_bra * C0 * ff_bra * ff_ket * fac_ket)
    end

    # ====================================================================
    # 代表态
    # ====================================================================
    reps = NPHFforFVE.find_representatives(N_α; Ncut=Ncut, d=d_total,
        species=species, particle_types=particle_types)
    @info "$(d_label): found $(length(reps)) orbital representatives"

    # ====================================================================
    # 收集投影分块
    # ====================================================================
    all_states = []
    state_to_idx = Dict()
    proj_blocks = []

    for rep in reps
        has_zm_spin(rep) && continue

        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species, particle_types=particle_types,
                spins=spins, d=d_total)
        catch
            [ntuple(_ -> 0.0, N_α)]
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))

            for Gamma in irrep_names
                result = try
                    NPHFforFVE.subspace_projection(rep, hel_float,
                        kappa_tuple, Gamma;
                        d_total=d_total, species=species,
                        particle_types=particle_types,
                        spins=spins, etas=etas)
                catch
                    continue
                end

                n_r = size(result.X, 2)
                n_r == 0 && continue

                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                end

                push!(proj_blocks, (X=result.X, states=result.subspace_states,
                                    rep=rep, Gamma=Gamma, n_r=n_r))
            end
        end
    end

    K = length(all_states)
    @info "$(d_label): total $K helicity states in unified basis"
    K == 0 && error("No states found!")

    # ====================================================================
    # T_diag (boost n → p* → CM kinetic energy)
    # ====================================================================
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        p_mov = [pv .* Float64.(n_) for n_ in n_tup]
        p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass, d_total, L_phys)
        T = sum(sqrt(m_^2 + Float64(sum(abs2, p_))) for (p_, m_) in zip(p_cm, per_mass))
        T_diag[idx, idx] = T
    end

    # ====================================================================
    # V_hel + H_raw
    # ====================================================================
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                       per_spin_r, per_spin_r, V_can_func)

    H_raw = T_diag + fv_factor * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]
    @info "$(d_label): H_raw dimension $K, eigenvalues: $(length(evals_full))"

    # ====================================================================
    # 按不可约表示分组
    # ====================================================================
    blocks_by_irrep = Dict{String, Vector}()
    for blk in proj_blocks
        Gamma = blk.Gamma
        haskey(blocks_by_irrep, Gamma) || (blocks_by_irrep[Gamma] = [])
        push!(blocks_by_irrep[Gamma], blk)
    end

    block_idx_map = Dict()
    for (i, blk) in enumerate(proj_blocks)
        block_idx_map[blk] = [state_to_idx[st] for st in blk.states]
    end

    all_ok = true
    n_nonempty = 0

    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        total_cols = sum(blk.n_r for blk in blocks)
        total_cols == 0 && continue
        n_nonempty += 1

        X_big = zeros(ComplexF64, K, total_cols)
        col = 1
        for blk in blocks
            idx = block_idx_map[blk]
            X_big[idx, col:col+blk.n_r-1] .= blk.X
            col += blk.n_r
        end

        H_proj = X_big' * H_raw * X_big
        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_proj = evals_proj[isfinite.(evals_proj)]

        matched = 0
        for ep in evals_proj
            if minimum(abs.(ep .- evals_full)) < 1e-8
                matched += 1
            end
        end

        ok = matched == length(evals_proj)
        all_ok = all_ok && ok
        @test ok
        println("  Γ=$Gamma  n_r=$(total_cols)  matched=$matched/$(length(evals_proj))  $(ok ? "✓" : "✗ FAIL")")
    end

    @test n_nonempty > 0
    println("  $(d_label): $(n_nonempty) non-empty irreps, all_ok=$all_ok\n")
    return all_ok
end

@testset "πN d=D001 (C4v2)" begin
    sg = NPHFforFVE.SymmetryGroup
    _run_d_test(NPHFforFVE.D001, sg.C4V_IRREP_NAMES, "d=D001")
end

@testset "πN d=D011 (C2v2)" begin
    sg = NPHFforFVE.SymmetryGroup
    _run_d_test(NPHFforFVE.D011, sg.C2V_IRREP_NAMES, "d=D011")
end

@testset "πN d=D111 (C3v2)" begin
    sg = NPHFforFVE.SymmetryGroup
    _run_d_test(NPHFforFVE.D111, sg.C3V_IRREP_NAMES, "d=D111")
end
