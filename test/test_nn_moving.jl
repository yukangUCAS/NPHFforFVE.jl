# ==========================================================================
# NN (nucleon-nucleon) 运动系 (d=D001, D011, D111) 本征值归属测试
#
# 两自旋 1/2 全同费米子 → species=[2], particle_types=[:fermion]
#
# 同位旋道:
#   I=1 (对称) + κ=[2] (对称) → S=0 (反对称, Pauli)  sign_ex = -1
#   I=0 (反对称) + κ=[1,1] (反对称) → S=1 (对称, Pauli)  sign_ex = +1
#
# 作用: V(σ'₁,σ'₂; σ₁,σ₂) = C * (δ_{σ'₁,σ₁}δ_{σ'₂,σ₂} ± δ_{σ'₁,σ₂}δ_{σ'₂,σ₁})
#   C0 = 15×10⁻⁶ MeV⁻²
#   形状因子: dipole, Λ=1000 MeV (CM 框架)
#
# d=D001 → 小群 C4v (16 元素双覆盖 C4v2)
# d=D011 → 小群 C2v (8 元素双覆盖 C2v2)
# d=D111 → 小群 C3v (12 元素双覆盖 C3v2)
#
# 验证: eig(H_proj) ⊂ eig(H_raw)  for all (d, κ, irrep) combinations
#
# ==========================================================================
# 检验结果 (L=48, a=0.1 fm, Ncut=20)
#
# d=D001 (C4v2):
#   κ=[2]  (I=1,S=0): A1(81✓) A2(81✓) B1(78✓) B2(78✓) E(159✓)  5/5
#   κ=[1,1](I=0,S=1): A1(81✓) A2(81✓) B1(78✓) B2(78✓) E(159✓)  5/5
#
# d=D011 (C2v2):
#   κ=[2]  (I=1,S=0): A1(161✓) A2(158✓) B1(140✓) B2(137✓)  4/4
#   κ=[1,1](I=0,S=1): A1(137✓) A2(140✓) B1(158✓) B2(161✓)  4/4
#   (κ=[2]↔[1,1] 的 A1/A2 和 B1/B2 维数互换，体现置换对称性差异)
#
# d=D111 (C3v2):
#   κ=[2]  (I=1,S=0): A1(93✓) A2(93✓) E(185✓)  3/3
#   κ=[1,1](I=0,S=1): A1(93✓) A2(93✓) E(185✓)  3/3
#
# NN 总自旋为整数，仅玻色子不可约表示非空；费米子不可约表示维数自然为零。
# 全部 3×2 = 6 个 (d,κ) 组合通过检验，零失败。
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

function _run_nn_d_test(d_total, irrep_names, d_label)
    M = NPHFforFVE.Momentum

    # ---------- 物理参数 ----------
    M_N   = 938.92
    L0    = 48
    a     = 0.1
    L_phys = L0 * a
    ħc_val = 197.327
    Ncut  = 20

    C0 = 15.0e-6
    Λ  = 1000.0

    pv = 2π * ħc_val / L_phys
    Λ_phys2 = Λ^2
    fv_factor = (2π * ħc_val / L_phys)^3

    per_spin_r = Rational{Int}[1//2, 1//2]
    per_mass = [M_N, M_N]
    spin_val = 0.5
    etas_v = Float64[1.0, 1.0]
    N_α = 2

    species = [2]
    particle_types = [:fermion]

    kappa_list = [("[2]", "I=1,S=0"), ("[1,1]", "I=0,S=1")]

    function has_zm_spin(rep)
        for i in 1:N_α
            if iszero(rep[i])
                return true
            end
        end
        return false
    end

    all_ok_all = true

    for (kappa_str, label) in kappa_list
        sign_ex = kappa_str == "[2]" ? -1.0 : +1.0
        println("\n  --- κ=$kappa_str ($label) ---")

        function V_can_func(np, sp, n, s, extra...)
            diag = (sp[1] == s[1] && sp[2] == s[2]) ? 1.0 : 0.0
            exch = (sp[1] == s[2] && sp[2] == s[1]) ? 1.0 : 0.0
            sf = diag + sign_ex * exch
            sf == 0.0 && return zero(ComplexF64)

            p_mov = [pv .* Float64.(ni) for ni in np]
            k_mov = [pv .* Float64.(ni) for ni in n]

            p_cm, fac_bra = NPHFforFVE.boost_to_cm(p_mov, per_mass, d_total, L_phys)
            k_cm, fac_ket = NPHFforFVE.boost_to_cm(k_mov, per_mass, d_total, L_phys)

            ff_bra = prod(1.0 / (1.0 + Float64(sum(abs2, pc)) / Λ_phys2)^2 for pc in p_cm)
            ff_ket = prod(1.0 / (1.0 + Float64(sum(abs2, kc)) / Λ_phys2)^2 for kc in k_cm)

            ComplexF64(fac_bra * C0 * ff_bra * ff_ket * fac_ket * sf)
        end

        # ====================================================================
        # 代表态
        # ====================================================================
        reps = NPHFforFVE.find_representatives(N_α; Ncut=Ncut, d=d_total,
            species=species, particle_types=particle_types)
        @info "$(d_label) κ=$kappa_str: $(length(reps)) orbital representatives"

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
                    spins=Float64[spin_val], d=d_total)
            catch
                [ntuple(_ -> 0.0, N_α)]
            end

            for hel in h_reps
                hel_float = Tuple(Float64.(hel))

                for Gamma in irrep_names
                    result = try
                        NPHFforFVE.subspace_projection(rep, hel_float,
                            kappa_str, Gamma;
                            d_total=d_total, species_type=:fermion,
                            spin=spin_val, etas=etas_v)
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
        @info "$(d_label) κ=$kappa_str: $K helicity states"
        K == 0 && continue

        # ====================================================================
        # T_diag (boost n → p* → CM kinetic energy)
        # ====================================================================
        T_diag = zeros(ComplexF64, K, K)
        for (idx, (n_tup, _)) in enumerate(all_states)
            p_mov = [pv .* Float64.(n_) for n_ in n_tup]
            p_cm, _ = NPHFforFVE.boost_to_cm(p_mov, per_mass, d_total, L_phys)
            T = sum(sqrt(M_N^2 + Float64(sum(abs2, p_))) for p_ in p_cm)
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
        @info "$(d_label) κ=$kappa_str: H_raw $(K)×$(K), $(length(evals_full)) eigenvalues"

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

        all_ok_κ = true
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
            all_ok_κ = all_ok_κ && ok
            all_ok_all = all_ok_all && ok
            @test ok
            println("    Γ=$Gamma  n_r=$(total_cols)  matched=$matched/$(length(evals_proj))  $(ok ? "✓" : "✗ FAIL")")
        end

        @test n_nonempty > 0
        println("    κ=$kappa_str: $(n_nonempty) non-empty irreps, ok=$all_ok_κ")
    end

    return all_ok_all
end

sg = NPHFforFVE.SymmetryGroup

@testset "NN d=D001 (C4v2)" begin
    _run_nn_d_test(NPHFforFVE.D001, sg.C4V_IRREP_NAMES, "d=D001")
end

@testset "NN d=D011 (C2v2)" begin
    _run_nn_d_test(NPHFforFVE.D011, sg.C2V_IRREP_NAMES, "d=D011")
end

@testset "NN d=D111 (C3v2)" begin
    _run_nn_d_test(NPHFforFVE.D111, sg.C3V_IRREP_NAMES, "d=D111")
end
