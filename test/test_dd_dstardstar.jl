# ==========================================================================
# DD + D*D* 耦合道体系测试 (I=1, κ="[2]")
#
# DD:   两个全同 s=0 玻色子, I=1 → κ="[2]" (空间对称)
# D*D*: 两个全同 s=1 玻色子, I=1 → κ="[2]" (空间对称, 自旋对称 S=0,2)
#
# 相互作用:
#   <DD|V|DD> = C_DD
#   <DD|V|D*D*,σ₁,σ₂> = C_mix * (-1)^{σ₁} * δ_{σ₁,-σ₂}
#   <D*D*|V|D*D*> = C_22 * (δ_{σ₁σ₁'}δ_{σ₂σ₂'} + δ_{σ₁σ₂'}δ_{σ₂σ₁'})
#
# H = T·S + fv·V_hel,  GEP: H v = λ S v
# S = direct + exchange (κ="[2]")
# 验证: eig(H_proj, S_proj) ⊂ eig(H_raw, S)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

function test_dd_dstardstar()
    M = NPHFforFVE.Momentum

    # ---------- 物理参数 ----------
    m_D   = 1864.84
    m_Ds  = 2008.5
    Λ     = 1000.0
    L0    = 48
    a     = 0.1
    L_phys = L0 * a
    ħc_val = 197.327
    Ncut  = 20

    C_DD  = 2.0e-6
    C_mix = 2.0e-6
    C_22  = 2.0e-6

    pref_T   = (2π * ħc_val / L_phys)^2
    Λ_n²     = (Λ / (2π * ħc_val / L_phys))^2
    fv_factor = (2π * ħc_val / L_phys)^3

    ff(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²)^2

    d_total = M(0,0,0)
    irrep_names = NPHFforFVE.OH_IRREP_NAMES
    kappa_sym = "[2]"

    # ---------- V_can 函数 ----------
    function V_DD_can(np, sp, n, s, extra...)
        n1p,n2p=np; n1,n2=n
        ComplexF64(C_DD * ff(n1p)*ff(n2p)*ff(n1)*ff(n2))
    end

    function V_DsDs_can(np, sp, n, s, extra...)
        direct   = (sp[1]==s[1] && sp[2]==s[2]) ? 1.0 : 0.0
        exchange = (sp[1]==s[2] && sp[2]==s[1]) ? 1.0 : 0.0
        spin_factor = direct + exchange
        spin_factor == 0.0 && return zero(ComplexF64)
        n1p,n2p=np; n1,n2=n
        ComplexF64(C_22 * spin_factor * ff(n1p)*ff(n2p)*ff(n1)*ff(n2))
    end

    function V_cross_can(np, sp, n, s, extra...)
        # build_V_hel 交叉块调用: bra 总是 DD (sp=(0,0)), ket 总是 D*D*
        σ₁, σ₂ = Int.(s)
        σ₁ == -σ₂ || return zero(ComplexF64)
        n1p,n2p=np; n1,n2=n
        sign_f = σ₁ == 0 ? 1.0 : -1.0
        ComplexF64(C_mix * sign_f * ff(n1p)*ff(n2p)*ff(n1)*ff(n2))
    end

    # ====================================================================
    # DD 道 (species=[2], s=0, κ="[2]")
    # ====================================================================
    reps = NPHFforFVE.find_representatives(2; Ncut, d=d_total,
        species=[2], particle_types=[:boson])

    all_states_DD = []
    sd_DD = Dict()
    proj_DD = []

    for rep in reps
        hel_tuple = (0.0, 0.0)
        for Gamma in irrep_names
            result = try
                NPHFforFVE.subspace_projection(rep, hel_tuple,
                    kappa_sym, Gamma; d_total=d_total,
                    species_type=:boson, spin=0.0, etas=[1.0])
            catch; continue; end
            n_r = size(result.X, 2)
            n_r == 0 && continue
            for st in result.subspace_states
                if !haskey(sd_DD, st)
                    push!(all_states_DD, st)
                    sd_DD[st] = length(all_states_DD)
                end
            end
            push!(proj_DD, (X=result.X, states=result.subspace_states,
                            rep=rep, n_r=n_r, Gamma=Gamma))
        end
    end
    K_DD = length(all_states_DD)

    # ====================================================================
    # D*D* 道 (species=[2], s=1, κ="[2]")
    # ====================================================================
    all_states_DsDs = []
    sd_DsDs = Dict()
    proj_DsDs = []

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=[2], particle_types=[:boson],
                spins=[1.0], d=d_total)
        catch
            []
        end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))
            for Gamma in irrep_names
                result = try
                    NPHFforFVE.subspace_projection(rep, hel_float,
                        kappa_sym, Gamma; d_total=d_total,
                        species_type=:boson, spin=1.0, etas=[1.0])
                catch; continue; end
                n_r = size(result.X, 2)
                n_r == 0 && continue
                for st in result.subspace_states
                    if !haskey(sd_DsDs, st)
                        push!(all_states_DsDs, st)
                        sd_DsDs[st] = length(all_states_DsDs)
                    end
                end
                push!(proj_DsDs, (X=result.X, states=result.subspace_states,
                                  rep=rep, n_r=n_r, Gamma=Gamma))
            end
        end
    end
    K_DsDs = length(all_states_DsDs)
    K = K_DD + K_DsDs

    # ====================================================================
    # S 矩阵 (κ="[2]": direct + exchange, 块对角)
    # ====================================================================
    S_mat = zeros(Float64, K, K)

    for i in 1:K_DD, j in 1:K_DD
        nA, lamA = all_states_DD[i]
        nB, lamB = all_states_DD[j]
        direct = (nA[1]==nB[1] && nA[2]==nB[2] &&
                  lamA[1]==lamB[1] && lamA[2]==lamB[2]) ? 1.0 : 0.0
        exchange = (nA[1]==nB[2] && nA[2]==nB[1] &&
                    lamA[1]==lamB[2] && lamA[2]==lamB[1]) ? 1.0 : 0.0
        S_mat[i,j] = direct + exchange
    end

    for i in 1:K_DsDs, j in 1:K_DsDs
        nA, lamA = all_states_DsDs[i]
        nB, lamB = all_states_DsDs[j]
        direct = (nA[1]==nB[1] && nA[2]==nB[2] &&
                  lamA[1]==lamB[1] && lamA[2]==lamB[2]) ? 1.0 : 0.0
        exchange = (nA[1]==nB[2] && nA[2]==nB[1] &&
                    lamA[1]==lamB[2] && lamA[2]==lamB[1]) ? 1.0 : 0.0
        S_mat[K_DD+i, K_DD+j] = direct + exchange
    end

    # ====================================================================
    # T_diag
    # ====================================================================
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states_DD)
        T_diag[idx,idx] = sum(sqrt(m_D^2 + pref_T*Float64(sum(abs2,n_)))
                               for n_ in n_tup)
    end
    for (idx, (n_tup, _)) in enumerate(all_states_DsDs)
        T_diag[K_DD+idx, K_DD+idx] = sum(sqrt(m_Ds^2 + pref_T*Float64(sum(abs2,n_)))
                                           for n_ in n_tup)
    end

    # ====================================================================
    # V_hel
    # ====================================================================
    per_spin_DD   = Rational{Int}[0//1, 0//1]
    per_spin_DsDs = Rational{Int}[1//1, 1//1]

    V_hel = zeros(ComplexF64, K, K)

    V_hel[1:K_DD, 1:K_DD] .= NPHFforFVE.build_V_hel(
        all_states_DD, all_states_DD, per_spin_DD, per_spin_DD, V_DD_can)

    V_hel[K_DD+1:K, K_DD+1:K] .= NPHFforFVE.build_V_hel(
        all_states_DsDs, all_states_DsDs, per_spin_DsDs, per_spin_DsDs, V_DsDs_can)

    V_hel[1:K_DD, K_DD+1:K] .= NPHFforFVE.build_V_hel(
        all_states_DD, all_states_DsDs, per_spin_DD, per_spin_DsDs, V_cross_can)
    V_hel[K_DD+1:K, 1:K_DD] .= V_hel[1:K_DD, K_DD+1:K]'

    # ====================================================================
    # H_raw = T_diag * S_mat + fv_factor * V_hel
    # ====================================================================
    H_raw = T_diag * S_mat + fv_factor * V_hel

    S_eig = eigvals(S_mat)
    S_reg = minimum(abs.(S_eig)) < 1e-10 ? S_mat + 1e-12 * Matrix(1.0I, K, K) : S_mat

    evals_gep = sort(real.(eigvals(Hermitian(H_raw), Hermitian(S_reg))))
    evals_gep = evals_gep[isfinite.(evals_gep) .&& evals_gep .> 100]

    # ====================================================================
    # 按不可约表示分组
    # ====================================================================
    blocks_by_irrep = Dict{String, Vector}()
    for blk in [proj_DD; proj_DsDs]
        Gamma = blk.Gamma
        haskey(blocks_by_irrep, Gamma) || (blocks_by_irrep[Gamma] = [])
        push!(blocks_by_irrep[Gamma], blk)
    end

    block_full_idx = []
    for blk in proj_DD
        push!(block_full_idx, [sd_DD[st] for st in blk.states])
    end
    for blk in proj_DsDs
        push!(block_full_idx, [K_DD + sd_DsDs[st] for st in blk.states])
    end
    all_blocks = [proj_DD; proj_DsDs]

    all_ok = true

    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        total_cols = sum(blk.n_r for blk in blocks)
        total_cols == 0 && continue

        X_big = zeros(ComplexF64, K, total_cols)
        col = 1
        for blk in blocks
            blk_idx = findfirst(b -> b === blk, all_blocks)
            idx = block_full_idx[blk_idx]
            X_big[idx, col:col+blk.n_r-1] .= blk.X
            col += blk.n_r
        end

        H_proj = X_big' * H_raw * X_big
        S_proj = X_big' * S_mat * X_big

        evals_proj = sort(real.(eigvals(Hermitian(H_proj), Hermitian(S_proj + 1e-12*I))))
        evals_proj = evals_proj[isfinite.(evals_proj)]

        matched = 0
        for ep in evals_proj
            if minimum(abs.(ep .- evals_gep)) < 1e-8
                matched += 1
            end
        end

        ok = matched == length(evals_proj)
        all_ok = all_ok && ok
        @test ok
    end

    return all_ok
end

test_dd_dstardstar()
