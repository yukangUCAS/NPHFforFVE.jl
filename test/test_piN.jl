# ==========================================================================
# πN → πN 单道系统测试
#
# π: s=0, I=1, boson
# N: s=1/2, I=1/2, fermion
# 可区分粒子 → species=[1,1], κ=("[1]","[1]"), S=I
#
# 作用: <πN,σ'|V|πN,σ> = C * δ_{σ',σ}   (正则自旋基)
# C = 1×10⁻⁵ MeV⁻²,  I=1/2 和 I=3/2 相同
# 形状因子: dipole, Λ=1000 MeV
#
# 双覆盖群 2O_h (96 元素): 10 玻色子 + 6 费米子不可约表示
# πN 系统: spatial ⊗ spin-1/2 → 仅费米子不可约表示 (G1±, G2±, H±)
#
# H_raw = T_diag + fv_factor * V_hel  (S=I)
# 验证: eig(H_proj) ⊂ eig(H_raw)
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra, Test

function test_piN()
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
    Λ_n²    = (Λ / (2π * ħc_val / L_phys))^2
    fv_factor = (2π * ħc_val / L_phys)^3

    ff(n::Momentum) = 1.0 / (1.0 + sum(abs2, n) / Λ_n²)^2

    d_total = M(0,0,0)
    irrep_names = NPHFforFVE.OH2_IRREP_NAMES

    species = [1, 1]
    particle_types = [:boson, :fermion]
    spins = Float64[0.0, 0.5]
    etas  = Float64[1.0, 1.0]
    N_α = 2

    per_spin_r = Rational{Int}[0//1, 1//2]
    per_mass = [m_π, m_N]
    kappa_tuple = ("[1]", "[1]")

    # ---------- ZM+spin 检测 ----------
    function has_zm_spin(rep)
        for i in 1:N_α
            if iszero(rep[i]) && spins[i] != 0.0
                return true
            end
        end
        return false
    end

    # ---------- V_can ----------
    function V_can_func(np, sp, n, s, extra...)
        σ_π_p, σ_N_p = sp
        σ_π, σ_N = s
        σ_π_p == σ_π || return zero(ComplexF64)
        σ_N_p == σ_N || return zero(ComplexF64)
        n1p, n2p = np; n1, n2 = n
        ComplexF64(C0 * ff(n1p)*ff(n1) * ff(n2p)*ff(n2))
    end

    # ====================================================================
    # 代表态
    # ====================================================================
    reps = NPHFforFVE.find_representatives(N_α; Ncut=Ncut, d=d_total,
        species=species, particle_types=particle_types)

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

    # ====================================================================
    # T_diag
    # ====================================================================
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass))
        T_diag[idx, idx] = T
    end

    # ====================================================================
    # V_hel + H_raw (S=I, 可区分粒子)
    # ====================================================================
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states,
                                       per_spin_r, per_spin_r, V_can_func)

    H_raw = T_diag + fv_factor * V_hel
    evals_full = sort(real.(eigvals(Hermitian(H_raw))))
    evals_full = evals_full[isfinite.(evals_full)]

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

    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        total_cols = sum(blk.n_r for blk in blocks)
        total_cols == 0 && continue

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
    end

    return all_ok
end

test_piN()
