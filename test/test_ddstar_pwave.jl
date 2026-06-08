# ==========================================================================
# D D* (pseudoscalar-vector) 体系测试 — p-波相互作用
#
# D (s=0) + D* (s=1), 可区分粒子, I=0, κ=("[1]","[1]")
#
# 作用: V_{σ'σ}(p',p) = C * p * p' * f(p')f(p) *
#                        (-1)^{σ'+σ}/3 * Y_{1,-σ'}(ŷ') * Y_{1,-σ}^*(ŷ)
#
# σ 仅指 D* 自旋 (σ_D ≡ 0). D 部分为 δ_{σ'₁,0} δ_{σ₁,0}.
#
# 利用 f_m(n) ≡ |n|·Y_{1,m}(n̂) 为 n 分量的一次齐次多项式:
#   f_0(n)  = √(3/4π) · n_z
#   f_+1(n) = −√(3/8π) · (n_x + i n_y)
#   f_-1(n) = +√(3/8π) · (n_x − i n_y)
#
# 得到 p·p'·Y·Y^* = (2πħc/L)² · f_{-σ'}(n') · f_{-σ}^*(n)
#
# 形状因子: dipole  f(n) = 1/(1 + |n|²/Λ_n²)²
# 检验: 子空间不变性 + 本征值归属性
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

const M_D  = 1864.84
const M_Ds = 2008.50
const ħc   = 197.327

# =========================== f_m 函数 ===========================

# f_m(n) = |n| * Y_{1,m}(n̂)  (Condon-Shortley 相位约定)
function fm_poly(m::Int, n::NPHFforFVE.Momentum)
    nx = Float64(n[1]); ny = Float64(n[2]); nz = Float64(n[3])
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

# =========================== 相互作用 ===========================

function make_V_ddstar(C0, Λ_n², L_phys)
    pref = (2π * ħc / L_phys)^2  # p·p' → 整数动量转换因子
    return function(np, sp, n, s, extra...)
        # D 自旋恒为 0
        sp[1] == 0 && s[1] == 0 || return zero(ComplexF64)

        nDsp = np[2]; nDs = n[2]   # D* 动量 (整数)
        σp   = Int(sp[2]); σ = Int(s[2])  # D* 自旋投影

        # p-波: 任一动量为零则相互作用为零 (f_m(0)=0 自动满足但显式处理)
        val = pref * fm_poly(-σp, nDsp) * conj(fm_poly(-σ, nDs))
        iszero(val) && return zero(ComplexF64)

        # 形状因子
        ff_p = 1.0; ff_n = 1.0
        for n_ in np; ff_p /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end
        for n_ in n;  ff_n /= (1.0 + Float64(sum(abs2, n_)) / Λ_n²)^2; end

        # 相位 + 组合
        phase = (-1.0)^(σp + σ)
        ComplexF64(C0 * ff_p * ff_n * phase / 3 * val)
    end
end

# =========================== 投影 + 检验 ===========================

function test_ddstar()
    L0 = 48; a = 0.1; L_phys = L0 * a
    C0 = 1.5e-12; Λ = 1000.0
    Λ_n² = (Λ / (2π * ħc / L_phys))^2
    Ncut = 20

    println("D D* 体系 — p-波相互作用")
    println("  L=$L0, a=$a fm, Ncut=$Ncut")
    println("  C=$(C0) MeV⁻⁴, Λ=$Λ MeV")
    println("  m_D=$(M_D) MeV, m_D*=$(M_Ds) MeV")

    species = [1, 1]
    particle_types = [:boson, :boson]
    spins = Float64[0.0, 1.0]
    etas  = Float64[1.0, 1.0]
    per_mass = [M_D, M_Ds]
    d_total = NPHFforFVE.Momentum(0,0,0)
    kappa_tuple = ("[1]", "[1]")

    per_spin_r = Rational{Int}[]
    for k in 1:length(species)
        for _ in 1:species[k]
            push!(per_spin_r, Rational{Int}(Int(2*spins[k]), 2))
        end
    end

    pref_T = (2π * ħc / L_phys)^2
    N_α = length(species)
    dd = 3*(N_α + N_α) - 6
    fv_factor = (2π * ħc / L_phys)^(dd/2)

    # ---------- 收集全空间态 + 投影分块 ----------
    reps = NPHFforFVE.find_representatives(length(species);
        Ncut=Ncut, d=d_total, species=species, particle_types=particle_types)

    all_states = []
    state_to_idx = Dict()
    proj_blocks = []  # (X, states, rep, Gamma, n_r)

    for rep in reps
        h_reps = try
            NPHFforFVE.helicity_representatives(rep;
                species=species, particle_types=particle_types,
                spins=spins, d=d_total)
        catch; []; end

        for hel in h_reps
            hel_float = Tuple(Float64.(hel))
            for Gamma in NPHFforFVE.OH_IRREP_NAMES
                result = try
                    NPHFforFVE.subspace_projection(rep, hel_float,
                        kappa_tuple, Gamma;
                        d_total=d_total, species=species,
                        particle_types=particle_types,
                        spins=spins, etas=etas)
                catch; continue; end
                size(result.X,2) == 0 && continue
                for st in result.subspace_states
                    if !haskey(state_to_idx, st)
                        push!(all_states, st)
                        state_to_idx[st] = length(all_states)
                    end
                end
                push!(proj_blocks, (X=result.X, states=result.subspace_states,
                    rep=rep, Gamma=Gamma, n_r=size(result.X,2)))
            end
        end
    end

    K = length(all_states)
    println("全子空间态数: K = $K")
    println("投影分块数: $(length(proj_blocks))")

    # ---------- H_raw ----------
    T_diag = zeros(ComplexF64, K, K)
    for (idx, (n_tup, _)) in enumerate(all_states)
        T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                for (n_, m_) in zip(n_tup, per_mass))
        T_diag[idx, idx] = T
    end

    V_func = make_V_ddstar(C0, Λ_n², L_phys)
    V_hel = NPHFforFVE.build_V_hel(all_states, all_states, per_spin_r, per_spin_r, V_func)
    H_raw = T_diag + fv_factor * V_hel

    # ---------- 按不可约表示检验 ----------
    blocks_by_irrep = Dict{String, Vector}()
    for blk in proj_blocks
        Gamma = blk.Gamma
        if !haskey(blocks_by_irrep, Gamma)
            blocks_by_irrep[Gamma] = []
        end
        push!(blocks_by_irrep[Gamma], blk)
    end

    n_pass = 0; n_fail = 0
    for Gamma in sort(collect(keys(blocks_by_irrep)))
        blocks = blocks_by_irrep[Gamma]
        total_cols = sum(b.n_r for b in blocks)

        X_big = zeros(ComplexF64, K, total_cols); col = 1
        for blk in blocks
            idx = [state_to_idx[st] for st in blk.states]
            X_big[idx, col:col+blk.n_r-1] .= blk.X
            col += blk.n_r
        end

        # 子空间不变性
        V_proj = X_big' * V_hel * X_big
        dv = norm(V_hel * X_big - X_big * V_proj)

        # 本征值归属性
        H_proj = X_big' * H_raw * X_big
        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_gen  = sort(real.(eigvals(H_raw, Matrix(1.0I, K, K))))
        evals_gen  = evals_gen[isfinite.(evals_gen) .&& evals_gen .> 100]
        matched = count(ep -> minimum(abs.(ep .- evals_gen)) < 1e-8, evals_proj)

        ok = dv < 1e-10 && matched == length(evals_proj)
        status = ok ? "✓" : "✗ FAIL"
        if ok; n_pass += 1; else; n_fail += 1; end

        detail = ok ? "" : "  dv=$(round(dv,digits=1)) attrib=$(matched)/$(length(evals_proj))"
        println("  $Gamma: dim=$total_cols, K=$K  $status$detail")
    end

    println("  通过: $n_pass, 失败: $n_fail")
    println("\n$(repeat("=", 60))")
    println(n_fail == 0 ? "全部测试通过 ✓" : "存在失败测试 ✗")
end

test_ddstar()
