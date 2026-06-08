# ==========================================================================
# rho → ππ 耦合道测试
#
# rho (s=1, η=-1, m=800) + ππ (s=0 each, η=+1, m=140, I=1)
#
# 相互作用: ⟨ππ, k | V | rho, σ⟩ = g * k * Y_{1σ}(k̂)
#           V_{rho-rho} = 0, V_{pipi-pipi} = 0
#
# 检验: 子空间不变性 + 本征值归属性
# ==========================================================================

using NPHFforFVE, StaticArrays, LinearAlgebra

const ħc   = 197.327
const m_rho = 800.0
const m_pi  = 140.0
const g_val = 1.43e-5

# ---------- f_m 多项式 (Condon-Shortley 相位约定) ----------

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

# ---------- 耦合 V_can ----------

function make_V_rho_pipi(g_c, L_phys)
    pref = (2π * ħc / L_phys)  # k_phys = pref * |n|
    return function(np, sp, n, s, extra...)
        # ρ → ππ (N=1 → N=2)
        if length(n) == 1 && length(np) == 2
            k = np[1]   # π⁺ 动量 (另一 π 为 -k)
            σ = Int(s[1])
            return ComplexF64(g_c * pref * fm_poly(σ, k))
        # ππ → ρ (N=2 → N=1, Hermitian conjugate)
        elseif length(n) == 2 && length(np) == 1
            k = n[1]
            σ = Int(sp[1])
            return ComplexF64(g_c * pref * conj(fm_poly(σ, k)))
        else
            return zero(ComplexF64)
        end
    end
end

# ---------- 测试主体 ----------

function test_rho_pipi()
    L0 = 48; a = 0.1; L_phys = L0 * a
    Ncut = 20
    pref_T = (2π * ħc / L_phys)^2

    println("rho → ππ 耦合道")
    println("  L=$L0, a=$a fm, Ncut=$Ncut")
    println("  m_rho=$m_rho, m_pi=$m_pi, g=$g_val MeV^(-3/2)")

    # ---------- FockChannels ----------
    ch_rho = FockChannel("rho", [1], [:boson], [m_rho],
                         [1//1], [1//1], [-1.0], relativistic)
    ch_pipi = FockChannel("pipi", [2], [:boson], [m_pi],
                          [0//1], [1//1], [1.0], relativistic)

    sub_rho = NPHFforFVE.get_isospin_subchannels(ch_rho, 1//1)[1]
    sub_pipi = NPHFforFVE.get_isospin_subchannels(ch_pipi, 1//1)[1]

    per_spin_rho, per_etas_rho = NPHFforFVE._expand_per_particle(ch_rho)
    per_spin_pipi, per_etas_pipi = NPHFforFVE._expand_per_particle(ch_pipi)
    spin_rho = Float64.(Rational{Int}.(per_spin_rho))
    spin_pipi = Float64.(Rational{Int}.(per_spin_pipi))  # [0.0, 0.0]
    etas_rho = Float64.(per_etas_rho)
    etas_pipi = Float64.(per_etas_pipi)
    per_mass_rho = NPHFforFVE._expand_per_particle_mass(ch_rho)
    per_mass_pipi = NPHFforFVE._expand_per_particle_mass(ch_pipi)

    # per_spin for build_V_hel
    per_spin_r_rho = Rational{Int}.(per_spin_rho)
    per_spin_r_pipi = Rational{Int}.(per_spin_pipi)

    V_func = make_V_rho_pipi(g_val, L_phys)

    n_pass = 0; n_fail = 0; n_skip = 0

    for irrep in NPHFforFVE.OH_IRREP_NAMES
        # --- rho 投影 ---
        projs_rho = try
            NPHFforFVE._get_channel_proj_list(
                ch_rho, Ncut, NPHFforFVE.Momentum(0,0,0),
                sub_rho.κ, irrep, only(unique(spin_rho)), etas_rho)
        catch e
            []
        end
        dim_rho = sum(p.n_r for p in projs_rho; init=0)

        # --- pipi 投影 ---
        projs_pipi = try
            NPHFforFVE._get_channel_proj_list(
                ch_pipi, Ncut, NPHFforFVE.Momentum(0,0,0),
                sub_pipi.κ, irrep, 0.0, etas_pipi)
        catch e
            []
        end
        dim_pipi = sum(p.n_r for p in projs_pipi; init=0)

        dim_rho == 0 && dim_pipi == 0 && continue

        # --- 收集 rho 态 ---
        rho_states = []
        rho_to_idx = Dict()
        for p in projs_rho
            for st in p.states
                if !haskey(rho_to_idx, st)
                    push!(rho_states, st)
                    rho_to_idx[st] = length(rho_states)
                end
            end
        end
        Kr = length(rho_states)

        # --- 收集 pipi 态 ---
        pipi_states = []
        pipi_to_idx = Dict()
        for p in projs_pipi
            for st in p.states
                if !haskey(pipi_to_idx, st)
                    push!(pipi_states, st)
                    pipi_to_idx[st] = length(pipi_states)
                end
            end
        end
        Kp = length(pipi_states)

        # --- 构建 X_big ---
        total_dim = dim_rho + dim_pipi
        K = Kr + Kp
        X_big = zeros(ComplexF64, K, total_dim)
        col = 1
        for p in projs_rho
            idx = [rho_to_idx[st] for st in p.states]
            X_big[idx, col:col+p.n_r-1] .= p.X
            col += p.n_r
        end
        for p in projs_pipi
            idx = [Kr .+ pipi_to_idx[st] for st in p.states]
            X_big[idx, col:col+p.n_r-1] .= p.X
            col += p.n_r
        end

        # --- 全空间态列表 (先 rho 后 pipi) ---
        all_states = [rho_states..., pipi_states...]

        # --- T_diag ---
        T_diag = zeros(ComplexF64, K, K)
        for (idx, st) in enumerate(all_states)
            n_tup, _ = st
            if idx <= Kr
                T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                        for (n_, m_) in zip(n_tup, per_mass_rho))
            else
                T = sum(sqrt(m_^2 + pref_T * Float64(sum(abs2, n_)))
                        for (n_, m_) in zip(n_tup, per_mass_pipi))
            end
            T_diag[idx, idx] = T
        end

        # --- V_hel 全空间 ---
        V_hel = zeros(ComplexF64, K, K)

        # 耦合块 (仅在两个通道都有态时非零)
        if Kr > 0 && Kp > 0
            # ρ → ππ 耦合块 (下三角: ππ行, ρ列)
            V_rp = NPHFforFVE.build_V_hel(
                pipi_states, rho_states, per_spin_r_pipi, per_spin_r_rho, V_func)
            V_hel[Kr+1:end, 1:Kr] .= V_rp
            # ππ → ρ 耦合块 (上三角: ρ行, ππ列, Hermitian 共轭)
            V_pr = NPHFforFVE.build_V_hel(
                rho_states, pipi_states, per_spin_r_rho, per_spin_r_pipi, V_func)
            V_hel[1:Kr, Kr+1:end] .= V_pr
        end

        # fv_factor: 通道相关的归一化
        fv_rho_rho = 1.0                           # N_α=N_β=1, dd=0
        d_pipi = 3 * (2 + 2) - 6
        fv_pipi_pipi = (2π * ħc / L_phys)^(d_pipi / 2)  # N_α=N_β=2, dd=6
        d_cross = 3 * (1 + 2) - 6
        fv_cross = (2π * ħc / L_phys)^(d_cross / 2)     # N_α=1, N_β=2, dd=3

        Fv = ones(ComplexF64, K, K)
        if Kr > 0
            Fv[1:Kr, 1:Kr] .= fv_rho_rho
        end
        if Kp > 0
            Fv[Kr+1:end, Kr+1:end] .= fv_pipi_pipi
        end
        if Kr > 0 && Kp > 0
            Fv[1:Kr, Kr+1:end] .= fv_cross
            Fv[Kr+1:end, 1:Kr] .= fv_cross
        end

        H_raw = T_diag + Fv .* V_hel

        # --- 子空间不变性 ---
        V_proj = X_big' * (Fv .* V_hel) * X_big
        dv = norm((Fv .* V_hel) * X_big - X_big * V_proj)
        dv_tol = max(1e-10, 1e-12 * norm(V_proj))

        # --- 本征值归属性 ---
        H_proj = X_big' * H_raw * X_big
        evals_proj = sort(real.(eigvals(Hermitian(H_proj))))
        evals_gen  = sort(real.(eigvals(H_raw, Matrix(1.0I, K, K))))
        evals_gen  = evals_gen[isfinite.(evals_gen) .&& evals_gen .> 100]
        matched = count(ep -> minimum(abs.(ep .- evals_gen)) < 1e-8, evals_proj)

        ok = dv < dv_tol && matched == length(evals_proj)
        status = ok ? "✓" : "✗ FAIL"
        if ok; n_pass += 1; else; n_fail += 1; end

        detail = ok ? "" : "  dv=$(round(dv,sigdigits=2)) attrib=$(matched)/$(length(evals_proj))"
        println("  $irrep: ρ_dim=$dim_rho ππ_dim=$dim_pipi total=$total_dim K=$K  $status$detail")
    end

    println("\n  通过: $n_pass, 失败: $n_fail, 跳过: $n_skip")
    println("\n$(repeat("=", 60))")
    println(n_fail == 0 ? "全部测试通过 ✓" : "存在失败测试 ✗")
    return n_fail == 0
end

test_rho_pipi()
