# ============================================================
# UserAPI — 面向用户的高层 API
#
# 工作流:
#   proj = Project(d, I, channels, Ncuts)        # 阶段 A
#   idx = add_config!(proj, L, a, irreps, n)      # 阶段 B (重复)
#   ... 用户提供 V_func, @params ...
#   result = compute!(proj, V_func, params)        # 阶段 D
#   result[idx]["T1-"]                             # 取值
# ============================================================

import .FockSpace: FockSystem, FockChannel, get_Ncut

"""
    Config

一次 `add_config!` 调用对应的组态配置。

字段:
- L::Int          格点数
- a::Float64      格距 (fm)
- irreps          需要的不可约表示列表
- n_levels        每个不可约表示所需最低能级数 (与 irreps 一一对应)
"""
struct Config
    L::Int
    a::Float64
    irreps::Vector{String}
    n_levels::Vector{Int}
end

"""
    Project

一个完整的格点能谱计算项目。

构建后通过 `add_config!` 逐一添加 (L,a,Γ,n) 组态，最后调用 `compute!` 一键计算所有组态的能谱。

字段:
- d::Momentum              总动量
- I::Rational{Int}         总同位旋
- channels::Vector{FockChannel}  道列表
- Ncuts::Vector{Int}       每道 Ncut (与 channels 一一对应; N=1 的道 Ncut 忽略)
- configs::Vector{Config}   阶段 B 逐一添加的组态列表
"""
mutable struct Project
    d::Momentum
    I::Rational{Int}
    channels::Vector{FockChannel}
    Ncuts::Vector{Int}
    configs::Vector{Config}
end

"""
    Project(d, I, channels, Ncuts) -> Project

创建项目（阶段 A）。

# 参数
- `d`: 总动量 (Momentum 或 3 元组)
- `I`: 总同位旋 (Rational{Int})
- `channels`: FockChannel 列表
- `Ncuts`: 每道动量截断，长度与 channels 一致。N=1 的道 Ncut 被忽略（可填 0）。

# 示例
```julia
ch_rho  = FockChannel("rho",  [1], [:boson], [800.0], [1//1], [1//1], [-1.0], relativistic)
ch_pipi = FockChannel("pipi", [2], [:boson], [140.0], [0//1], [1//1], [-1.0], relativistic)
proj = Project(Momentum(0,0,0), 1//1, [ch_rho, ch_pipi], [0, 8])
```
"""
function Project(d, I::Rational{Int}, channels::Vector{FockChannel}, Ncuts::Vector{Int})
    n_ch = length(channels)
    length(Ncuts) == n_ch ||
        throw(ArgumentError("Ncuts 长度 ($(length(Ncuts))) 必须与 channels 长度 ($n_ch) 一致"))

    for i in 1:n_ch
        n = Ncuts[i]
        if channels[i].N > 1 && n < 1
            throw(ArgumentError("道 \"$(channels[i].name)\" (N=$(channels[i].N)) 的 Ncut 必须 ≥ 1，传入 $n"))
        end
    end

    d_mom = d isa Momentum ? d : Momentum(d...)
    return Project(d_mom, I, channels, Ncuts, Config[])
end

"""
    add_config!(proj::Project, L::Int, a, irreps, n_levels) -> Int

添加一个 (L,a,Γ,n) 组态（阶段 B）。返回该组态的索引（1-based），后续通过 `result[idx]` 获取能级。

# 参数
- `L`: 有限体积格点数
- `a`: 格距 (fm)
- `irreps`: 需要计算的 O_h 不可约表示列表
- `n_levels`: 每个不可约表示所需的最低能级数 (长度与 irreps 一致)

# 返回
组态索引 (1-based)，用于从 `compute!` 返回的结果中取值。

# 示例
```julia
idx1 = add_config!(proj, 48, 0.1, ["T1-","A2-"], [5,1])
idx2 = add_config!(proj, 64, 0.08, ["T1-"], [3])
```
相同 (L,a) 的组态在 `compute!` 内会自动共享几何缓存，无需手动处理。
"""
function add_config!(proj::Project, L::Int, a::Real,
                     irreps::Vector{String}, n_levels::Vector{Int})
    L > 0 || throw(ArgumentError("L 必须 > 0，传入 $L"))
    a > 0 || throw(ArgumentError("a 必须 > 0，传入 $a"))
    length(irreps) == length(n_levels) ||
        throw(ArgumentError("irreps 和 n_levels 长度必须一致"))

    for Γ in irreps
        (Γ in OH_IRREP_NAMES || Γ in OH2_IRREP_NAMES) ||
            throw(ArgumentError("无效的不可约表示: $Γ"))
    end

    push!(proj.configs, Config(L, Float64(a), irreps, n_levels))
    return length(proj.configs)
end

# ============================================================
# 阶段 D: 计算与结果
# ============================================================

"""
    ProjectResult

`compute!` 的返回类型。通过索引获取各 Config 的能谱。

# 访问
- `result[idx]` → `Dict{String, Vector{Float64}}`（第 idx 个组态的能谱）
- `length(result)` → 组态总数
"""
struct ProjectResult
    configs::Vector{Config}
    spectra::Vector{Dict{String, Vector{Float64}}}
end

Base.length(r::ProjectResult) = length(r.spectra)

function Base.getindex(r::ProjectResult, idx::Int)
    1 <= idx <= length(r.spectra) || throw(BoundsError(r, idx))
    return r.spectra[idx]
end

"""
    compute!(proj::Project, V_func, params) -> ProjectResult

计算所有组态的能谱。相同 (L,a) 的组态自动共享几何缓存和 V_hel，
无需用户手动处理。

# 参数
- `proj`: 已完成阶段 A+B 的 Project
- `V_func`: 相互作用函数，签名同 `build_hamiltonian_block` 要求
- `params`: V_func 的参数（@params struct 实例）

# 返回
`ProjectResult`，通过 `result[idx]` 获取第 idx 个组态的 `Dict{String, Vector{Float64}}`。

# 示例
```julia
result = compute!(proj, V_rho_pipi, RhoPiPiParams(g=1.43e-5))
result[1]          # Dict("T1-" => [...], "A2-" => [...])
result[1]["T1-"]   # T1- 能级向量
```
"""
function compute!(proj::Project, V_func, params)
    isempty(proj.configs) &&
        throw(ArgumentError("请先用 add_config! 添加至少一个组态"))

    n_ch = length(proj.channels)
    n_configs = length(proj.configs)

    # N=1 道的 Ncut 为 dummy（被忽略），确保 FockSystem 构造不报错
    ncuts_sys = Int[proj.Ncuts[i] > 0 ? proj.Ncuts[i] : 1 for i in 1:n_ch]

    # 按 (L,a) 分组，同组共享 SystemBasis
    groups = Dict{Tuple{Int, Float64}, Vector{Int}}()
    for (i, cfg) in enumerate(proj.configs)
        key = (cfg.L, cfg.a)
        idxs = get!(Vector{Int}, groups, key)
        push!(idxs, i)
    end

    spectra = Vector{Dict{String, Vector{Float64}}}(undef, n_configs)

    for ((L, a), config_indices) in groups
        # 组内所有 irreps 的并集
        all_irreps = String[]
        for idx in config_indices
            for Gamma in proj.configs[idx].irreps
                Gamma in all_irreps || push!(all_irreps, Gamma)
            end
        end

        sys = FockSystem(proj.d, 1, proj.channels, L, a, proj.I, all_irreps;
                         Ncut_channel=ncuts_sys)
        basis = SystemBasis(sys)
        build_V_hel_blocks!(basis, V_func, params)

        for idx in config_indices
            cfg = proj.configs[idx]
            n_levels = Dict(cfg.irreps[i] => cfg.n_levels[i]
                            for i in 1:length(cfg.irreps))
            spectra[idx] = compute_spectrum(basis; n_levels=n_levels)
        end
    end

    return ProjectResult(proj.configs, spectra)
end

# ============================================================
# 交互模式
# ============================================================

"""
    setup_project() -> Project

REPL 交互式构建 Project。分两阶段：

阶段 A — 输入总动量 d、总同位旋 I、各 Fock 道及其 Ncut。
阶段 B — 循环添加 (L,a,Γ,n_Γ) 组态，输入空白 L 结束。

返回带有所有组态的 Project，可直接传入 `compute!`。

# 示例
```julia
proj = setup_project()
result = compute!(proj, my_V, my_params)
result[1][\"T1-\"]   # 第 1 个组态的 T1- 能级
```
"""
function setup_project()
    println("="^56)
    println("  NPHFforFVE Project 交互构建")
    println("="^56)

    # ===== 阶段 A: d, I, channels, Ncuts =====
    println("\n── 阶段 A: 物理系统定义 ──")

    print("总动量 d (格式: nx ny nz, 默认 0 0 0): ")
    d_input = strip(readline())
    d = if isempty(d_input)
        D000
    else
        parts = parse.(Int, split(d_input))
        length(parts) == 3 || throw(ArgumentError("总动量必须是 3 个整数"))
        Momentum(parts...)
    end
    println("  d = $d")

    print("总同位旋 I (如 0, 1/2, 1, 默认 0): ")
    I_input = strip(readline())
    I = isempty(I_input) ? 0//1 : _parse_rational(I_input)
    println("  I = $I")

    print("\nFock 道数: ")
    n_ch = parse(Int, readline())
    n_ch >= 1 || throw(ArgumentError("至少需要 1 个道"))

    channels = FockChannel[]
    Ncuts = Int[]

    for i in 1:n_ch
        println("\n--- 道 $i ---")
        ch = _prompt_fock_channel(i)
        push!(channels, ch)
        if ch.N > 1
            print("  该道 Ncut (|n|² ≤ Ncut): ")
            nc = parse(Int, readline())
            nc >= 1 || throw(ArgumentError("N>1 道的 Ncut 必须 ≥ 1"))
            push!(Ncuts, nc)
        else
            println("  N=1 道，Ncut 自动设为 0（动量被 d 锁定）")
            push!(Ncuts, 0)
        end
    end

    proj = Project(d, I, channels, Ncuts)
    println("\n✓ 阶段 A 完成: $(n_ch) 个道")

    # ===== 阶段 B: configs =====
    println("\n── 阶段 B: 添加组态 (L,a,Γ,n_Γ) ──")
    println("可选不可约表示: $(join(OH_IRREP_NAMES, ", "))")
    println("输入空白 L 结束添加\n")

    while true
        print("L (格点数, 空白结束): ")
        L_input = strip(readline())
        isempty(L_input) && break
        L = parse(Int, L_input)

        print("a (格距 fm): ")
        a = parse(Float64, readline())

        print("需要的不可约表示 (空格分隔): ")
        irr_input = strip(readline())
        isempty(irr_input) && throw(ArgumentError("至少需要一个不可约表示"))
        irreps = String[String(s) for s in split(irr_input)]

        print("对应能级数 (空格分隔, 共 $(length(irreps)) 个): ")
        n_input = strip(readline())
        n_levels = parse.(Int, split(n_input))

        idx = add_config!(proj, L, a, irreps, n_levels)
        println("  ✓ 已添加组态 #$idx ($(length(irreps)) 个 irrep)\n")
    end

    isempty(proj.configs) && @warn("未添加任何组态，compute! 将报错")

    # ===== 汇总 =====
    println("="^56)
    println("Project 构建完成")
    println("  总动量 d = $(proj.d)")
    println("  总同位旋 I = $(proj.I)")
    println("  道数: $(length(proj.channels))")
    for (i, ch) in enumerate(proj.channels)
        println("    道 $i: \"$(ch.name)\" N=$(ch.N) Ncut=$(proj.Ncuts[i])")
    end
    println("  组态数: $(length(proj.configs))")
    for (i, cfg) in enumerate(proj.configs)
        println("    #$i: L=$(cfg.L), a=$(cfg.a), Γ=$(join(cfg.irreps, ","))")
    end
    println("="^56)
    println("\n下一步: result = compute!(proj, V_func, params)")
    println("        result[idx][\"Γ\"] 获取能级")

    println()
    path = generate_potential_template(proj)
    println("✓ 模板已生成: $path")
    println("请在模板中填写 my_V(...) 矩阵元和 MyParams 参数")

    return proj
end

# ============ 交互辅助 ============

function _prompt_fock_channel(i::Int)
    print("  道名 (如 \"rho\"): ")
    name = strip(readline())
    isempty(name) && throw(ArgumentError("道名不能为空"))

    print("  粒子种类数: ")
    n_species = parse(Int, readline())
    n_species >= 1 || throw(ArgumentError("粒子种类数 ≥ 1"))

    species = Int[]
    ptypes  = Symbol[]
    masses  = Float64[]
    spins   = Rational{Int}[]
    isospins = Rational{Int}[]
    etas    = Float64[]

    for s in 1:n_species
        println("  粒子种类 $s:")
        print("    粒子数: ")
        push!(species, parse(Int, readline()))
        print("    类型 (boson/fermion): ")
        pt = Symbol(lowercase(strip(readline())))
        pt in (:boson, :fermion) || throw(ArgumentError("类型必须是 boson 或 fermion"))
        push!(ptypes, pt)
        print("    质量 (MeV): ")
        push!(masses, parse(Float64, readline()))
        print("    自旋 j (0, 1/2, 1): ")
        push!(spins, _parse_rational(readline()))
        print("    同位旋 j (0, 1/2, 1): ")
        push!(isospins, _parse_rational(readline()))
        print("    内禀宇称 η (+1/-1): ")
        push!(etas, parse(Float64, readline()))
    end

    print("  动能色散关系 (relativistic/nonrelativistic, 默认 relativistic): ")
    kt_input = strip(readline())
    kt = if isempty(kt_input) || lowercase(kt_input) == "relativistic"
        relativistic
    elseif lowercase(kt_input) == "nonrelativistic"
        nonrelativistic
    else
        throw(ArgumentError("色散关系必须是 relativistic 或 nonrelativistic"))
    end

    return FockChannel(name, species, ptypes, masses, spins, isospins, etas, kt)
end

function _parse_rational(s::AbstractString)
    s = strip(s)
    if occursin("//", s)
        parts = split(s, "//")
        length(parts) == 2 || throw(ArgumentError("无法解析有理数: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    elseif occursin('/', s)
        parts = split(s, '/')
        length(parts) == 2 || throw(ArgumentError("无法解析有理数: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    else
        return parse(Int, s) // 1
    end
end

# ============================================================
# V 函数模板生成
# ============================================================

"""
    generate_potential_template(proj::Project) -> String

根据 Project 中的道列表和总同位旋，自动生成 V 函数模板文件 `potential_defs.jl`。

包含:
- `@params` 参数结构体骨架
- 每对道 (α,β) 的 `my_V_αβ` 函数骨架（含同位旋子道分支）
- `param_bounds` 覆盖模板

用户只需填写矩阵元物理内容即可。

返回生成的文件路径。
"""
function generate_potential_template(proj::Project)
    path = "potential_defs.jl"
    io = open(path, "w")

    I = proj.I
    n_ch = length(proj.channels)

    # ===== 文件头 =====
    println(io, "# ============================================================")
    println(io, "# Interaction potential V — user-defined matrix elements")
    println(io, "# ============================================================")
    println(io, "#")
    println(io, "# Usage:")
    println(io, "#   1. Edit the MyParams struct below with your LEC / cutoff params.")
    println(io, "#")
    println(io, "#       p0 = MyParams()                       # all defaults")
    println(io, "#       p0 = MyParams(; C0=2.0)               # override C0")
    println(io, "#       x  = to_vector(p0)                    # -> Vector{Float64}")
    println(io, "#       names = param_names(MyParams)         # -> [\"C0\", \"C1\", ...]")
    println(io, "#       p_best = from_vector(MyParams, x_fit) # after JuMinuit")
    println(io, "#")
    println(io, "#   2. Fill in the my_V function below.")
    println(io, "#      Dispatch on (chA, chB) to select the channel pair,")
    println(io, "#      then use (kapA, kapB, rA, rB, aA, aB) for isospin sub-channel.")
    println(io, "#")
    println(io, "#   3. Define param_bounds / param_errors overrides at the bottom.")
    println(io, "#")
    println(io, "# Project summary:")
    println(io, "#   total d = $(proj.d)")
    println(io, "#   total I = $(proj.I)")
    for (i, ch) in enumerate(proj.channels)
        println(io, "#   ch $i: \"$(ch.name)\" N=$(ch.N) species=$(ch.species)")
    end
    for (i, cfg) in enumerate(proj.configs)
        println(io, "#   config $i: L=$(cfg.L), a=$(cfg.a), Γ=$(join(cfg.irreps, ","))")
    end
    println(io, "#")
    println(io, "# NOTE: The template auto-converts integer n → physical momentum p (MeV).")
    println(io, "#       ħc = 197.327 MeV·fm,  p = (2πħc / L_phys) · n")
    println(io, "#       pA[i] and pB[i] are pre-computed SVector{3,Float64} — use these")
    println(io, "#       directly when writing matrix elements.")
    println(io, "# ============================================================")
    println(io)
    println(io, "using NPHFforFVE")
    println(io)

    # ===== @params =====
    println(io, "# ============ Parameter struct ============")
    println(io, "# Add your LECs, cutoffs etc. below.")
    println(io, "@params struct MyParams")
    println(io, "    # Examples (uncomment and edit):")
    println(io, "    # C0  = 1.0    # leading-order contact")
    println(io, "    # C1  = 0.5    # NLO contact")
    println(io, "    # Λ   = 1000.0 # cutoff (MeV)")
    println(io, "end")
    println(io)

    # ===== 统一 V 函数 =====
    println(io, "# ============ V matrix elements ============")
    println(io, "#")
    println(io, "# Signature:")
    println(io, "#   my_V(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, params)")
    println(io, "#")
    println(io, "#   nA, nB : bra / ket momentum tuple  NTuple{N, Momentum}")
    println(io, "#   sp, s  : bra / ket spin projections  NTuple{N, Rational{Int}}")
    println(io, "#   kapA, kapB : S_N irrep label (String, or Tuple for multi-species)")
    println(io, "#   rA, rB : multiplicity index      aA, aB : irrep column index")
    println(io, "#   chA, chB : channel index (see table below)")
    println(io, "#   params : MyParams instance")
    println(io, "#")
    println(io, "# IMPORTANT: V must be Hermitian: my_V(nA, nB, ..., chA, chB, L_phys, params)")
    println(io, "#            = conj(my_V(nB, nA, ..., chB, chA, L_phys, params))")
    println(io)

    # 道索引表
    _write_channel_table(io, proj, I)
    println(io)

    # 检测运动系
    is_moving = proj.d != D000

    # 生成按 (chA, chB) 分发的统一函数
    println(io, "function my_V(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, params)")
    println(io, "    # ===== Convert integer n → physical momentum p (MeV) =====")
    println(io, "    # ħc = 197.327 MeV·fm")
    println(io, "    pv = 2π * 197.327 / L_phys")

    if is_moving
        # 运动系：每道展开的每粒子质量 + boost
        println(io)
        println(io, "    # ── Lorentz boost to CM frame (moving system) ──")
        println(io, "    # Per-channel per-particle masses:")
        println(io, "    _masses_by_ch = [")
        for (i, ch) in enumerate(proj.channels)
            expanded = Float64[]
            for (s, m) in zip(ch.species, ch.masses)
                append!(expanded, fill(m, s))
            end
            println(io, "        $(expanded),  # ch $(i): $(ch.name)")
        end
        println(io, "    ]")
        println(io, "    _d_tot = $(proj.d)")
        println(io)
        println(io, "    pA_mov = [pv .* Float64.(n) for n in nA]")
        println(io, "    pB_mov = [pv .* Float64.(n) for n in nB]")
        println(io, "    pA, facA = boost_to_cm(pA_mov, _masses_by_ch[chA], _d_tot, L_phys)")
        println(io, "    pB, facB = boost_to_cm(pB_mov, _masses_by_ch[chB], _d_tot, L_phys)")
    else
        println(io, "    # n is dimensionless integer;  p = (2πħc / L_phys) · n")
        println(io, "    # pA[i], pB[i] are SVector{3,Float64} — ready for dot/cross/norm")
        println(io, "    pA = [pv .* Float64.(n) for n in nA]")
        println(io, "    pB = [pv .* Float64.(n) for n in nB]")
        println(io, "    facA = 1.0; facB = 1.0")
    end
    println(io)

    # 返回值包装: 运动系需乘 kinematic factor
    ret_wrap = is_moving ? ("facA * (", ") * facB") : ("", "")

    first_block = true
    emitted_hermitian = Set{Tuple{Int,Int}}()  # 标记已由 Hermitian 注释处理的 (chA,chB) 对

    for α in 1:n_ch, β in 1:n_ch
        (α, β) in emitted_hermitian && continue
        ch_α = proj.channels[α]
        ch_β = proj.channels[β]
        subs_α = get_isospin_subchannels(ch_α, I)
        subs_β = get_isospin_subchannels(ch_β, I)
        multi_α = length(ch_α.species) > 1
        multi_β = length(ch_β.species) > 1

        keyword = first_block ? "if" : "elseif"
        arrow = "←"

        if α == β
            # 对角块
            println(io, "    # ── $(ch_α.name) ($(ch_α.N)-body) diagonal ──")
            println(io, "    $keyword chA == $α && chB == $β")
            _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                     multi_α, multi_β, α, β, arrow;
                                     ret_prefix=ret_wrap[1], ret_suffix=ret_wrap[2])
        else
            # α<β: 写 chA←chB，chB←chA 标为 Hermitian conjugate
            other = (β, α)
            push!(emitted_hermitian, other)

            println(io, "    # ── $(ch_α.name) ← $(ch_β.name) ──")
            println(io, "    $keyword chA == $α && chB == $β")
            _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                     multi_α, multi_β, α, β, arrow;
                                     ret_prefix=ret_wrap[1], ret_suffix=ret_wrap[2])

            println(io)
            keyword = "elseif"
            println(io, "    # ── $(ch_β.name) ← $(ch_α.name)  (Hermitian conjugate of $α←$β) ──")
            println(io, "    $keyword chA == $β && chB == $α")
            _write_hermitian_conj_body(io, α, β)  # Hermitian: 不包装 (递归调用已有因子)
        end

        first_block = false
    end

    println(io, "    end")
    println(io, "    error(\"unreachable: no matching channel pair for chA=\$chA chB=\$chB\")")
    println(io, "end")
    println(io)

    # ===== 边界覆盖 =====
    println(io, "# ============ JuMinuit bounds ============")
    println(io, "# Override param_bounds for MyParams. Each tuple is (lower, upper).")
    println(io, "# function param_bounds(::Type{MyParams})")
    println(io, "#     return [(0.0, Inf), (0.0, Inf)]")
    println(io, "# end")
    println(io)
    println(io, "# Override param_errors for initial step sizes.")
    println(io, "# function param_errors(::Type{MyParams})")
    println(io, "#     return [0.1, 0.1]")
    println(io, "# end")

    close(io)
    return abspath(path)
end

# ===== 道索引表 =====

function _write_channel_table(io, proj, I)
    # 紧凑索引表
    println(io, "# ╔════╤═══════╤════╤══════════╤══════════════════════════════════╗")
    println(io, "# ║ ch │ name  │ N  │ species  │ κ  (I=$(I))                     ║")
    println(io, "# ╟────┼───────┼────┼──────────┼──────────────────────────────────╢")
    for (i, ch) in enumerate(proj.channels)
        subs = get_isospin_subchannels(ch, I)
        kappas = join(unique(s.κ for s in subs), ", ")
        println(io, "# ║ $i   │ \"$(ch.name)\"  │ $(ch.N)  │ $(ch.species)     │ $(kappas)")
    end
    println(io, "# ╚════╧═══════╧════╧══════════╧══════════════════════════════════╝")
    println(io)
    # 每道详细信息
    println(io, "# Channel details:")
    for (i, ch) in enumerate(proj.channels)
        kt_str = ch.kinetic_type == nonrelativistic ? "nonrelativistic" : "relativistic"
        println(io, "#   ch $i: \"$(ch.name)\" ($(ch.N)-body, $kt_str)")
        for s in 1:length(ch.species)
            pt = ch.particle_types[s] == :boson ? "boson" : "fermion"
            println(io, "#     sp.$s: N=$(ch.species[s]), $pt, m=$(ch.masses[s]) MeV, j=$(ch.spins[s]), I=$(ch.isospins[s]), η=$(ch.etas[s])")
        end
    end
end

# ===== 道对分支体 =====

function _write_channel_pair_body(io, ch_α, ch_β, subs_α, subs_β,
                                  multi_α, multi_β, α, β, arrow;
                                  ret_prefix="", ret_suffix="")
    # 空子道保护: 该道在给定 I 下无同位旋子道
    if isempty(subs_α) || isempty(subs_β)
        name = α == β ? ch_α.name : "$(ch_α.name)←$(ch_β.name)"
        println(io, "        # $name: no isospin sub-channels at this I — V ≡ 0")
        println(io, "        return $(ret_prefix)0.0$(ret_suffix)")
        return
    end

    same_group = ch_α.species == ch_β.species

    if α == β
        # 对角: Wigner-Eckart, κ + a 对角, 仅 reduced ME 依赖 r
        println(io, "        # Same permutation group: κ-diagonal, a-diagonal")
        println(io, "        kapA == kapB && aA == aB || return $(ret_prefix)0.0$(ret_suffix)")
        println(io)
        _write_same_group_branches_body(io, subs_α, subs_β; ret_prefix=ret_prefix, ret_suffix=ret_suffix)
    elseif same_group
        # 不同道但同置换群 (如 ρρ ↔ ππ, 均为 species=[2]): 仍适用 WE
        println(io, "        # Same permutation group (different channels): κ-diagonal, a-diagonal")
        println(io, "        kapA == kapB && aA == aB || return $(ret_prefix)0.0$(ret_suffix)")
        println(io)
        _write_same_group_branches_body(io, subs_α, subs_β; ret_prefix=ret_prefix, ret_suffix=ret_suffix)
    else
        # 不同置换群: 显式所有 (κ, r, a) 分支
        println(io, "        # Different permutation groups: no Wigner-Eckart — explicit branches")
        _write_diff_group_branches_body(io, subs_α, subs_β, ch_α.name, ch_β.name;
                                        ret_prefix=ret_prefix, ret_suffix=ret_suffix)
    end
end

function _write_hermitian_conj_body(io, α, β)
    println(io, "        # WARNING: Hermitian conjugate — you must implement")
    println(io, "        #   return conj(my_V(nB, nA, s, sp, kapB, kapA, rB, rA, aB, aA, $β, $α, L_phys, params))")
    println(io, "        # or write the explicit matrix element.")
    println(io, "        return conj(my_V(nB, nA, s, sp, kapB, kapA, rB, rA, aB, aA, $β, $α, L_phys, params))")
end

# ===== 同群分支 =====

function _write_same_group_branches_body(io, subs_α, subs_β; ret_prefix="", ret_suffix="")
    α_uniq = unique(s -> (s.κ, s.r), subs_α)
    β_uniq = unique(s -> (s.κ, s.r), subs_β)
    need_κ = _need_branch_value(subs_α, :κ)
    need_r = _need_branch_value(subs_α, :r) || _need_branch_value(subs_β, :r)

    if !need_κ && !need_r
        s_α = α_uniq[1]
        s_β = β_uniq[1]
        println(io, "        # <$(s_α.κ), r=$(s_α.r) || V || $(s_β.κ), r=$(s_β.r)>")
        println(io, "        error(\"TODO: fill reduced matrix element for κ=$(_escape_str(s_α.κ)), rA=$(s_α.r), rB=$(s_β.r)\")")
        return
    end

    first = true
    for s_α in α_uniq, s_β in β_uniq
        s_α.κ == s_β.κ || continue
        conds = String[]
        need_κ && push!(conds, "kapA == $(repr(s_α.κ))")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")
        keyword = first ? "if" : "elseif"
        println(io, "        $keyword $(join(conds, " && "))")
        println(io, "            # <$(s_α.κ), r=$(s_α.r) || V || $(s_β.κ), r=$(s_β.r)>")
        println(io, "            error(\"TODO: fill reduced matrix element for κ=$(_escape_str(s_α.κ)), rA=$(s_α.r), rB=$(s_β.r)\")")
        first = false
    end
    println(io, "        end")
    println(io, "        error(\"unreachable: no matching (κ,r) branch for κ=\$(repr(kapA)) rA=\$rA rB=\$rB\")")
end

# ===== 不同群分支 =====

function _write_diff_group_branches_body(io, subs_α, subs_β, name_α, name_β;
                                        ret_prefix="", ret_suffix="")
    need_κ = _need_branch_value(subs_α, :κ) || _need_branch_value(subs_β, :κ)
    need_r = _need_branch_value(subs_α, :r) || _need_branch_value(subs_β, :r)
    need_a = _need_branch_value(subs_α, :a) || _need_branch_value(subs_β, :a)
    any_branch = need_κ || need_r || need_a

    if !any_branch
        s_α = subs_α[1]
        s_β = subs_β[1]
        println(io, "        # <$(s_α.κ), r=$(s_α.r), a=$(s_α.a) | $(name_α)←$(name_β) | $(s_β.κ), r=$(s_β.r), a=$(s_β.a)>")
        println(io, "        error(\"TODO: fill matrix element for $(name_α)←$(name_β), κA=$(_escape_str(s_α.κ)), κB=$(_escape_str(s_β.κ)), rA=$(s_α.r), rB=$(s_β.r), aA=$(s_α.a), aB=$(s_β.a)\")")
        return
    end

    first = true
    for s_α in subs_α, s_β in subs_β
        conds = String[]
        need_κ && push!(conds, "kapA == $(repr(s_α.κ)) && kapB == $(repr(s_β.κ))")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")
        need_a && push!(conds, "aA == $(s_α.a) && aB == $(s_β.a)")
        keyword = first ? "if" : "elseif"
        println(io, "        $keyword $(join(conds, " && "))")
        println(io, "            # <$(s_α.κ), r=$(s_α.r), a=$(s_α.a) | $(name_α)←$(name_β) | $(s_β.κ), r=$(s_β.r), a=$(s_β.a)>")
        println(io, "            error(\"TODO: fill matrix element for $(name_α)←$(name_β), κA=$(_escape_str(s_α.κ)), κB=$(_escape_str(s_β.κ)), rA=$(s_α.r), rB=$(s_β.r), aA=$(s_α.a), aB=$(s_β.a)\")")
        first = false
    end
    println(io, "        end")
    println(io, "        error(\"unreachable: no matching branch for $(name_α)←$(name_β), kapA=\$(repr(kapA)) kapB=\$(repr(kapB)) rA=\$rA rB=\$rB aA=\$aA aB=\$aB\")")
end

# ============ V 模板辅助 ============

function _need_branch_value(subs, field::Symbol)
    vals = Set{Any}()
    for s in subs
        push!(vals, field == :κ ? s.κ : field == :r ? s.r : s.a)
    end
    return length(vals) > 1
end

# 将任意值转为可嵌入 Julia 字符串字面量的表示（转义 \" 和 \\）
function _escape_str(x)
    s = repr(x)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    return s
end

