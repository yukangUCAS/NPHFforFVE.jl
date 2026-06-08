module FockSpace

using ..NPHFforFVE: Momentum, D000, _to_momentum, isospin_decomposition, multi_isospin_decomposition, get_SN_irrep_dim
using ..NPHFforFVE: cache_channel_reps!
using ..NPHFforFVE: group_for_momentum
using ..NPHFforFVE: OH_IRREP_NAMES, OH2_IRREP_NAMES, LG_IRREP_NAMES

export FockChannel, FockSystem, setup_fock_system
export get_N, get_num_species, get_total_N
export get_isospin_subchannels
export KineticType, relativistic, nonrelativistic

# ============ 动能色散关系 ============

"""
    KineticType

枚举类型，指定动能色散关系。

- `relativistic`: E(p) = √(m² + p²)
- `nonrelativistic`: E(p) = m + p²/(2m)
"""
@enum KineticType begin
    relativistic
    nonrelativistic
end

# ============ FockChannel ============

"""
    FockChannel(name, species, particle_types, masses, spins, isospins)

单个 Fock 道。总同位旋 I 为全局量，存于 FockSystem。

# 参数
- `name::String`: 道名，如 \"ππN\"
- `species::Vector{Int}`: 各粒子种类粒子数
- `particle_types::Vector{Symbol}`: 各粒子种类粒子类型 (`:boson` / `:fermion`)
- `masses::Vector{Float64}`: 各粒子种类质量 (MeV)
- `spins::Vector{Rational{Int}}`: 各粒子种类自旋 j
- `isospins::Vector{Rational{Int}}`: 各粒子种类同位旋 j
"""
struct FockChannel
    name::String
    N::Int
    species::Vector{Int}
    particle_types::Vector{Symbol}
    masses::Vector{Float64}
    spins::Vector{Rational{Int}}
    isospins::Vector{Rational{Int}}
    etas::Vector{Float64}
    kinetic_type::KineticType

    function FockChannel(name::AbstractString, species::Vector{Int},
                         particle_types::Vector{Symbol}, masses,
                         spins, isospins, etas, kinetic_type::KineticType)
        n = length(species)
        name     = String(name)
        masses   = Float64.(masses)
        spins    = Rational{Int}.(spins)
        isospins = Rational{Int}.(isospins)
        etas     = Float64.(etas)
        all(x -> x > 0, species) || throw(ArgumentError("每种粒子种类的粒子数必须 > 0"))
        length(particle_types) == n || throw(ArgumentError("particle_types 长度必须等于 species 长度"))
        length(masses) == n        || throw(ArgumentError("masses 长度必须等于 species 长度"))
        length(spins) == n         || throw(ArgumentError("spins 长度必须等于 species 长度"))
        length(isospins) == n      || throw(ArgumentError("isospins 长度必须等于 species 长度"))
        length(etas) == n          || throw(ArgumentError("etas 长度必须等于 species 长度"))
        for (i, pt) in enumerate(particle_types)
            pt in (:boson, :fermion) || throw(ArgumentError(
                "Species $i: particle_type 必须是 :boson 或 :fermion，得到 :$pt"))
        end
        for (i, eta) in enumerate(etas)
            eta ∈ (-1.0, 1.0) || throw(ArgumentError(
                "Species $i: 内禀宇称 eta 必须是 +1 或 -1，得到 $eta"))
        end
        _validate("spin", spins)
        _validate("isospin", isospins)
        N = sum(species)
        new(name, N, species, particle_types, masses, spins, isospins, etas, kinetic_type)
    end
end

function _validate(label::String, vals::Vector{Rational{Int}})
    allowed = (0, 1//2, 1)
    for (i, v) in enumerate(vals)
        v in allowed || throw(ArgumentError(
            "Species $i: $label = $v 暂不支持，目前仅支持 0, 1/2, 1"))
    end
end

get_N(ch::FockChannel) = ch.N
get_num_species(ch::FockChannel) = length(ch.species)

# ============ 同位旋子道 ============

"""
    IsospinSubChannel

一个同位旋子道，由 (κ, r, a) 标识。

- `κ`: S_N 不可约表示标签。单物种: String (如 \"[2,1]\")；
       多物种: NTuple{K,String} (如 (\"[2]\", \"[1]\") 表示 S₂×S₁)
- `r::Int`: 多重度标号 (1 ≤ r ≤ multiplicity)
- `a::Int`: 表示列指标 (1 ≤ a ≤ dim(κ))，dim = ∏ dim(κₛ)
- `dim::Int`: dim(κ)，即不可约表示总维度
- `mult::Int`: 该 κ 的重数
"""
struct IsospinSubChannel
    κ
    r::Int
    a::Int
    dim::Int
    mult::Int
end

"""
    get_isospin_subchannels(ch::FockChannel, I::Rational{Int}) -> Vector{IsospinSubChannel}

返回该道在总同位旋 I 下的所有同位旋子道列表。
单粒子种类道调用 `isospin_decomposition`，多粒子种类道调用 `multi_isospin_decomposition`。
"""
function get_isospin_subchannels(ch::FockChannel, I::Rational{Int})
    if length(ch.species) != 1
        return _multi_species_subchannels(ch, I)
    end
    # 单物种：保持原有逻辑不变
    N_spec = ch.species[1]
    j_spec = ch.isospins[1]
    decomp = isospin_decomposition(N_spec, j_spec)
    sub_channels = IsospinSubChannel[]
    for (J, κ, mult) in decomp.entries
        J == I || continue
        dim_κ = get_SN_irrep_dim(N_spec, κ)
        for r in 1:mult
            for a in 1:dim_κ
                push!(sub_channels, IsospinSubChannel(κ, r, a, dim_κ, mult))
            end
        end
    end
    return sub_channels
end

function _multi_species_subchannels(ch::FockChannel, I::Rational{Int})
    decomp = multi_isospin_decomposition(ch.species, ch.isospins, I)
    sub_channels = IsospinSubChannel[]
    for entry in decomp.entries
        dim_κ = 1
        for (s, κ_s) in enumerate(entry.κ_tuple)
            dim_κ *= get_SN_irrep_dim(ch.species[s], κ_s)
        end
        total_mult = entry.coupling_mult * entry.internal_mult
        for r in 1:total_mult
            for a in 1:dim_κ
                push!(sub_channels, IsospinSubChannel(entry.κ_tuple, r, a, dim_κ, total_mult))
            end
        end
    end
    return sub_channels
end

# ============ FockSystem ============

"""
    FockSystem(d, Ncut, channels, L, a, I; Ncut_channel=nothing)

多道 Fock 体系。

# 参数
- `d::Momentum`: 总动量（全局）
- `Ncut::Int`: 全局动量截断；若 `Ncut_channel` 非空则被覆盖
- `channels::Vector{FockChannel}`: 各 Fock 道
- `L::Int`: 有限体积尺寸（格点单位，正整数）
- `a::Float64`: 格距（fm）
- `I::Rational{Int}`: 总同位旋（全局守恒量）
- `Ncut_channel`: 可选，每道单独的 Ncut 向量，长度等于通道数
"""
struct FockSystem
    d::Momentum
    Ncut::Int
    Ncut_channel::Union{Nothing, Vector{Int}}
    channels::Vector{FockChannel}
    L::Int
    a::Float64
    I::Rational{Int}
    selected_irreps::Vector{String}

    function FockSystem(d, Ncut::Int, channels::Vector{FockChannel},
                        L::Int, a::Real, I, selected_irreps::Vector{String};
                        Ncut_channel::Union{Nothing, Vector{Int}}=nothing)
        Ncut >= 1 || throw(ArgumentError("Ncut 必须 ≥ 1"))
        L > 0 || throw(ArgumentError("L 必须为正整数"))
        a > 0 || throw(ArgumentError("格距 a 必须 > 0"))
        length(selected_irreps) > 0 || throw(ArgumentError("必须至少选择一个不可约表示"))
        for irr in selected_irreps
            all(x -> isascii(x), irr) || throw(ArgumentError("无效的不可约表示名: $irr"))
        end
        I_r = Rational{Int}(I)
        if Ncut_channel !== nothing
            length(Ncut_channel) == length(channels) || throw(ArgumentError(
                "Ncut_channel 长度必须等于 channels 长度"))
            all(x -> x >= 1, Ncut_channel) || throw(ArgumentError("各道 Ncut 必须 ≥ 1"))
        end
        d_vec = _to_momentum(d)
        new(d_vec, Ncut, Ncut_channel, channels, L, Float64(a), I_r, selected_irreps)
    end
end

get_total_N(sys::FockSystem) = sum(ch.N for ch in sys.channels)

"""
    get_Ncut(sys::FockSystem, ch_idx::Int)

返回第 `ch_idx` 道的有效 Ncut。
"""
function get_Ncut(sys::FockSystem, ch_idx::Int)
    if sys.Ncut_channel !== nothing
        return sys.Ncut_channel[ch_idx]
    else
        return sys.Ncut
    end
end

# ============ 交互式构建 ============

"""
    setup_fock_system()

REPL 交互式构建 `FockSystem`。逐步引导用户输入总动量、截断和各道信息。
"""
function setup_fock_system()
    println("="^50)
    println("Fock Space 构建")
    println("="^50)

    # 总动量
    print("\n总动量 (格式: nx ny nz, 默认 0 0 0): ")
    d_input = strip(readline())
    if isempty(d_input)
        d = D000
    else
        parts = parse.(Int, split(d_input))
        length(parts) == 3 || throw(ArgumentError("总动量必须是 3 个整数"))
        d = Momentum(parts...)
    end
    println("  总动量 d = $d")

    # Ncut 策略
    print("\n使用全局 Ncut? (y/n, 默认 y): ")
    use_global = strip(readline())
    use_global = isempty(use_global) || lowercase(use_global)[1] == 'y'

    Ncut_global = 0
    if use_global
        print("全局 Ncut (格点单位，|n|² 截断): ")
        Ncut_global = parse(Int, readline())
        Ncut_global >= 1 || throw(ArgumentError("Ncut 必须 ≥ 1"))
    end

    # 有限体积参数
    print("\n有限体积尺寸 L (格点单位, 正整数): ")
    L = parse(Int, readline())
    L > 0 || throw(ArgumentError("L 必须为正整数"))

    print("格距 a (fm): ")
    a = parse(Float64, readline())
    a > 0 || throw(ArgumentError("格距 a 必须 > 0"))

    # 总同位旋（全局守恒量）
    print("\n总同位旋 I (如 0, 1/2, 1, 3/2, 2): ")
    I = _parse_rational(readline())

    # 道数
    print("\nFock 道数: ")
    num_fock = parse(Int, readline())
    num_fock >= 1 || throw(ArgumentError("至少需要 1 个 Fock 道"))

    channels = FockChannel[]
    Ncut_channel = use_global ? nothing : Int[]

    for i in 1:num_fock
        println("\n--- 道 $i ---")
        print("  道名 (需键入引号，如 \"ππN\"): ")
        name_input = strip(readline())
        name = replace(name_input, '"' => "")
        isempty(name) && throw(ArgumentError("道名不能为空"))

        print("  粒子种类数: ")
        n_species = parse(Int, readline())
        n_species >= 1 || throw(ArgumentError("粒子种类数 ≥ 1"))

        species     = Int[]
        ptypes      = Symbol[]
        masses      = Float64[]
        spins       = Rational{Int}[]
        isospins    = Rational{Int}[]
        etas        = Float64[]

        for s in 1:n_species
            println("  粒子种类 $s:")
            print("    粒子数: ")
            push!(species, parse(Int, readline()))
            print("    质量 (MeV): ")
            push!(masses, parse(Float64, readline()))
            print("    类型 (boson/fermion): ")
            pt = Symbol(lowercase(strip(readline())))
            pt in (:boson, :fermion) || throw(ArgumentError("类型必须是 boson 或 fermion"))
            push!(ptypes, pt)
            print("    自旋 j (0, 1/2, 1): ")
            push!(spins, _parse_rational(readline()))
            print("    同位旋 j (0, 1/2, 1): ")
            push!(isospins, _parse_rational(readline()))
            print("    内禀宇称 (+1/-1): ")
            push!(etas, parse(Float64, readline()))
        end

        print("    动能色散关系 (relativistic/nonrelativistic, 默认 relativistic): ")
        kt_input = strip(readline())
        if isempty(kt_input) || lowercase(kt_input) == "relativistic"
            kt = relativistic
        elseif lowercase(kt_input) == "nonrelativistic"
            kt = nonrelativistic
        else
            throw(ArgumentError("色散关系必须是 relativistic 或 nonrelativistic"))
        end

        if !use_global
            print("  该道 Ncut (|n|² 截断): ")
            push!(Ncut_channel, parse(Int, readline()))
        end

        ch = FockChannel(name, species, ptypes, masses, spins, isospins, etas, kt)
        push!(channels, ch)

        # 显示同位旋子道信息
        subs = get_isospin_subchannels(ch, I)
        if !isempty(subs)
            println("  ✓ 道 \"$name\" (N=$(ch.N)) → " *
                    "$(length(unique(s->(s.κ, s.mult), subs))) 个子道")
            for s_ch in unique(s -> (s.κ, s.mult), subs)
                dim_str = _kappa_dim_display(ch, s_ch.κ)
                println("      κ=$(s_ch.κ)  r=1:$(s_ch.mult)  dim=$dim_str")
            end
        else
            println("  ✓ 道 \"$name\" (N=$(ch.N)) 已添加")
        end
    end

    # 确定对称群（考虑费米子 → 双覆盖）
    has_fermion = any(ch -> any(pt -> pt == :fermion, ch.particle_types), channels)
    _, group_name = group_for_momentum(d; double_cover=has_fermion)
    available_irreps = _get_irrep_list(group_name)
    println("\n对称群: $group_name (has_fermion=$has_fermion)")
    println("可选不可约表示:")
    for (i, irr) in enumerate(available_irreps)
        println("  [$i] $irr")
    end
    print("选择需要的不可约表示 (空格分隔序号): ")
    irr_input = strip(readline())
    if isempty(irr_input)
        error("必须至少选择一个不可约表示")
    end
    indices = parse.(Int, split(irr_input))
    selected_irreps = [available_irreps[i] for i in indices]
    println("  已选: $selected_irreps")

    sys = FockSystem(d, Ncut_global, channels, L, a, I, selected_irreps;
                     Ncut_channel=Ncut_channel)

    println("\n" * "="^50)
    println("Fock 体系构建完成")
    _print_system(sys)

    # 预填充代表动量/螺旋度缓存
    println("\n预计算代表动量与代表螺旋度...")
    for (i, ch) in enumerate(channels)
        ncut_i = get_Ncut(sys, i)
        reps = cache_channel_reps!(ch.species, ch.particle_types, ncut_i, sys.d, ch.spins)
        n_hel = sum(length(get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, sys.d))
                    for rep in reps)
        println("  道 $i \"$(ch.name)\": $(length(reps)) 代表动量, $n_hel 代表螺旋度")
    end

    # 生成 V 函数模板
    template_path = _generate_potential_template(sys)
    println("\n请在 $(template_path) 中填写各道对之间的 V 函数。")

    return sys
end

function _parse_rational(s::AbstractString)
    s = strip(s)
    if '/' in s
        parts = split(s, '/')
        length(parts) == 2 || throw(ArgumentError("无法解析有理数: $s"))
        return parse(Int, parts[1]) // parse(Int, parts[2])
    else
        return parse(Int, s) // 1
    end
end

function _get_irrep_list(group_name::Symbol)
    if group_name == :Oh
        return copy(OH_IRREP_NAMES)
    elseif group_name == :Oh2
        return copy(OH2_IRREP_NAMES)
    elseif haskey(LG_IRREP_NAMES, group_name)
        return copy(LG_IRREP_NAMES[group_name])
    else
        error("未知对称群: $group_name")
    end
end

function _kappa_dim(ch::FockChannel, κ)
    if κ isa String
        return get_SN_irrep_dim(ch.N, κ)
    end
    # multi-species: tuple of S_N irreps
    dim = 1
    for (sidx, κ_s) in enumerate(κ)
        dim *= get_SN_irrep_dim(ch.species[sidx], κ_s)
    end
    return dim
end

_kappa_dim_display(ch::FockChannel, κ) = string(_kappa_dim(ch, κ))

function _print_system(sys::FockSystem)
    println("  总动量 d = $(sys.d)")
    println("  L = $(sys.L), a = $(sys.a) fm")
    println("  总同位旋 I = $(sys.I)")
    println("  选定的不可约表示: $(sys.selected_irreps)")
    nc = sys.Ncut_channel
    for (i, ch) in enumerate(sys.channels)
        ncut_i = nc !== nothing ? nc[i] : sys.Ncut
        subs = get_isospin_subchannels(ch, sys.I)
        n_sub = length(unique(s -> (s.κ, s.mult), subs))
        println("  道 $i: $(ch.name)  N=$(ch.N)  Ncut=$ncut_i  ($n_sub 个同位旋子道)  $(ch.kinetic_type)")
        for s in 1:length(ch.species)
            println("    粒子种类$s: $(ch.species[s]) × ($(ch.particle_types[s]), " *
                    "j=$(ch.spins[s]), I=$(ch.isospins[s]), η=$(ch.etas[s]), m=$(ch.masses[s]) MeV)")
        end
    end
end

# ============ 生成 V 函数模板 ============

function _generate_potential_template(sys::FockSystem)
    path = "potential_defs.jl"
    io = open(path, "w")

    println(io, "# ============================================================")
    println(io, "# Interaction potential V -- user-defined matrix elements")
    println(io, "# ============================================================")
    println(io, "#")
    println(io, "# Usage:")
    println(io, "#   1. Edit the MyParams struct below with your LEC / cutoff params.")
    println(io, "#      Default values are used when you call MyParams().")
    println(io, "#")
    println(io, "#       p0 = MyParams()                       # all defaults")
    println(io, "#       p0 = MyParams(; C0=2.0)               # override C0")
    println(io, "#       x  = to_vector(p0)                    # -> Vector{Float64}")
    println(io, "#       names = param_names(MyParams)         # -> [\"C0\", \"C1\", ...]")
    println(io, "#       p_best = from_vector(MyParams, x_fit) # after IMINUIT")
    println(io, "#")
    println(io, "#   2. Fill in the my_V_αβ functions below.")
    println(io, "#      They receive `params::MyParams` as the last argument.")
    println(io, "#")
    println(io, "#   3. Define `param_bounds(::Type{MyParams})` at the bottom")
    println(io, "#      of this file with your IMINUIT fit bounds.")
    println(io, "#")
    println(io, "# System summary:")
    println(io, "#   total d = $(sys.d)")
    println(io, "#   total I = $(sys.I)")
    println(io, "#   Ncut     = $(sys.Ncut)")
    println(io, "#   L        = $(sys.L)")
    println(io, "#   a        = $(sys.a) fm")
    println(io, "#   channels = $(length(sys.channels))")
    for (i, ch) in enumerate(sys.channels)
        println(io, "#     $i: \"$(ch.name)\" N=$(ch.N) species=$(ch.species)")
    end
    println(io, "# ============================================================")
    println(io)
    println(io, "using NPHFforFVE")
    println(io)
    println(io, "# ============ Parameter struct ============")
    println(io, "# Add your LECs, cutoffs etc. below. Default values are used")
    println(io, "# when no keyword argument is given.")
    println(io, "@params struct MyParams")
    println(io, "    # Examples (uncomment and edit):")
    println(io, "    # C0  = 1.0    # leading-order contact")
    println(io, "    # C1  = 0.5    # NLO contact")
    println(io, "    # Λ   = 4.0    # cutoff (1/fm)")
    println(io, "end")
    println(io)
    println(io, "# ============ V matrix elements ============")
    println(io, "#")
    println(io, "# Each function receives integer momentum vectors nA, nB.")
    println(io, "# Physical momenta pAi, pBi = (2π/L) * n are auto-generated.")
    println(io, "# For delta-function conversion:  δ^3(p'-p) → (L/(2π))^3 · δ_{n',n}")
    println(io, "#   use: if nA[i] == nB[j]  for Kronecker delta")
    println(io, "# L = $(sys.L),  (L/(2π))^3 = $((sys.L/(2π))^3)")
    println(io)

    I = sys.I
    n_ch = length(sys.channels)
    for α in 1:n_ch
        ch_α = sys.channels[α]
        subs_α = get_isospin_subchannels(ch_α, I)
        N_α = ch_α.N

        for β in 1:n_ch
            ch_β = sys.channels[β]
            subs_β = get_isospin_subchannels(ch_β, I)
            N_β = ch_β.N

            multi_α = length(ch_α.species) > 1
            multi_β = length(ch_β.species) > 1
            has_κ = !isempty(subs_α) || !isempty(subs_β)
            if has_κ
                if multi_α || multi_β
                    push!(sig_keys, "kapA", "kapB")
                else
                    push!(sig_keys, "kapA::String", "kapB::String")
                end
                push!(sig_keys, "rA::Int", "rB::Int")
                push!(sig_keys, "aA::Int", "aB::Int")
            end

            sig_str = join(["nA::NTuple{$N_α,Momentum}",
                            "nB::NTuple{$N_β,Momentum}",
                            "sp::NTuple{$N_α,Rational{Int}}",
                            "s::NTuple{$N_β,Rational{Int}}",
                            sig_keys...,
                            "params::MyParams"], ", ")

            println(io, "# Channel $α \"$(ch_α.name)\" <- Channel $β \"$(ch_β.name)\"")
            println(io, "function my_V_$(α)$(β)($sig_str)")
            _gen_physical_momenta(io, sys.L, N_α, "A")
            _gen_physical_momenta(io, sys.L, N_β, "B")

            if multi_α || multi_β
                println(io, "    # TODO: multi-species channel -- auto isospin sub-channel branches not yet supported")
                println(io, "    return 0.0")
            else
                same_group = _same_sn_group(ch_α, ch_β)
                if same_group
                    _gen_same_group_branches(io, subs_α, subs_β)
                else
                    _gen_diff_group_branches(io, subs_α, subs_β)
                end
            end
            println(io, "end")
            println(io)
        end
    end

    println(io, "# ============ IMINUIT bounds ============")
    println(io, "# Override param_bounds for MyParams. Each tuple is (lower, upper).")
    println(io, "# function param_bounds(::Type{MyParams})")
    println(io, "#     return [(0.0, Inf), (0.0, Inf), (0.0, Inf)]")
    println(io, "# end")

    close(io)
    return path
end

# ============ 分支生成辅助 ============

function _gen_physical_momenta(io, L::Int, N::Int, label::String)
    pis = join(["p$(label)$i = (2π/$L) .* n$label[$i]" for i in 1:N], "; ")
    println(io, "    $pis")
end

_same_sn_group(ch_α::FockChannel, ch_β::FockChannel) = ch_α.species == ch_β.species

function _need_branch(subs::Vector{IsospinSubChannel}, field::Symbol)
    vals = Set{Any}()
    for s in subs
        push!(vals, field == :κ ? s.κ : field == :r ? s.r : s.a)
    end
    return length(vals) > 1
end

function _gen_same_group_branches(io, subs_α, subs_β)
    α_uniq = unique(s -> (s.κ, s.r), subs_α)
    β_uniq = unique(s -> (s.κ, s.r), subs_β)

    need_κ = _need_branch(subs_α, :κ)
    need_r = _need_branch(subs_α, :r) || _need_branch(subs_β, :r)

    println(io, "    # Same permutation group: kappa-diagonal + a-diagonal, only reduced matrix elements depend on r")
    println(io, "    kapA == kapB && aA == aB || return 0.0")
    println(io)

    if !need_κ && !need_r
        s_α = α_uniq[1]
        println(io, "    # <$(s_α.κ),r=$(s_α.r)||V||$(s_α.κ),r=$(s_α.r)>")
        println(io, "    return 0.0")
        return
    end

    first = true
    for s_α in α_uniq, s_β in β_uniq
        s_α.κ == s_β.κ || continue

        conds = String[]
        need_κ && push!(conds, "kapA == \"$(s_α.κ)\"")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")

        comment = "<$(s_α.κ),r=$(s_α.r)||V||$(s_β.κ),r=$(s_β.r)>"

        keyword = first ? "if" : "elseif"
        println(io, "    $(keyword) $(join(conds, " && "))")
        println(io, "        # $comment")
        println(io, "        return 0.0")
        first = false
    end
    println(io, "    end")
    println(io, "    return 0.0  # fallback")
end

function _gen_diff_group_branches(io, subs_α, subs_β)
    need_κ = _need_branch(subs_α, :κ) || _need_branch(subs_β, :κ)
    need_r = _need_branch(subs_α, :r) || _need_branch(subs_β, :r)
    need_a = _need_branch(subs_α, :a) || _need_branch(subs_β, :a)
    any_branch = need_κ || need_r || need_a

    if !any_branch
        s_α = subs_α[1]
        s_β = subs_β[1]
        println(io, "    # <$(s_α.κ),r=$(s_α.r),a=$(s_α.a)| <- |$(s_β.κ),r=$(s_β.r),a=$(s_β.a)>")
        println(io, "    return 0.0")
        return
    end

    first = true
    for s_α in subs_α, s_β in subs_β
        conds = String[]
        need_κ && push!(conds, "kapA == \"$(s_α.κ)\" && kapB == \"$(s_β.κ)\"")
        need_r && push!(conds, "rA == $(s_α.r) && rB == $(s_β.r)")
        need_a && push!(conds, "aA == $(s_α.a) && aB == $(s_β.a)")

        keyword = first ? "if" : "elseif"
        println(io, "    $(keyword) $(join(conds, " && "))")
        println(io, "        # <$(s_α.κ),r=$(s_α.r),a=$(s_α.a)| <- |$(s_β.κ),r=$(s_β.r),a=$(s_β.a)>")
        println(io, "        return 0.0")
        first = false
    end
    println(io, "    end")
    println(io, "    return 0.0  # fallback")
end

end # module FockSpace
