# ============================================================
# Hamiltonian — 投影哈密顿量编排
# ============================================================
# 三级分块:
#   Level 1: (rep_α,lam_α) × (rep_β,lam_β) → project_V 原子块
#   Level 2: 子道对 (κ_α,r_α, κ_β,r_β, Γ) → Level 1 块拼接
#   Level 3: 道对 (α,β) → Level 2 块拼接 → 最终大矩阵
# ============================================================

const _PROJ_CACHE = Dict{NamedTuple, @NamedTuple{
    X::Matrix{ComplexF64}, states::Vector, Z::Vector{Float64}}}()
const _PROJ_LIST_CACHE = Dict{NamedTuple, Vector}()

# ============ 优化管道: 跨不可约表示共享的几何缓存 ============

"""
    ProjBlockData

单个 (rep, lam) 块在一个不可约表示 Γ 下的投影数据。
X 矩阵大小为 (n_states × n_r)，row_indices 是这些 states 在统一基中的行号。
"""
struct ProjBlockData
    X::Matrix{ComplexF64}
    n_r::Int
    row_indices::Vector{Int}
end

"""
    SubBasisData

一个 (channel, subchannel) 的统一基数据:
- states / state_to_idx: 该子道在所有不可约表示中出现的全部 (n_tuple, lambda_tuple) 态的并集
- per_spin, per_mass, kinetic_type: 每粒子自旋/质量/色散关系
- rot_coeffs: 预计算的旋转系数 (每个态一个向量)
- proj_by_irrep: 每个不可约表示 → ProjBlockData 列表
"""
struct SubBasisData
    kappa
    r_val::Int
    a_val::Int       # vestigial, always 1 since sub-channels no longer split by irrep column
    dim_kappa::Int
    states::Vector
    state_to_idx::Dict
    per_spin::Vector{Rational{Int}}
    per_mass::Vector{Float64}
    kinetic_type::KineticType
    rot_coeffs::Vector{Vector{ComplexF64}}
    proj_by_irrep::Dict{String, Vector{ProjBlockData}}
end

"""
    SystemBasis <: Any

FockSystem 的几何缓存。包含所有不可约表示共享的态列表、旋转系数、
以及每个不可约表示的投影矩阵。与 V_func/params 无关，只需在 (L,a,Ncut) 变化时重建。

构建后调用 build_V_hel_blocks! 填入相互作用矩阵，然后用 project_and_diag 逐不可约表示求本征值。
"""
mutable struct SystemBasis
    sys::FockSystem
    L_phys::Float64
    chan_sub_data::Vector{Vector{SubBasisData}}
    irrep_total_dim::Dict{String, Int}
    # V_hel 缓存: (α, sα, β, sβ) → Matrix{ComplexF64}
    V_hel_blocks::Dict{Tuple{Int,Int,Int,Int}, Matrix{ComplexF64}}
end

"""
    SystemBasis(sys::FockSystem) -> SystemBasis

从 FockSystem 构建几何缓存。收集各道各子道在所有不可约表示下的统一态基，
预计算旋转系数和各不可约表示的投影矩阵。
"""
function SystemBasis(sys::FockSystem)
    L_phys = Float64(sys.L) * sys.a
    n_ch = length(sys.channels)

    # 运动系暂仅支持 N ≤ 2
    if sys.d != D000
        for ch in sys.channels
            ch.N > 2 && throw(ArgumentError(
                "运动系 (d≠0) 暂仅支持 N≤2 的道，道 \"$(ch.name)\" N=$(ch.N)"))
        end
    end

    chan_sub_data = Vector{Vector{SubBasisData}}(undef, n_ch)
    irrep_total_dim = Dict{String, Int}(Gamma => 0 for Gamma in sys.selected_irreps)

    for α in 1:n_ch
        ch = sys.channels[α]
        ncut = get_Ncut(sys, α)
        per_spin_raw, per_etas_arr = _expand_per_particle(ch)
        per_spin_v = Rational{Int}.(per_spin_raw)
        etas_v = Float64.(per_etas_arr)
        multi = length(ch.species) > 1
        per_mass = _expand_per_particle_mass(ch)

        subs = get_isospin_subchannels(ch, sys.I)
        sub_list = Vector{SubBasisData}(undef, length(subs))

        for (si, sub) in enumerate(subs)
            # 1) 收集所有不可约表示的所有态 (并集)
            ST = Tuple{NTuple{ch.N, Momentum}, NTuple{ch.N, Float64}}
            all_states = Vector{ST}()
            state_to_idx = Dict{ST, Int}()
            for Gamma in sys.selected_irreps
                projs = if multi
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
                else
                    spin_val = Float64(only(unique(per_spin_v)))
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
                end
                for p in projs
                    for st in p.states
                        if !haskey(state_to_idx, st)
                            push!(all_states, st)
                            state_to_idx[st] = length(all_states)
                        end
                    end
                end
            end

            # 2) 预计算旋转系数
            rot_coeffs = [get_rotation_vector(n, lam, per_spin_v) for (n, lam) in all_states]

            # 3) 为每个不可约表示收集投影块
            proj_by_irrep = Dict{String, Vector{ProjBlockData}}()
            for Gamma in sys.selected_irreps
                projs = if multi
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
                else
                    spin_val = Float64(only(unique(per_spin_v)))
                    _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
                end
                blocks = ProjBlockData[]
                dim_κ = sub.dim
                for p in projs
                    spatial_idx = [state_to_idx[st] for st in p.states]
                    row_idx = Int[(i - 1) * dim_κ + a for i in spatial_idx for a in 1:dim_κ]
                    push!(blocks, ProjBlockData(p.X, p.n_r, row_idx))
                end
                irrep_total_dim[Gamma] += sum(b.n_r for b in blocks; init=0)
                proj_by_irrep[Gamma] = blocks
            end

            sub_list[si] = SubBasisData(sub.κ, sub.r, sub.a, sub.dim,
                                        all_states, state_to_idx, per_spin_v,
                                        per_mass, ch.kinetic_type, rot_coeffs, proj_by_irrep)
        end

        chan_sub_data[α] = sub_list
    end

    return SystemBasis(sys, L_phys, chan_sub_data, irrep_total_dim,
                       Dict{Tuple{Int,Int,Int,Int}, Matrix{ComplexF64}}())
end

"""
    build_V_hel_blocks!(basis::SystemBasis, V_func, params) -> SystemBasis

为 SystemBasis 中所有 (sub_α, sub_β) 道对构建完整的 V_hel 矩阵（在统一态基中），
存入 basis.V_hel_blocks。所有不可约表示共享这些矩阵，后续只需做投影。

仅需在 V_func 或 params 变化时重新调用。basis（几何缓存）不变时保留复用。
"""
function build_V_hel_blocks!(basis::SystemBasis, V_func::Function, params;
                             V_basis::Symbol=:canonical,
                             V_filter=nothing)
    empty!(basis.V_hel_blocks)
    n_ch = length(basis.sys.channels)

    # 收集所有 (子道对) 工作项
    tasks = Tuple{Int,Int,SubBasisData,Int,Int,SubBasisData}[]
    for α in 1:n_ch
        ch_α = basis.sys.channels[α]
        sub_data_α = basis.chan_sub_data[α]
        for (sα, sd_α) in enumerate(sub_data_α)
            length(sd_α.states) == 0 && continue
            for β in 1:n_ch
                ch_β = basis.sys.channels[β]
                sub_data_β = basis.chan_sub_data[β]
                for (sβ, sd_β) in enumerate(sub_data_β)
                    length(sd_β.states) == 0 && continue
                    push!(tasks, (α, sα, sd_α, β, sβ, sd_β))
                end
            end
        end
    end

    # 串行调度各道对，并行在 build_V_hel / build_V_hel_direct 内部
    n_tasks = length(tasks)
    results = Vector{Matrix{ComplexF64}}(undef, n_tasks)
    keys    = Vector{Tuple{Int,Int,Int,Int}}(undef, n_tasks)

    for i in 1:n_tasks
        α, sα, sd_α, β, sβ, sd_β = tasks[i]
        V_adapted = _V_adapter(V_func, sd_α.kappa, sd_β.kappa,
                                   sd_α.r_val, sd_β.r_val,
                                   sd_α.a_val, sd_β.a_val,
                                   sd_α.dim_kappa, sd_β.dim_kappa,
                                   α, β, basis.L_phys, params)
        filter_adapted = if V_filter !== nothing
            (n_α, lam_or_sig_α, n_β, lam_or_sig_β, extra...) ->
                V_filter(n_α, n_β, lam_or_sig_α, lam_or_sig_β,
                         sd_α.kappa, sd_β.kappa,
                         sd_α.r_val, sd_β.r_val,
                         sd_α.a_val, sd_β.a_val,
                         α, β, basis.L_phys, params)
        else
            nothing
        end
        # channel 级哨兵: nA=nothing 仅检查道对是否可跳过
        if filter_adapted !== nothing && !filter_adapted(nothing, nothing, nothing, nothing)
            results[i] = Matrix{ComplexF64}(undef, 0, 0)
            keys[i] = (α, sα, β, sβ)
            continue
        end
        results[i] = if V_basis == :helicity
            build_V_hel_direct(sd_α.states, sd_β.states, V_adapted;
                               V_filter=filter_adapted)
        else
            build_V_hel(sd_α.states, sd_β.states,
                        sd_α.per_spin, sd_β.per_spin, V_adapted;
                        V_filter=filter_adapted)
        end
        keys[i] = (α, sα, β, sβ)
    end

    # 串行填入字典（避免并发写 Dict）
    for i in 1:n_tasks
        basis.V_hel_blocks[keys[i]] = results[i]
    end

    return basis
end

# ============ 优化管道: irrep 投影 + 本征值 ============

"""
    _assemble_irrep_hamiltonian(basis::SystemBasis, Gamma::String) -> Matrix{ComplexF64}

从 SystemBasis 的 V_hel_blocks 和投影矩阵，组装指定不可约表示 Γ 的完整投影哈密顿量矩阵。
"""
function _assemble_irrep_hamiltonian(basis::SystemBasis, Gamma::String)
    dim = basis.irrep_total_dim[Gamma]
    dim == 0 && return zeros(ComplexF64, 0, 0)

    H = zeros(ComplexF64, dim, dim)
    n_ch = length(basis.sys.channels)
    L_phys = basis.L_phys

    row_start = 1
    for α in 1:n_ch
        sub_data_α = basis.chan_sub_data[α]

        for (sα, sd_α) in enumerate(sub_data_α)
            blocks_α = sd_α.proj_by_irrep[Gamma]
            n_r_α = sum(b.n_r for b in blocks_α; init=0)
            n_r_α == 0 && continue

            col_start = 1
            for β in 1:n_ch
                sub_data_β = basis.chan_sub_data[β]

                for (sβ, sd_β) in enumerate(sub_data_β)
                    blocks_β = sd_β.proj_by_irrep[Gamma]
                    n_r_β = sum(b.n_r for b in blocks_β; init=0)

                    if n_r_β == 0
                        col_start += 0  # no contribution, advance nothing
                        continue
                    end
                    # n_r_β > 0: advance col_start after processing

                    V_hel = get(basis.V_hel_blocks, (α, sα, β, sβ), nothing)

                    if V_hel !== nothing && size(V_hel, 1) > 0 && size(V_hel, 2) > 0
                        # 有限体积因子
                        N_α_val = length(first(sd_α.states)[1])
                        N_β_val = length(first(sd_β.states)[1])
                        d_val = 3 * (N_α_val + N_β_val) - 6
                        fv = (2π * ħc / L_phys)^(d_val / 2)

                        # 投影 V_hel:  每个 (blk_α, blk_β) 块对
                        row_off = 1
                        for blk_α in blocks_α
                            col_off = 1
                            for blk_β in blocks_β
                                V_sub = V_hel[blk_α.row_indices, blk_β.row_indices]
                                V_proj = fv * blk_α.X' * V_sub * blk_β.X
                                r_rng = row_start + row_off - 1 : row_start + row_off + blk_α.n_r - 2
                                c_rng = col_start + col_off - 1 : col_start + col_off + blk_β.n_r - 2
                                H[r_rng, c_rng] .+= V_proj
                                col_off += blk_β.n_r
                            end
                            row_off += blk_α.n_r
                        end
                    end

                    # 动能对角项 (仅道对角 + 子道对角)
                    if α == β && sα == sβ
                        T_off = 1
                        for blk_α in blocks_α
                            n_tuple = sd_α.states[blk_α.row_indices[1]][1]
                            T_rep = _kinetic_energy_rep(n_tuple, sd_α.per_mass,
                                                        L_phys, sd_α.kinetic_type; d=basis.sys.d)
                            r_rng = row_start + T_off - 1 : row_start + T_off + blk_α.n_r - 2
                            for i in r_rng
                                H[i, i] += T_rep
                            end
                            T_off += blk_α.n_r
                        end
                    end

                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end

    return H
end

# ============ 稀疏线性算子 + eigs ============

"""
    _VBlock — 单个投影 V 块的存储

mat 为 fv * X_α' * V_sub * X_β，维度 n_r_α × n_r_β。
"""
struct _VBlock
    r_rng::UnitRange{Int}
    c_rng::UnitRange{Int}
    mat::Matrix{ComplexF64}
end

"""
    _build_hamiltonian_operator(basis, Gamma) -> (T_diag, v_blocks)

从 SystemBasis 构建哈密顿量线性算子表示。
返回动能对角向量 T_diag 和 V_proj 块列表，替代稠密 H 矩阵 (dim×dim)。
遍历逻辑与 _assemble_irrep_hamiltonian 完全一致，仅输出形式不同。
"""
function _build_hamiltonian_operator(basis::SystemBasis, Gamma::String)
    dim = basis.irrep_total_dim[Gamma]
    dim == 0 && return Float64[], _VBlock[]

    T_diag = zeros(Float64, dim)
    v_blocks = _VBlock[]
    n_ch = length(basis.sys.channels)
    L_phys = basis.L_phys

    row_start = 1
    for α in 1:n_ch
        sub_data_α = basis.chan_sub_data[α]
        for (sα, sd_α) in enumerate(sub_data_α)
            blocks_α = sd_α.proj_by_irrep[Gamma]
            n_r_α = sum(b.n_r for b in blocks_α; init=0)
            n_r_α == 0 && continue

            col_start = 1
            for β in 1:n_ch
                sub_data_β = basis.chan_sub_data[β]
                for (sβ, sd_β) in enumerate(sub_data_β)
                    blocks_β = sd_β.proj_by_irrep[Gamma]
                    n_r_β = sum(b.n_r for b in blocks_β; init=0)
                    n_r_β == 0 && continue

                    V_hel = get(basis.V_hel_blocks, (α, sα, β, sβ), nothing)

                    if V_hel !== nothing && size(V_hel, 1) > 0 && size(V_hel, 2) > 0
                        N_α_val = length(first(sd_α.states)[1])
                        N_β_val = length(first(sd_β.states)[1])
                        d_val = 3 * (N_α_val + N_β_val) - 6
                        fv = (2π * ħc / L_phys)^(d_val / 2)

                        row_off = 1
                        for blk_α in blocks_α
                            col_off = 1
                            for blk_β in blocks_β
                                V_sub = V_hel[blk_α.row_indices, blk_β.row_indices]
                                V_proj = fv * blk_α.X' * V_sub * blk_β.X
                                if !all(iszero, V_proj)
                                    r_rng = row_start+row_off-1 : row_start+row_off+blk_α.n_r-2
                                    c_rng = col_start+col_off-1 : col_start+col_off+blk_β.n_r-2
                                    push!(v_blocks, _VBlock(r_rng, c_rng, V_proj))
                                end
                                col_off += blk_β.n_r
                            end
                            row_off += blk_α.n_r
                        end
                    end

                    if α == β && sα == sβ
                        T_off = 1
                        for blk_α in blocks_α
                            n_tuple = sd_α.states[blk_α.row_indices[1]][1]
                            T_rep = _kinetic_energy_rep(n_tuple, sd_α.per_mass,
                                                        L_phys, sd_α.kinetic_type; d=basis.sys.d)
                            r_rng = row_start+T_off-1 : row_start+T_off+blk_α.n_r-2
                            T_diag[r_rng] .+= T_rep
                            T_off += blk_α.n_r
                        end
                    end

                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end

    return T_diag, v_blocks
end

"""
    _H_matvec(x, T_diag, v_blocks) -> y

线性算子 y = H * x，无需存储稠密 H 矩阵。
遍历所有 V 块: y[r_rng] += V_proj * x[c_rng]，加上动能对角贡献。
"""
function _H_matvec(x::AbstractVector{<:ComplexF64}, T_diag::Vector{Float64},
                   v_blocks::Vector{_VBlock})
    y = T_diag .* x
    for vb in v_blocks
        y[vb.r_rng] .+= vb.mat * x[vb.c_rng]
    end
    return y
end

"""
    compute_spectrum_eigs(basis::SystemBasis; n_levels=20, tol=1e-8)
        -> Dict{String, Vector{Float64}}

使用稀疏线性算子 + KrylovKit.eigsolve 计算各不可约表示的最低 n_levels 个本征值。

不构造稠密 dim×dim 哈密顿量矩阵，适用于大维度系统（>5000）。
"""
function compute_spectrum_eigs(basis::SystemBasis;
                                n_levels::Union{Int, Dict{String, Int}} = 20,
                                tol::Float64 = 1e-8)
    result = Dict{String, Vector{Float64}}()
    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue

        n_g = n_levels isa Int ? n_levels : get(n_levels, Gamma, 20)
        T_diag, v_blocks = _build_hamiltonian_operator(basis, Gamma)

        if isempty(v_blocks)
            result[Gamma] = sort(T_diag)[1:min(n_g, dim)]
            continue
        end

        n_ev = min(n_g, dim)
        H_op = x -> _H_matvec(x, T_diag, v_blocks)

        evals, _, info = eigsolve(
            H_op, randn(ComplexF64, dim), n_ev, :SR,
            Arnoldi(; tol=tol, maxiter=max(200, dim ÷ 5))
        )
        info.converged < n_ev &&
            @warn "KrylovKit 仅收敛 $(info.converged)/$n_ev 个本征值 (Γ=$Gamma)"

        ev = sort(real.(evals))[1:min(length(evals), n_ev)]

        if basis.sys.d != D000
            P_mag = (2π * ħc / basis.L_phys) * sqrt(Float64(sum(abs2, basis.sys.d)))
            ev = [sqrt(E^2 + P_mag^2) for E in ev]
        end
        result[Gamma] = ev
    end
    return result
end

"""
    compute_spectrum(basis::SystemBasis; n_levels = nothing) -> Dict{String, Vector{Float64}}

从已填充 V_hel 的 SystemBasis 计算各不可约表示的投影哈密顿量本征值（升序）。

若提供 `n_levels::Dict{String,Int}`，仅计算每个不可约表示最低的 n_levels[Γ] 个本征值。
"""
function compute_spectrum(basis::SystemBasis;
                          n_levels::Union{Nothing, Dict{String, Int}} = nothing)
    result = Dict{String, Vector{Float64}}()
    for Gamma in basis.sys.selected_irreps
        dim = basis.irrep_total_dim[Gamma]
        dim == 0 && continue

        H = _assemble_irrep_hamiltonian(basis, Gamma)
        n = if n_levels === nothing
            dim
        else
            min(get(n_levels, Gamma, dim), dim)
        end
        ev = if n >= dim
            sort(real.(eigvals(Hermitian(H))))
        else
            sort(real.(eigvals(Hermitian(H), 1:n)))
        end
        # 运动系: 质心系本征值 boost 到运动系
        if basis.sys.d != D000
            P_mag = (2π * ħc / basis.L_phys) * sqrt(Float64(sum(abs2, basis.sys.d)))
            ev = [sqrt(E^2 + P_mag^2) for E in ev]
        end
        result[Gamma] = ev
    end
    return result
end

# ============ 单次投影缓存 ============

function _get_projection(n_tuple::NTuple{N, Momentum},
                         lambda_tuple::NTuple{N, Float64},
                         kappa::String, Gamma::String,
                         d::Momentum, spin::Float64,
                         etas::Vector{Float64}) where N
    key = (n_tuple=n_tuple, lambda_tuple=lambda_tuple,
           kappa=kappa, Gamma=Gamma, d=d, spin=spin, etas=Tuple(etas))
    return get!(_PROJ_CACHE, key) do
        st = isinteger(spin) ? :boson : :fermion
        res = subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                                  d_total=d, species_type=st, spin=spin, etas=etas)
        (X=res.X, states=res.subspace_states, Z=res.Z)
    end
end

function _get_projection(n_tuple::NTuple{N, Momentum},
                         lambda_tuple::NTuple{N, Float64},
                         kappa::Tuple, Gamma::String,
                         d::Momentum,
                         species::Vector{Int}, particle_types::Vector{Symbol},
                         spins::Vector{Float64}, etas::Vector{Float64}) where N
    key = (n_tuple=n_tuple, lambda_tuple=lambda_tuple,
           kappa=kappa, Gamma=Gamma, d=d,
           species=Tuple(species), pt=Tuple(particle_types),
           spins=Tuple(spins), etas=Tuple(etas))
    return get!(_PROJ_CACHE, key) do
        res = subspace_projection(n_tuple, lambda_tuple, kappa, Gamma;
                                  d_total=d, species=species,
                                  particle_types=particle_types,
                                  spins=spins, etas=etas)
        (X=res.X, states=res.subspace_states, Z=res.Z)
    end
end

# ============ 工具函数 ============

function _expand_per_particle(ch::FockChannel)
    per_spin = Rational{Int}[]
    per_etas = Float64[]
    for (s, j, eta) in zip(ch.species, ch.spins, ch.etas)
        append!(per_spin, fill(j, s))
        append!(per_etas, fill(eta, s))
    end
    return per_spin, per_etas
end

# ============ 道的代表态投影列表 ============

function _get_channel_proj_list(ch::FockChannel, Ncut::Int, d::Momentum,
                                kappa::String, Gamma::String,
                                spin::Float64, etas::Vector{Float64})
    key = (species=Tuple(ch.species), pt=Tuple(ch.particle_types),
           Ncut=Ncut, d=d, kappa=kappa, Gamma=Gamma,
           spin=spin, etas=Tuple(etas))
    return get!(_PROJ_LIST_CACHE, key) do
        result = []
        reps = get_momentum_reps(ch.species, ch.particle_types, Ncut, d)
        for rep in reps
            M = count(n -> n == Momentum(0, 0, 0), rep)
            if M > 0 && spin != 0.0
                # 零动量 + 非零自旋: 对有限动量部分取螺旋度代表
                N_total = length(rep)
                N_fin = N_total - M
                if N_fin > 0
                    fin_rep = ntuple(i -> rep[M + i], N_fin)
                    fin_hels = get_helicity_reps(fin_rep, [N_fin],
                        [ch.particle_types[1]], [ch.spins[1]], d)
                else
                    fin_hels = [()]  # 全部粒子零动量
                end
                for fin_lam in fin_hels
                    fin_lam_f = Tuple(Float64.(fin_lam))
                    full_lam = (ntuple(_ -> 0.0, M)..., fin_lam_f...)
                    proj = _get_projection(rep, full_lam, kappa, Gamma, d, spin, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=full_lam, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            else
                hels = get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, d)
                for lam in hels
                    lam_f = Tuple(Float64.(lam))
                    proj = _get_projection(rep, lam_f, kappa, Gamma, d, spin, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=lam_f, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            end
        end
        result
    end
end

function _get_channel_proj_list(ch::FockChannel, Ncut::Int, d::Momentum,
                                kappa::Tuple, Gamma::String,
                                spins::Vector{Float64}, etas::Vector{Float64})
    key = (species=Tuple(ch.species), pt=Tuple(ch.particle_types),
           Ncut=Ncut, d=d, kappa=kappa, Gamma=Gamma,
           spins=Tuple(spins), etas=Tuple(etas))
    return get!(_PROJ_LIST_CACHE, key) do
        result = []
        reps = get_momentum_reps(ch.species, ch.particle_types, Ncut, d)
        for rep in reps
            hels = get_helicity_reps(rep, ch.species, ch.particle_types, ch.spins, d)
            if !isempty(hels)
                for lam in hels
                    lam_f = Tuple(Float64.(lam))
                    proj = _get_projection(rep, lam_f, kappa, Gamma, d,
                                           ch.species, ch.particle_types, spins, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=lam_f, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            else
                N_total = length(rep)
                M_total = count(n -> iszero(n), rep)
                N_fin = N_total - M_total

                fin_momenta = Momentum[]
                fin_species_vec = Int[]
                fin_types_vec = Symbol[]
                fin_spins_vec = Rational{Int}[]
                off = 0
                for (k, Nk) in enumerate(ch.species)
                    fm_in_sp = 0
                    for i in 1:Nk
                        !iszero(rep[off + i]) && (fm_in_sp += 1)
                    end
                    if fm_in_sp > 0
                        push!(fin_species_vec, fm_in_sp)
                        push!(fin_types_vec, ch.particle_types[k])
                        push!(fin_spins_vec, Rational{Int}(Int(2*ch.spins[k]), 2))
                        for i in 1:Nk
                            !iszero(rep[off + i]) && push!(fin_momenta, rep[off + i])
                        end
                    end
                    off += Nk
                end

                if N_fin > 0
                    fin_rep = Tuple(fin_momenta)
                    fin_hels = get_helicity_reps(fin_rep, fin_species_vec,
                        fin_types_vec, fin_spins_vec, d)
                else
                    fin_hels = [()]
                end

                for fin_lam in fin_hels
                    fin_lam_f = Float64[Float64(x) for x in fin_lam]
                    full_lam_vec = Float64[]
                    fm_idx = 1
                    off2 = 0
                    for (k, Nk) in enumerate(ch.species)
                        for i in 1:Nk
                            if iszero(rep[off2 + i])
                                push!(full_lam_vec, 0.0)
                            else
                                push!(full_lam_vec, fin_lam_f[fm_idx])
                                fm_idx += 1
                            end
                        end
                        off2 += Nk
                    end
                    full_lam = Tuple(full_lam_vec)

                    proj = _get_projection(rep, full_lam, kappa, Gamma, d,
                                           ch.species, ch.particle_types, spins, etas)
                    if length(proj.Z) > 0
                        push!(result, (rep=rep, lam=full_lam, X=proj.X,
                                       states=proj.states, Z=proj.Z, n_r=length(proj.Z)))
                    end
                end
            end
        end
        result
    end
end

# ============ 运动系工具 ============

"""
    boost_to_cm(p_mov, masses, d, L_phys) -> (p_cm, factor)

将运动系物理动量 boost 到质心系，并计算运动学因子。

# 参数
- `p_mov`: 运动系物理动量 (可迭代的 3-矢量，单位 MeV)
- `masses`: 每粒子质量 (MeV)，长度与 p_mov 一致
- `d`: 总动量整数矢量 d = (L/2πħc)·P
- `L_phys`: 物理盒子尺寸 (fm)

# 返回
- `p_cm`: 质心系动量 (与 p_mov 同类型)
- `factor`: 运动学因子 = [ΣE'/ΣE^cm · ∏(E^cm/E')]^{1/2}

当 d == (0,0,0) 时，p_cm = p_mov, factor = 1.0。
"""
function boost_to_cm(p_mov, masses::Vector{Float64}, d::Momentum, L_phys::Float64)
    N = length(p_mov)
    P_tot = (2π * ħc / L_phys) .* SVector{3,Float64}(d)
    P2 = sum(abs2, P_tot)

    # 静止系：恒等变换
    if P2 == 0.0
        return collect(p_mov), 1.0
    end

    # 运动系在壳能量 (boost 一律用相对论色散关系)
    E_prime = [sqrt(m^2 + sum(abs2, p)) for (p, m) in zip(p_mov, masses)]
    E_tot = sum(E_prime)

    # Lorentz boost
    gamma = E_tot / sqrt(E_tot^2 - P2)

    p_cm = similar(p_mov, eltype(p_mov))
    T = eltype(p_mov)
    for i in 1:N
        p = p_mov[i]
        p_dot_P = p[1]*P_tot[1] + p[2]*P_tot[2] + p[3]*P_tot[3]
        coeff = (gamma - 1.0) * p_dot_P / P2 - gamma * E_prime[i] / E_tot
        p_cm[i] = T(p[1] + coeff * P_tot[1],
                     p[2] + coeff * P_tot[2],
                     p[3] + coeff * P_tot[3])
    end

    # 质心系在壳能量
    E_cm = [sqrt(m^2 + sum(abs2, p)) for (p, m) in zip(p_cm, masses)]
    E_tot_cm = sum(E_cm)

    # 运动学因子
    prod_ratio = prod(E_cm[i] / E_prime[i] for i in 1:N)
    factor = sqrt(E_tot / E_tot_cm * prod_ratio)

    return p_cm, factor
end

# ============ 动能辅助函数 ============

function _expand_per_particle_mass(ch::FockChannel)
    per_mass = Float64[]
    for (s, m) in zip(ch.species, ch.masses)
        append!(per_mass, fill(m, s))
    end
    return per_mass
end

function _kinetic_energy_rep(n_tuple, per_mass::Vector{Float64}, L_phys::Float64, kt::KineticType;
                             d::Momentum = D000)
    pref = (2π * ħc / L_phys)^2
    T = 0.0
    # 运动系：先转换到质心系动量
    if d != D000
        pv = 2π * ħc / L_phys
        p_mov = [pv .* Float64.(n) for n in n_tuple]
        p_cm, _ = boost_to_cm(p_mov, per_mass, d, L_phys)
        for (p, m) in zip(p_cm, per_mass)
            p2 = Float64(sum(abs2, p))
            if kt == relativistic
                T += sqrt(m^2 + p2)
            else
                T += m + p2 / (2m)
            end
        end
    else
        for (n, m) in zip(n_tuple, per_mass)
            n2 = Float64(sum(abs2, n))
            if kt == relativistic
                T += sqrt(m^2 + pref * n2)
            else
                T += m + pref * n2 / (2m)
            end
        end
    end
    return T
end

function _build_kinetic_diag(projs::Vector, per_mass::Vector{Float64}, L_phys::Float64, kt::KineticType;
                             d::Momentum = D000)
    n_r = sum(p.n_r for p in projs; init=0)
    T = zeros(ComplexF64, n_r, n_r)
    n_r == 0 && return T
    row = 1
    for p in projs
        T_rep = _kinetic_energy_rep(p.rep, per_mass, L_phys, kt; d=d)
        for i in 1:p.n_r
            T[row, row] = T_rep
            row += 1
        end
    end
    return T
end

# ============ V 函数参数重排适配器 ============
# build_V_hel 调用: V_inner(n_α, σ_α, n_β, σ_β, extra_args...)
# 用户 V_func:      V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, params)
#
# 注: 修改 1 去除子道 a-循环后, aA/aB 恒为 1, V_func 直接返回 dimA×dimB 矩阵
#     (dim=1 时即标量), _V_adapter 统一透传, 不再做矩阵包装.

function _V_adapter(V_func, kapA, kapB, rA, rB, aA, aB, dimA, dimB, ch_α, ch_β, L_phys, params)
    return (n_α, σ_α, n_β, σ_β, extra...) ->
        V_func(n_α, n_β, σ_α, σ_β, kapA, kapB, rA, rB, aA, aB, ch_α, ch_β, L_phys, params)
end

# ============ Level 2: 子道对块 ============

function _build_subchannel_block(projs_α::Vector, projs_β::Vector,
                                 per_spin_α, per_spin_β,
                                 L_phys::Float64, V_adapter::Function;
                                 V_basis::Symbol=:canonical)
    n_r_α = sum(p.n_r for p in projs_α; init=0)
    n_r_β = sum(p.n_r for p in projs_β; init=0)
    (n_r_α == 0 || n_r_β == 0) && return zeros(ComplexF64, n_r_α, n_r_β)

    block = zeros(ComplexF64, n_r_α, n_r_β)
    row_start = 1
    for p_α in projs_α
        col_start = 1
        for p_β in projs_β
            sub = if V_basis == :helicity
                project_V_hel(p_α.X, p_β.X, p_α.states, p_β.states,
                             L_phys, V_adapter)
            else
                project_V(p_α.X, p_β.X, p_α.states, p_β.states,
                         per_spin_α, per_spin_β, L_phys, V_adapter)
            end
            block[row_start:row_start+p_α.n_r-1,
                  col_start:col_start+p_β.n_r-1] .= sub
            col_start += p_β.n_r
        end
        row_start += p_α.n_r
    end
    return block
end

# ============ Level 3: 主编排函数 ============

"""
    build_hamiltonian_block(sys::FockSystem, Gamma::String, V_func, params)
        -> Matrix{ComplexF64}

为指定的不可约表示 Γ 构造完整的投影哈密顿量矩阵。

`V_func` 签名同 potential_defs.jl:
    V_func(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, params)
"""
function build_hamiltonian_block(sys::FockSystem, Gamma::String,
                                 V_func::Function, params;
                                 V_basis::Symbol=:canonical)
    Gamma in sys.selected_irreps ||
        throw(ArgumentError("Γ=$Gamma 不在 sys.selected_irreps ($(sys.selected_irreps)) 中"))

    # 运动系暂仅支持 N ≤ 2
    if sys.d != D000
        for ch in sys.channels
            ch.N > 2 && throw(ArgumentError(
                "运动系 (d≠0) 暂仅支持 N≤2 的道，道 \"$(ch.name)\" N=$(ch.N)"))
        end
    end

    I = sys.I
    n_ch = length(sys.channels)
    d = sys.d

    # ===== 预处理: 各道各子道的投影列表 =====
    chan_sub_data = []
    for α in 1:n_ch
        ch = sys.channels[α]
        ncut = get_Ncut(sys, α)
        per_spin, per_etas_arr = _expand_per_particle(ch)
        per_spin_v = Rational{Int}.(per_spin)
        etas_v = Float64.(per_etas_arr)
        multi = length(ch.species) > 1
        per_mass = _expand_per_particle_mass(ch)

        subs = get_isospin_subchannels(ch, I)
        sub_entries = []
        for sub in subs
            projs = if multi
                _get_channel_proj_list(ch, ncut, d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
            else
                spin_val = Float64(only(unique(per_spin_v)))
                _get_channel_proj_list(ch, ncut, d, sub.κ, Gamma, spin_val, etas_v)
            end
            push!(sub_entries, (sub=sub, projs=projs, per_spin=per_spin_v))
        end
        push!(chan_sub_data, (ch=ch, subs=sub_entries, per_mass=per_mass))
    end

    # ===== 计算总维度 =====
    total_dim = 0
    for α in 1:n_ch
        for se in chan_sub_data[α].subs
            total_dim += sum(p.n_r for p in se.projs; init=0)
        end
    end
    total_dim == 0 && return zeros(ComplexF64, 0, 0)

    H_full = zeros(ComplexF64, total_dim, total_dim)

    # ===== 填充各块 =====
    row_start = 1
    for α in 1:n_ch
        ch_α = chan_sub_data[α].ch
        for se_α in chan_sub_data[α].subs
            s_α = se_α.sub
            projs_α = se_α.projs
            pspin_α = se_α.per_spin
            n_r_α = sum(p.n_r for p in projs_α; init=0)

            col_start = 1
            for β in 1:n_ch
                ch_β = chan_sub_data[β].ch
                for se_β in chan_sub_data[β].subs
                    s_β = se_β.sub
                    projs_β = se_β.projs
                    pspin_β = se_β.per_spin
                    n_r_β = sum(p.n_r for p in projs_β; init=0)

                    L_phys = Float64(sys.L) * sys.a

                    V_adapted = _V_adapter(V_func, s_α.κ, s_β.κ,
                                           s_α.r, s_β.r, s_α.a, s_β.a,
                                           s_α.dim, s_β.dim, α, β, L_phys, params)
                    sub_block = _build_subchannel_block(
                        projs_α, projs_β, pspin_α, pspin_β, L_phys, V_adapted;
                        V_basis=V_basis)

                    # 动能对角矩阵 (仅道对角 + 子道对角)
                    if α == β && s_α.κ == s_β.κ && s_α.r == s_β.r
                        T_diag = _build_kinetic_diag(projs_α, chan_sub_data[α].per_mass,
                                                     L_phys, ch_α.kinetic_type; d=d)
                        sub_block += T_diag
                    end

                    if n_r_α > 0 && n_r_β > 0
                        H_full[row_start:row_start+n_r_α-1,
                               col_start:col_start+n_r_β-1] .= sub_block
                    end
                    col_start += n_r_β
                end
            end
            row_start += n_r_α
        end
    end

    return H_full
end

# ============ 本征值计算 ============

"""
    compute_spectrum(sys::FockSystem, V_func, params) -> Dict{String, Vector{Float64}}

给定 FockSystem 和相互作用函数，返回各不可约表示的投影哈密顿量本征值（升序）。
键为不可约表示名，值为本征值向量 (MeV)。

# 示例
```julia
evals = compute_spectrum(sys, my_V_11, MyParams(C0=2.0))
evals["T1-"]  # T1- 能级列表
```
"""
function compute_spectrum(sys::FockSystem, V_func::Function, params;
                          n_levels::Union{Nothing, Dict{String, Int}} = nothing,
                          V_basis::Symbol=:canonical,
                          eigs::Bool=false,
                          V_filter=nothing)
    basis = SystemBasis(sys)
    build_V_hel_blocks!(basis, V_func, params; V_basis=V_basis,
                        V_filter=V_filter)
    if eigs
        n_use = something(n_levels, 20)
        return compute_spectrum_eigs(basis; n_levels=n_use)
    else
        return compute_spectrum(basis; n_levels=n_levels)
    end
end

"""
    compute_kinetic_spectrum(sys::FockSystem) -> Dict{String, Vector{Float64}}

仅动能（无相互作用）的本征值。
"""
function compute_kinetic_spectrum(sys::FockSystem)
    zero_V(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p) = zero(ComplexF64)
    return compute_spectrum(sys, zero_V, nothing)
end

# ============ 能谱输出 ============

"""
    write_energy_spectrum(sys::FockSystem, filename::String; V_func=nothing, params=nothing)

将各不可约表示的动能谱（及可选相互作用能谱）写入文本文件。

若提供 `V_func`，同时输出裸动能谱和相互作用后的能谱并列出偏移 ΔE。
"""
function write_energy_spectrum(sys::FockSystem, filename::String; V_func=nothing, params=nothing)
    io = open(filename, "w")

    println(io, "="^72)
    println(io, "NPHFforFVE 能谱")
    println(io, "="^72)
    println(io, "  总动量 d = $(sys.d)")
    println(io, "  总同位旋 I = $(sys.I)")
    println(io, "  L = $(sys.L), a = $(sys.a) fm  →  L_phys = $(Float64(sys.L) * sys.a) fm")
    println(io, "  道数: $(length(sys.channels))")
    for (i, ch) in enumerate(sys.channels)
        ncut_i = get_Ncut(sys, i)
        n_str = join(["$(s)×$(pt)(j=$(j),I=$(Ij),η=$η,m=$m)" for (s,pt,j,Ij,η,m) in
                       zip(ch.species, ch.particle_types, ch.spins, ch.isospins, ch.etas, ch.masses)], ", ")
        println(io, "    道 $i: \"$(ch.name)\"  N=$(ch.N)  Ncut=$ncut_i  ($n_str)")
    end
    has_V = V_func !== nothing
    println(io, "  相互作用: $(has_V ? "已提供" : "无（纯动能）")")
    println(io)

    zero_V(nA, nB, sp, s, kapA, kapB, rA, rB, aA, aB, chA, chB, L_phys, p) = zero(ComplexF64)

    for Gamma in sys.selected_irreps
        H_kin = build_hamiltonian_block(sys, Gamma, zero_V, nothing)
        dim = size(H_kin, 1)
        dim == 0 && continue

        ev_kin = sort(real.(eigvals(Hermitian(H_kin))))

        if has_V
            H_full = build_hamiltonian_block(sys, Gamma, V_func, params)
            ev_full = sort(real.(eigvals(Hermitian(H_full))))
        end

        # 运动系: 质心系本征值 boost 到运动系
        if sys.d != D000
            L_phys = Float64(sys.L) * sys.a
            P_mag = (2π * ħc / L_phys) * sqrt(Float64(sum(abs2, sys.d)))
            ev_kin = [sqrt(E^2 + P_mag^2) for E in ev_kin]
            if has_V
                ev_full = [sqrt(E^2 + P_mag^2) for E in ev_full]
            end
        end

        # 道维度分解
        ch_dims = Int[]
        for α in 1:length(sys.channels)
            ch_dims_α = _channel_dim_for_irrep(sys, Gamma, α)
            append!(ch_dims, ch_dims_α)
        end

        println(io, "─"^72)
        println(io, "不可约表示: $Gamma  (投影维数 = $dim)")
        if has_V
            println(io, rpad("  #", 5), rpad("E_kin (MeV)", 14), rpad("E_full (MeV)", 14), "ΔE (MeV)")
            println(io, "  " * "─"^50)
            for i in 1:dim
                Δ = ev_full[i] - ev_kin[i]
                mark = abs(Δ) > 0.01 ? (Δ > 0 ? "↑" : "↓") : " "
                println(io, rpad("  $i", 5),
                        rpad(lpad(round(ev_kin[i], digits=4), 10), 14),
                        rpad(lpad(round(ev_full[i], digits=4), 10), 14),
                        lpad(round(Δ, digits=4), 8), "  $mark")
            end
        else
            println(io, rpad("  #", 5), "E_kin (MeV)")
            println(io, "  " * "─"^25)
            for i in 1:dim
                println(io, rpad("  $i", 5), lpad(round(ev_kin[i], digits=4), 10))
            end
        end
        println(io)
    end

    println(io, "="^72)
    close(io)
    println("能谱已写入: $(abspath(filename))")
    return nothing
end

function _channel_dim_for_irrep(sys::FockSystem, Gamma::String, α::Int)
    ch = sys.channels[α]
    ncut = get_Ncut(sys, α)
    subs = get_isospin_subchannels(ch, sys.I)
    per_spin, per_etas = _expand_per_particle(ch)
    etas_v = Float64.(per_etas)
    multi = length(ch.species) > 1
    dims = Int[]
    for sub in subs
        projs = if multi
            _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, Float64.(ch.spins), etas_v)
        else
            spin_val = Float64(only(unique(Rational{Int}.(per_spin))))
            _get_channel_proj_list(ch, ncut, sys.d, sub.κ, Gamma, spin_val, etas_v)
        end
        push!(dims, sum(p.n_r for p in projs; init=0))
    end
    return dims
end
