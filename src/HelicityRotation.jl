# ============================================================
# HelicityRotation — 螺旋度表象 ↔ 正则极化表象 旋转
# ============================================================
# Wigner D 矩阵实现，支持 j = 0, 1/2, 1。
# D^j(R_st(n)): 把 ẑ 转到 n̂ 的标准转动。
# ============================================================

const _D_CACHE = Dict{Tuple{Momentum, Rational{Int}}, Matrix{ComplexF64}}()
const _ROT_VEC_CACHE = Dict{NamedTuple, Vector{ComplexF64}}()
const _SPIN_PROJ_CACHE = Dict{Rational{Int}, Vector{Rational{Int}}}()
const _M_VALS_CACHE = Dict{Tuple, Vector{Vector{Float64}}}()

# ============ 球坐标 ============

function _sph_coords(n::Momentum)
    r = sqrt(Float64(sum(abs2, n)))
    if r == 0.0
        return 1.0, 0.0, 1.0, 0.0  # θ=0, φ=0: cosθ=1
    end
    cosθ = n[3] / r
    sinθ = sqrt(n[1]^2 + n[2]^2) / r
    if sinθ == 0.0
        # φ 约定: n_z > 0 → φ = 0; n_z < 0 → φ = -π (与 _momentum_to_euler 一致)
        if cosθ < 0.0
            return cosθ, sinθ, -1.0, 0.0  # θ=π, φ=-π
        else
            return cosθ, sinθ, 1.0, 0.0   # θ=0, φ=0
        end
    end
    cosφ = n[1] / (r * sinθ)
    sinφ = n[2] / (r * sinθ)
    return cosθ, sinθ, cosφ, sinφ
end

# ============ Wigner 小 d 矩阵 ============

function _wigner_small_d(j::Rational{Int}, cosθ::Float64, sinθ::Float64)
    if j == 0
        return ComplexF64[1.0+0.0im;;]
    elseif j == 1//2
        c = cos(acos(cosθ) / 2)
        s = sin(acos(cosθ) / 2)
        # 更稳的公式: cos(θ/2) = sqrt((1+cosθ)/2), sin(θ/2) = sqrt((1-cosθ)/2)
        # 但需要处理符号。用标准约定:
        cθ2 = sqrt((1.0 + cosθ) / 2.0)
        sθ2 = (sinθ >= 0 ? 1.0 : -1.0) * sqrt(max(0.0, (1.0 - cosθ) / 2.0))
        return ComplexF64[cθ2+0.0im -sθ2+0.0im;
                          sθ2+0.0im  cθ2+0.0im]
    elseif j == 1
        c = cosθ
        s = sinθ
        c2 = (1.0 + c) / 2.0
        s2 = (1.0 - c) / 2.0
        return ComplexF64[c2+0.0im    -s/sqrt(2)+0.0im    s2+0.0im;
                          s/sqrt(2)+0.0im   c+0.0im   -s/sqrt(2)+0.0im;
                          s2+0.0im     s/sqrt(2)+0.0im    c2+0.0im]
    else
        throw(ArgumentError("自旋 j=$j 暂不支持，仅支持 0, 1/2, 1"))
    end
end

# ============ Wigner D 矩阵 ============

"""
    wigner_D(j::Rational{Int}, n::Momentum) -> Matrix{ComplexF64}

返回 D^j(R_st(n)): 把 ẑ 转到 n̂ 的标准转动 Wigner D 矩阵。
D^j_{m,m'}(φ,θ,0) = e^{-imφ} d^j_{m,m'}(θ)。
零动量返回单位阵。
"""
function wigner_D(j::Rational{Int}, n::Momentum)
    key = (n, j)
    get!(_D_CACHE, key) do
        _compute_wigner_D(j, n)
    end
end

function _compute_wigner_D(j::Rational{Int}, n::Momentum)
    cosθ, sinθ, cosφ, sinφ = _sph_coords(n)
    dmat = _wigner_small_d(j, cosθ, sinθ)
    dim = size(dmat, 1)
    phi = atan(sinφ, cosφ)
    # Normalize to [-π, π) to match _momentum_to_euler convention
    if phi >= pi - 1e-15
        phi = -pi
    end
    # m = j, j-1, ..., -j  (从上到下: 行 1,2,...,dim)
    D = similar(dmat)
    for (i, m) in enumerate(Float64(j):-1:(-Float64(j)))
        phase = exp(ComplexF64(0.0, -m * phi))
        for k in 1:dim
            D[i, k] = phase * dmat[i, k]
        end
    end
    return D
end

# ============ 螺旋度 → 正则极化 旋转系数向量 ============

"""
    get_rotation_vector(n_tuple, lambda_tuple, per_spin) -> Vector{ComplexF64}

返回单个子空间态 (n_tuple, λ_tuple) 的旋转系数向量 c_σ。
c_σ = ∏_i D^{j_i}_{σ_i, λ_i}(R_st(n_i))

per_spin 是每粒子自旋列表 (长度 N, 元素 Rational{Int})。
"""
function get_rotation_vector(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             per_spin::AbstractVector{<:Real}) where N
    lam_r = Rational{Int}.(lambda_tuple)
    spin_r = Rational{Int}.(per_spin)
    key = (n_tuple=n_tuple, lambda_tuple=Tuple(lam_r),
           per_spin=Tuple(spin_r))
    return get!(_ROT_VEC_CACHE, key) do
        _compute_rotation_vector(n_tuple, lam_r, spin_r)
    end
end

function _compute_rotation_vector(n_tuple::NTuple{N, Momentum},
                                  lambda_tuple::NTuple{N, Rational{Int}},
                                  per_spin::Vector{Rational{Int}}) where N
    # 每粒子 D 矩阵和螺旋度索引
    D_mats = [wigner_D(per_spin[i], n_tuple[i]) for i in 1:N]
    # m 值: j, j-1, ..., -j (仅依赖 per_spin)
    m_vals = get!(_M_VALS_CACHE, Tuple(per_spin)) do
        [collect(Float64(j):-1:(-Float64(j))) for j in per_spin]
    end
    # λ 在各 D 矩阵中的列索引 (1-based)
    lam_indices = [findfirst(x -> Float64(x) == Float64(lambda_tuple[i]), m_vals[i])
                   for i in 1:N]
    any(x -> x === nothing, lam_indices) &&
        throw(ArgumentError("螺旋度 $lambda_tuple 不在自旋投影范围内"))

    # 枚举所有 σ 构型
    σ_ranges = [1:length(m_vals[i]) for i in 1:N]
    total_dim = prod(length.(σ_ranges))
    vec = Vector{ComplexF64}(undef, total_dim)

    # 逐 σ 构型计算
    for (flat_idx, σ_indices) in enumerate(Base.Iterators.product(σ_ranges...))
        coeff = ComplexF64(1.0, 0.0)
        for i in 1:N
            coeff *= D_mats[i][σ_indices[i], lam_indices[i]]
        end
        vec[flat_idx] = coeff
    end
    return vec
end

"""
    get_rotation_vector(n_tuple, lambda_tuple, spins, species) -> Vector{ComplexF64}

便利函数: 接受 per-species 的 spins 和 species，自动展开为 per-particle。
"""

# ============ V_can → V_hel 变换 ============

"""
    build_V_hel(subspace_states_α, subspace_states_β,
                per_spin_α, per_spin_β, V_can_func, extra_args...)
        -> Matrix{ComplexF64}

从正则极化表象下的 V_can 构造螺旋度表象下的 V_hel (K_α × K_β)。

# 参数
- `subspace_states_α/β`: `_collect_subspace_states` 返回的态列表，每项 `(n_tuple, λ_tuple)`
- `per_spin_α/β`: 每粒子自旋 (长度 N_α/N_β)
- `V_can_func(n'_tuple, σ'_tuple, n_tuple, σ_tuple, extra_args...)`: 返回 V_can 矩阵元
- `extra_args...`: 透传给 V_can_func 的额外参数 (如 kapA, kapB, rA, rB, aA, aB, params)

V_hel[k', k] = Σ_{σ',σ} conj(c^(k')_{σ'}) · V_can(n'^(k'), σ', n^(k), σ) · c^(k)_{σ}
"""
function build_V_hel(subspace_states_α::Vector,
                     subspace_states_β::Vector,
                     per_spin_α::AbstractVector{<:Real},
                     per_spin_β::AbstractVector{<:Real},
                     V_can_func::Function, extra_args...;
                     V_filter=nothing)
    K_α = length(subspace_states_α)
    K_β = length(subspace_states_β)
    K_α == 0 && return Matrix{ComplexF64}(undef, 0, 0)
    K_β == 0 && return Matrix{ComplexF64}(undef, 0, 0)

    # 预计算所有子空间态的旋转系数向量
    rot_α = [get_rotation_vector(n, lam, per_spin_α) for (n, lam) in subspace_states_α]
    rot_β = [get_rotation_vector(n, lam, per_spin_β) for (n, lam) in subspace_states_β]

    dim_σα = length(rot_α[1])
    dim_σβ = length(rot_β[1])

    # 去重动量构型 (相同 n_tuple 的态共享正则极化基)
    n_α_unique = unique!([n for (n, _) in subspace_states_α])
    n_β_unique = unique!([n for (n, _) in subspace_states_β])

    # 探测 V_can 返回值类型: 标量 → dim_κ=1; 矩阵 → 取行/列数
    σ_probe_α = _σ_configurations(n_α_unique[1], per_spin_α)
    σ_probe_β = _σ_configurations(n_β_unique[1], per_spin_β)
    sample = V_can_func(n_α_unique[1], σ_probe_α[1],
                        n_β_unique[1], σ_probe_β[1], extra_args...)
    dim_κA = sample isa Number ? 1 : size(sample, 1)
    dim_κB = sample isa Number ? 1 : size(sample, 2)

    # 动量 → helicity 态索引: 相同动量 n 的态共享 V_can 块
    n_to_k_α = Dict{eltype(n_α_unique), Vector{Int}}()
    for (k, (n, _)) in enumerate(subspace_states_α)
        push!(get!(Vector{Int}, n_to_k_α, n), k)
    end
    n_to_k_β = Dict{eltype(n_β_unique), Vector{Int}}()
    for (k, (n, _)) in enumerate(subspace_states_β)
        push!(get!(Vector{Int}, n_to_k_β, n), k)
    end

    # 融合: 计算 V_can 块并直接散射到 V_hel, 无中间 Dict
    n_α_list = collect(n_α_unique)
    n_β_list = collect(n_β_unique)
    N_α = length(n_α_list)
    N_β = length(n_β_list)
    n_pairs = N_α * N_β

    V_hel = zeros(ComplexF64, K_α * dim_κA, K_β * dim_κB)

    # σ 构型仅依赖 per_spin，与 n 无关 → 预计算一次
    σ_vals_all_α = _σ_configurations(per_spin_α)
    σ_vals_all_β = _σ_configurations(per_spin_β)

    # 线程局部 block 缓冲区池 (避免 per-pair zeros 分配)
    blk_rows = dim_σα * dim_κA
    blk_cols = dim_σβ * dim_κB
    n_threads = Threads.nthreads()
    block_pool = [zeros(ComplexF64, blk_rows, blk_cols) for _ in 1:n_threads]

    Threads.@threads for idx in 1:n_pairs
        i_α = (idx - 1) ÷ N_β + 1
        i_β = (idx - 1) % N_β + 1
        n_α = n_α_list[i_α]
        n_β = n_β_list[i_β]

        # 轻量预筛: 先试一个 σ 对，filter 全拒则跳过
        σ_vals_α = σ_vals_all_α
        σ_vals_β = σ_vals_all_β
        if V_filter !== nothing
            any_pass = false
            for σα in σ_vals_α, σβ in σ_vals_β
                if V_filter(n_α, σα, n_β, σβ)
                    any_pass = true
                    break
                end
            end
            any_pass || continue
        end

        # 取线程局部 block，fill! 清零复用
        block = block_pool[Threads.threadid()]
        fill!(block, 0)
        any_nonzero = false
        for (i_σα, σα) in enumerate(σ_vals_α)
            r0_b = (i_σα - 1) * dim_κA + 1
            for (i_σβ, σβ) in enumerate(σ_vals_β)
                c0_b = (i_σβ - 1) * dim_κB + 1
                if V_filter !== nothing && !V_filter(n_α, σα, n_β, σβ)
                    continue
                end
                val = V_can_func(n_α, σα, n_β, σβ, extra_args...)
                if val isa Number
                    iszero(val) && continue
                    block[r0_b, c0_b] = val
                    any_nonzero = true
                else
                    all(iszero, val) && continue
                    block[r0_b:r0_b+dim_κA-1, c0_b:c0_b+dim_κB-1] .= val
                    any_nonzero = true
                end
            end
        end
        any_nonzero || continue

        # 散射到所有对应 helicity 态
        k_list_α = n_to_k_α[n_α]
        k_list_β = n_to_k_β[n_β]
        for k_α in k_list_α
            c_α = rot_α[k_α]
            r0 = (k_α - 1) * dim_κA + 1
            for k_β in k_list_β
                c_β = rot_β[k_β]
                c0 = (k_β - 1) * dim_κB + 1
                for a in 1:dim_κA, b in 1:dim_κB
                    s = ComplexF64(0.0, 0.0)
                    for i_σα in 1:dim_σα
                        ca = conj(c_α[i_σα])
                        abs2(ca) == 0.0 && continue
                        for i_σβ in 1:dim_σβ
                            cb = c_β[i_σβ]
                            abs2(cb) == 0.0 && continue
                            s += ca * block[(i_σα-1)*dim_κA+a, (i_σβ-1)*dim_κB+b] * cb
                        end
                    end
                    V_hel[r0+a-1, c0+b-1] = s
                end
            end
        end
    end

    return V_hel
end

# ============ V_hel 直接输入 (无 Wigner 旋转) ============

"""
    build_V_hel_direct(subspace_states_α, subspace_states_β,
                       V_hel_adapter, extra_args...) -> Matrix{ComplexF64}

直接从螺旋度表象下的 V_hel 矩阵元构造 V_hel (K_α·dim_κA × K_β·dim_κB)。

与 `build_V_hel` 的区别：不需要 Wigner D 旋转，不需要 σ 枚举。
用户提供的 `V_hel_adapter(n_α, λ_α, n_β, λ_β, extra_args...)` 直接返回
螺旋度基下的矩阵元（标量或 dim_κA × dim_κB 矩阵）。
"""
function build_V_hel_direct(subspace_states_α::Vector,
                            subspace_states_β::Vector,
                            V_hel_adapter::Function, extra_args...;
                            V_filter=nothing)
    K_α = length(subspace_states_α)
    K_β = length(subspace_states_β)
    K_α == 0 && return Matrix{ComplexF64}(undef, 0, 0)
    K_β == 0 && return Matrix{ComplexF64}(undef, 0, 0)

    # 探测 dim_κ
    n_α_0, λ_α_0 = subspace_states_α[1]
    n_β_0, λ_β_0 = subspace_states_β[1]
    sample = V_hel_adapter(n_α_0, λ_α_0, n_β_0, λ_β_0, extra_args...)
    dim_κA = sample isa Number ? 1 : size(sample, 1)
    dim_κB = sample isa Number ? 1 : size(sample, 2)

    V_hel = zeros(ComplexF64, K_α * dim_κA, K_β * dim_κB)
    for k_α in 1:K_α
        n_α, λ_α = subspace_states_α[k_α]
        r0 = (k_α - 1) * dim_κA + 1
        for k_β in 1:K_β
            n_β, λ_β = subspace_states_β[k_β]
            c0 = (k_β - 1) * dim_κB + 1
            V_filter !== nothing && !V_filter(n_α, λ_α, n_β, λ_β) && continue
            val = V_hel_adapter(n_α, λ_α, n_β, λ_β, extra_args...)
            if val isa Number
                V_hel[r0, c0] = val
            else
                V_hel[r0:r0+dim_κA-1, c0:c0+dim_κB-1] .= val
            end
        end
    end
    return V_hel
end

"""
    _σ_configurations(n_tuple, per_spin) -> Vector{NTuple{N, Rational{Int}}}

返回给定动量构型 n_tuple 下所有可能的正则极化 σ 构型。
σ_i ∈ {-j_i, -j_i+1, ..., j_i}。
"""
function _σ_configurations(n_tuple::NTuple{N, Momentum},
                           per_spin::AbstractVector{<:Real}) where N
    m_ranges = [_spin_projections(j) for j in per_spin]
    return collect(Iterators.product(m_ranges...))
end

function _σ_configurations(per_spin::AbstractVector{<:Real})
    m_ranges = [_spin_projections(j) for j in per_spin]
    return collect(Iterators.product(m_ranges...))
end

function _spin_projections(j::Real)
    jr = Rational{Int}(j)
    get!(_SPIN_PROJ_CACHE, jr) do
        vals = Rational{Int}[]
        v = jr
        while v >= -jr
            push!(vals, v)
            v -= 1
        end
        vals
    end
end

function get_rotation_vector(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             spins::AbstractVector{<:Real},
                             species::AbstractVector{<:Integer}) where N
    per_spin = _expand_spins(Rational{Int}.(spins), Int.(species))
    return get_rotation_vector(n_tuple, lambda_tuple, per_spin)
end
