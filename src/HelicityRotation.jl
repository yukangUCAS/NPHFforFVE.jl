# ============================================================
# HelicityRotation — 螺旋度表象 ↔ 正则极化表象 旋转
# ============================================================
# Wigner D 矩阵实现，支持 j = 0, 1/2, 1。
# D^j(R_st(n)): 把 ẑ 转到 n̂ 的标准转动。
# ============================================================

const _D_CACHE = Dict{Tuple{Momentum, Rational{Int}}, Matrix{ComplexF64}}()
const _ROT_VEC_CACHE = Dict{NamedTuple, Vector{ComplexF64}}()

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
    # m 值: j, j-1, ..., -j
    m_vals = [collect(Float64(j):-1:(-Float64(j))) for j in per_spin]
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
                     V_can_func::Function, extra_args...)
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

    # 对每对唯一动量构型，预计算 V_can 在所有 σ 组合下的值
    V_can_blocks = Dict{Tuple, Matrix{ComplexF64}}()

    for n_α in n_α_unique, n_β in n_β_unique
        block = Matrix{ComplexF64}(undef, dim_σα, dim_σβ)
        σ_vals_α = _σ_configurations(n_α, per_spin_α)
        σ_vals_β = _σ_configurations(n_β, per_spin_β)
        for (i_σα, σα) in enumerate(σ_vals_α)
            for (i_σβ, σβ) in enumerate(σ_vals_β)
                block[i_σα, i_σβ] = V_can_func(n_α, σα, n_β, σβ, extra_args...)
            end
        end
        V_can_blocks[(n_α, n_β)] = block
    end

    # 构造 V_hel
    V_hel = Matrix{ComplexF64}(undef, K_α, K_β)
    for k_α in 1:K_α
        n_α, _ = subspace_states_α[k_α]
        c_α = rot_α[k_α]
        for k_β in 1:K_β
            n_β, _ = subspace_states_β[k_β]
            c_β = rot_β[k_β]
            block = V_can_blocks[(n_α, n_β)]
            s = ComplexF64(0.0, 0.0)
            for i_σα in 1:dim_σα
                ca = conj(c_α[i_σα])
                abs2(ca) == 0.0 && continue
                for i_σβ in 1:dim_σβ
                    cb = c_β[i_σβ]
                    abs2(cb) == 0.0 && continue
                    s += ca * block[i_σα, i_σβ] * cb
                end
            end
            V_hel[k_α, k_β] = s
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

function _spin_projections(j::Real)
    jr = Rational{Int}(j)
    vals = Rational{Int}[]
    v = jr
    while v >= -jr
        push!(vals, v)
        v -= 1
    end
    return vals
end

function get_rotation_vector(n_tuple::NTuple{N, Momentum},
                             lambda_tuple::NTuple{N, <:Real},
                             spins::AbstractVector{<:Real},
                             species::AbstractVector{<:Integer}) where N
    per_spin = _expand_spins(Rational{Int}.(spins), Int.(species))
    return get_rotation_vector(n_tuple, lambda_tuple, per_spin)
end
