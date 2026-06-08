module SymmetryGroup

using StaticArrays, LinearAlgebra

export group_elements, group_for_momentum, apply_transform
export O_h, C4v, C2v, C3v
export O_h2, C4v2, C3v2, C2v2

# ============ O_h 群 48 个群元（3×3 整数矩阵）============

const _OH_ALL = SMatrix{3,3,Int}[
    # 1-24: 纯旋转（行列式 +1）
    SMatrix{3,3,Int}( 1, 0, 0,  0, 1, 0,  0, 0, 1),   #  1: 恒等
    SMatrix{3,3,Int}( 0, 0, 1,  1, 0, 0,  0, 1, 0),   #  2
    SMatrix{3,3,Int}( 0, 1, 0,  0, 0, 1,  1, 0, 0),   #  3
    SMatrix{3,3,Int}( 0,-1, 0,  0, 0, 1, -1, 0, 0),   #  4
    SMatrix{3,3,Int}( 0, 0,-1, -1, 0, 0,  0, 1, 0),   #  5
    SMatrix{3,3,Int}( 0, 0,-1,  1, 0, 0,  0,-1, 0),   #  6
    SMatrix{3,3,Int}( 0, 1, 0,  0, 0,-1, -1, 0, 0),   #  7
    SMatrix{3,3,Int}( 0,-1, 0,  0, 0,-1,  1, 0, 0),   #  8
    SMatrix{3,3,Int}( 0, 0, 1, -1, 0, 0,  0,-1, 0),   #  9
    SMatrix{3,3,Int}( 1, 0, 0,  0, 0,-1,  0, 1, 0),   # 10
    SMatrix{3,3,Int}( 1, 0, 0,  0, 0, 1,  0,-1, 0),   # 11
    SMatrix{3,3,Int}( 0, 0, 1,  0, 1, 0, -1, 0, 0),   # 12
    SMatrix{3,3,Int}( 0, 0,-1,  0, 1, 0,  1, 0, 0),   # 13
    SMatrix{3,3,Int}( 0,-1, 0,  1, 0, 0,  0, 0, 1),   # 14
    SMatrix{3,3,Int}( 0, 1, 0, -1, 0, 0,  0, 0, 1),   # 15
    SMatrix{3,3,Int}(-1, 0, 0,  0, 0, 1,  0, 1, 0),   # 16
    SMatrix{3,3,Int}(-1, 0, 0,  0, 0,-1,  0,-1, 0),   # 17
    SMatrix{3,3,Int}( 0, 1, 0,  1, 0, 0,  0, 0,-1),   # 18
    SMatrix{3,3,Int}( 0,-1, 0, -1, 0, 0,  0, 0,-1),   # 19
    SMatrix{3,3,Int}( 0, 0, 1,  0,-1, 0,  1, 0, 0),   # 20
    SMatrix{3,3,Int}( 0, 0,-1,  0,-1, 0, -1, 0, 0),   # 21
    SMatrix{3,3,Int}( 1, 0, 0,  0,-1, 0,  0, 0,-1),   # 22
    SMatrix{3,3,Int}(-1, 0, 0,  0, 1, 0,  0, 0,-1),   # 23
    SMatrix{3,3,Int}(-1, 0, 0,  0,-1, 0,  0, 0, 1),   # 24
    # 25-48: 反演 × 纯旋转（行列式 -1）
    SMatrix{3,3,Int}(-1, 0, 0,  0,-1, 0,  0, 0,-1),   # 25
    SMatrix{3,3,Int}( 0, 0,-1, -1, 0, 0,  0,-1, 0),   # 26
    SMatrix{3,3,Int}( 0,-1, 0,  0, 0,-1, -1, 0, 0),   # 27
    SMatrix{3,3,Int}( 0, 1, 0,  0, 0,-1,  1, 0, 0),   # 28
    SMatrix{3,3,Int}( 0, 0, 1,  1, 0, 0,  0,-1, 0),   # 29
    SMatrix{3,3,Int}( 0, 0, 1, -1, 0, 0,  0, 1, 0),   # 30
    SMatrix{3,3,Int}( 0,-1, 0,  0, 0, 1,  1, 0, 0),   # 31
    SMatrix{3,3,Int}( 0, 1, 0,  0, 0, 1, -1, 0, 0),   # 32
    SMatrix{3,3,Int}( 0, 0,-1,  1, 0, 0,  0, 1, 0),   # 33
    SMatrix{3,3,Int}(-1, 0, 0,  0, 0, 1,  0,-1, 0),   # 34
    SMatrix{3,3,Int}(-1, 0, 0,  0, 0,-1,  0, 1, 0),   # 35
    SMatrix{3,3,Int}( 0, 0,-1,  0,-1, 0,  1, 0, 0),   # 36
    SMatrix{3,3,Int}( 0, 0, 1,  0,-1, 0, -1, 0, 0),   # 37
    SMatrix{3,3,Int}( 0, 1, 0, -1, 0, 0,  0, 0,-1),   # 38
    SMatrix{3,3,Int}( 0,-1, 0,  1, 0, 0,  0, 0,-1),   # 39
    SMatrix{3,3,Int}( 1, 0, 0,  0, 0,-1,  0,-1, 0),   # 40
    SMatrix{3,3,Int}( 1, 0, 0,  0, 0, 1,  0, 1, 0),   # 41
    SMatrix{3,3,Int}( 0,-1, 0, -1, 0, 0,  0, 0, 1),   # 42
    SMatrix{3,3,Int}( 0, 1, 0,  1, 0, 0,  0, 0, 1),   # 43
    SMatrix{3,3,Int}( 0, 0,-1,  0, 1, 0, -1, 0, 0),   # 44
    SMatrix{3,3,Int}( 0, 0, 1,  0, 1, 0,  1, 0, 0),   # 45
    SMatrix{3,3,Int}(-1, 0, 0,  0, 1, 0,  0, 0, 1),   # 46
    SMatrix{3,3,Int}( 1, 0, 0,  0,-1, 0,  0, 0, 1),   # 47
    SMatrix{3,3,Int}( 1, 0, 0,  0, 1, 0,  0, 0,-1),   # 48
]

# ============ 子群定义 ============

# C_4v: O_h 子群，对应总动量 (0,0,1)
const _C4V_INDICES = [1, 14, 15, 24, 42, 43, 46, 47]

# C_2v: O_h 子群，对应总动量 (0,1,1)
const _C2V_INDICES = [1, 16, 41, 46]

# C_3v: O_h 子群，对应总动量 (1,1,1)
const _C3V_INDICES = [1, 2, 3, 41, 43, 45]

# 构造子群矩阵列表
const _C4V_ALL = [_OH_ALL[i] for i in _C4V_INDICES]
const _C2V_ALL = [_OH_ALL[i] for i in _C2V_INDICES]
const _C3V_ALL = [_OH_ALL[i] for i in _C3V_INDICES]

# ============ 公开常量（类型别名） ============

const O_h = _OH_ALL
const C4v = _C4V_ALL
const C2v = _C2V_ALL
const C3v = _C3V_ALL

# ============ 双覆盖群元（O(3) 矩阵重复两次，两次 SU(2) 提升）============

const _OH2_ALL  = vcat(_OH_ALL, _OH_ALL)
const _C4V2_ALL = vcat(_C4V_ALL, _C4V_ALL)
const _C3V2_ALL = vcat(_C3V_ALL, _C3V_ALL)
const _C2V2_ALL = vcat(_C2V_ALL, _C2V_ALL)

const O_h2 = _OH2_ALL
const C4v2 = _C4V2_ALL
const C3v2 = _C3V2_ALL
const C2v2 = _C2V2_ALL

# ============ API 函数 ============

"""
    group_elements(group::Symbol) -> Vector{SMatrix{3,3,Int}}

返回指定对称群的所有群元矩阵。

单覆盖:
- `:Oh`  — O_h，48 个群元
- `:C4v` — C_4v，8 个群元
- `:C2v` — C_2v，4 个群元
- `:C3v` — C_3v，6 个群元

双覆盖:
- `:Oh2`  — 2O_h，96 个群元（48 O(3) 矩阵重复两次）
- `:C4v2` — C_4v 双覆盖，16 个群元
- `:C2v2` — C_2v 双覆盖，8 个群元
- `:C3v2` — C_3v 双覆盖，12 个群元
"""
function group_elements(group::Symbol)
    if group == :Oh
        return _OH_ALL
    elseif group == :C4v
        return _C4V_ALL
    elseif group == :C2v
        return _C2V_ALL
    elseif group == :C3v
        return _C3V_ALL
    elseif group == :Oh2
        return _OH2_ALL
    elseif group == :C4v2
        return _C4V2_ALL
    elseif group == :C2v2
        return _C2V2_ALL
    elseif group == :C3v2
        return _C3V2_ALL
    else
        throw(ArgumentError("未知对称群 :$(group)，可选 :Oh, :Oh2, :C4v, :C4v2, :C2v, :C2v2, :C3v, :C3v2"))
    end
end

"""
    group_for_momentum(d; double_cover=false)

根据三动量自动返回对应的对称群。

| 动量     | 单覆盖    | 双覆盖     |
|----------|-----------|------------|
| (0,0,0)  | O_h (48)  | 2O_h (96)  |
| (0,0,1)  | C_4v (8)  | C_4v (16)  |
| (0,1,1)  | C_2v (4)  | C_2v (8)   |
| (1,1,1)  | C_3v (6)  | C_3v (12)  |

当 `double_cover=true` 时返回双覆盖群元素（O(3) 矩阵重复两次）。
"""
function group_for_momentum(d; double_cover::Bool=false)
    if d isa SVector{3,Int}
        dv = d
    elseif d isa NTuple{3,Int}
        dv = SVector{3,Int}(d)
    elseif d isa AbstractVector{<:Integer}
        dv = SVector{3,Int}(d[1], d[2], d[3])
    else
        throw(ArgumentError("动量 d 必须是 SVector{3,Int}, NTuple{3,Int} 或整数向量"))
    end

    a = abs.(dv)
    if a == SVector(0, 0, 0)
        return double_cover ? (_OH2_ALL, :Oh2) : (_OH_ALL, :Oh)
    elseif a == SVector(0, 0, 1)
        return double_cover ? (_C4V2_ALL, :C4v2) : (_C4V_ALL, :C4v)
    elseif a == SVector(0, 1, 1)
        return double_cover ? (_C2V2_ALL, :C2v2) : (_C2V_ALL, :C2v)
    elseif a == SVector(1, 1, 1)
        return double_cover ? (_C3V2_ALL, :C3v2) : (_C3V_ALL, :C3v)
    else
        throw(ArgumentError("未找到与动量 $dv 对应的对称群"))
    end
end

"""
    apply_transform(g::SMatrix{3,3,Int}, n)

将群元 g 作用于动量矢量 n 上（矩阵乘法）。
"""
function apply_transform(g::SMatrix{3,3,Int}, n)
    return g * n
end

"""
    apply_transform(g::SMatrix{3,3,Int}, n::NTuple{N, Momentum}) where N

将群元 g 作用于 N 粒子态（每个动量同时变换）。
"""
function apply_transform(g::SMatrix{3,3,Int}, state::NTuple{N, <:Any}) where N
    return ntuple(i -> g * state[i], N)
end

# ============ 不可约表示矩阵（O_h 群）============

export irrep_matrices, irrep_matrix, OH_IRREP_NAMES

const OH_IRREP_NAMES = ["A1+", "A2+", "E+", "T1+", "T2+",
                         "A1-", "A2-", "E-", "T1-", "T2-"]

# 辅助：由 (I,J,V) 三元组构造 3×3 矩阵
function _sp3(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64})
    M = zeros(3, 3)
    @inbounds for k in 1:3
        M[I[k], J[k]] = V[k]
    end
    return M
end

function _build_oh_irreps()
    irr = Dict{String, Vector{Matrix{Float64}}}()

    # ---- A1+, A1- (1×1) ----
    A1p = [fill(1.0, 1, 1) for _ in 1:48]
    A1m = [i <= 24 ? fill(1.0, 1, 1) : fill(-1.0, 1, 1) for i in 1:48]
    irr["A1+"] = A1p
    irr["A1-"] = A1m

    # ---- A2+, A2- (1×1) ----
    A2p = Vector{Matrix{Float64}}(undef, 48)
    A2m = Vector{Matrix{Float64}}(undef, 48)
    for i in 1:48
        if i <= 24
            val = (10 <= i <= 21) ? -1.0 : 1.0
            A2p[i] = fill(val, 1, 1)
            A2m[i] = fill(val, 1, 1)
        else
            A2p[i] = A2p[i-24]
            A2m[i] = -A2p[i-24]
        end
    end
    irr["A2+"] = A2p
    irr["A2-"] = A2m

    # ---- E+, E- (2×2) ----
    s3 = sqrt(3.0)
    E_identity = Matrix(1.0I, 2, 2)
    Ep = Vector{Matrix{Float64}}(undef, 48)
    Em = Vector{Matrix{Float64}}(undef, 48)
    for i in 1:48
        if i <= 24
            Ep[i] = if i == 1
                copy(E_identity)
            elseif i == 2
                [-1.0  s3; -s3  -1.0] / 2.0
            elseif i in (3, 4)
                [-1.0  -s3;  s3  -1.0] / 2.0
            elseif i in (5, 6, 9)
                [-1.0  s3; -s3  -1.0] / 2.0
            elseif i in (10, 11, 16, 17)
                [-1.0  -s3; -s3  1.0] / 2.0
            elseif i in (12, 13, 20, 21)
                [-1.0  s3;  s3  1.0] / 2.0
            elseif i in (14, 15, 18, 19)
                [1.0  0.0; 0.0  -1.0]
            elseif i in (7, 8)
                [-1.0  -s3;  s3  -1.0] / 2.0
            else
                copy(E_identity)
            end
            Em[i] = Ep[i]
        else
            Ep[i] = Ep[i-24]
            Em[i] = -Ep[i-24]
        end
    end
    irr["E+"] = Ep
    irr["E-"] = Em

    # ---- T1+, T1-, T2+, T2- (3×3) ----
    function _build_T1p_sub()
        mats = Vector{Matrix{Float64}}(undef, 24)
        for i in 1:24
            mats[i] = if i == 1
                Matrix(1.0I, 3, 3)
            elseif i == 2
                _sp3([1,2,3], [2,3,1], [1.0, 1.0, 1.0])
            elseif i == 3
                _sp3([1,2,3], [3,1,2], [1.0, 1.0, 1.0])
            elseif i == 4
                _sp3([1,2,3], [3,1,2], [-1.0, -1.0, 1.0])
            elseif i == 5
                _sp3([1,2,3], [2,3,1], [-1.0, 1.0, -1.0])
            elseif i == 6
                _sp3([1,2,3], [2,3,1], [1.0, -1.0, -1.0])
            elseif i == 7
                _sp3([1,2,3], [3,1,2], [-1.0, 1.0, -1.0])
            elseif i == 8
                _sp3([1,2,3], [3,1,2], [1.0, -1.0, -1.0])
            elseif i == 9
                _sp3([1,2,3], [2,3,1], [-1.0, -1.0, 1.0])
            elseif i == 10
                _sp3([1,2,3], [1,3,2], [1.0, 1.0, -1.0])
            elseif i == 11
                _sp3([1,2,3], [1,3,2], [1.0, -1.0, 1.0])
            elseif i == 12
                _sp3([1,2,3], [3,2,1], [-1.0, 1.0, 1.0])
            elseif i == 13
                _sp3([1,2,3], [3,2,1], [1.0, 1.0, -1.0])
            elseif i == 14
                _sp3([1,2,3], [2,1,3], [1.0, -1.0, 1.0])
            elseif i == 15
                _sp3([1,2,3], [2,1,3], [-1.0, 1.0, 1.0])
            elseif i == 16
                _sp3([1,2,3], [1,3,2], [-1.0, 1.0, 1.0])
            elseif i == 17
                _sp3([1,2,3], [1,3,2], [-1.0, -1.0, -1.0])
            elseif i == 18
                _sp3([1,2,3], [2,1,3], [1.0, 1.0, -1.0])
            elseif i == 19
                _sp3([1,2,3], [2,1,3], [-1.0, -1.0, -1.0])
            elseif i == 20
                _sp3([1,2,3], [3,2,1], [1.0, -1.0, 1.0])
            elseif i == 21
                _sp3([1,2,3], [3,2,1], [-1.0, -1.0, -1.0])
            elseif i == 22
                _sp3([1,2,3], [1,2,3], [1.0, -1.0, -1.0])
            elseif i == 23
                _sp3([1,2,3], [1,2,3], [-1.0, 1.0, -1.0])
            elseif i == 24
                _sp3([1,2,3], [1,2,3], [-1.0, -1.0, 1.0])
            end
        end
        return mats
    end

    T1p_sub = _build_T1p_sub()
    T1p = Vector{Matrix{Float64}}(undef, 48)
    T1m = Vector{Matrix{Float64}}(undef, 48)
    T2p = Vector{Matrix{Float64}}(undef, 48)
    T2m = Vector{Matrix{Float64}}(undef, 48)

    for i in 1:48
        if i <= 24
            T1p[i] = T1p_sub[i]
            T1m[i] = T1p_sub[i]
            T2p[i] = (10 <= i <= 21) ? -T1p_sub[i] : T1p_sub[i]
            T2m[i] = T2p[i]
        else
            T1p[i] = T1p_sub[i-24]
            T1m[i] = -T1p_sub[i-24]
            T2p[i] = T2p[i-24]
            T2m[i] = -T2p[i-24]
        end
    end

    irr["T1+"] = T1p
    irr["T1-"] = T1m
    irr["T2+"] = T2p
    irr["T2-"] = T2m

    return irr
end

const _OH_IRREPS = _build_oh_irreps()

"""
    irrep_matrices(irrep::String; group::Symbol=:Oh) -> Vector{Matrix}

返回不可约表示矩阵。`group` 可选：
- `:Oh` — O_h 群（48），10 个玻色子不可约表示
- `:Oh2` — 2O_h 双覆盖群（96），16 个不可约表示（10 玻色子 + 6 费米子）
- `:C4v`, `:C2v`, `:C3v` — 小群（运动系），使用 Morningstar 约定的小群原生不可约表示。

小群不可约表示：
  C_4v: A1,A2,B1,B2,E (玻色子), G1,G2 (费米子)
  C_3v: A1,A2,E (玻色子), F1,F2,G (费米子)
  C_2v: A1,A2,B1,B2 (玻色子), G (费米子)
  费米子不可约表示自动使用双覆盖群。

# 示例
```julia
D = irrep_matrices("T1+")              # O_h 群 T1+，48 个 3×3 矩阵
D = irrep_matrices("G1+"; group=:Oh2)  # 2O_h 费米子不可约表示，96 个 2×2 矩阵
D = irrep_matrices("E"; group=:C4v)    # C_4v 小群 E 不可约表示（玻色子），8 个 2×2 矩阵
D = irrep_matrices("G1"; group=:C4v)   # C_4v 小群 G1 不可约表示（费米子），16 个 2×2 矩阵
```
"""
function irrep_matrices(irrep::String; group::Symbol=:Oh)
    if group == :Oh
        return _OH_IRREPS[irrep]
    elseif group == :Oh2
        return _OH2_IRREPS[irrep]
    elseif group == :C4v
        if irrep in C4V_BOSONIC_NAMES
            return _C4V_IRREPS_SINGLE[irrep]
        elseif irrep in C4V_FERMIONIC_NAMES
            return _C4VD_IRREPS[irrep]
        else
            throw(ArgumentError("未知 C_4v 不可约表示 :$(irrep)，可选 $(C4V_IRREP_NAMES)"))
        end
    elseif group == :C4v2
        return _C4VD_IRREPS[irrep]
    elseif group == :C2v
        if irrep in C2V_BOSONIC_NAMES
            return _C2V_IRREPS_SINGLE[irrep]
        elseif irrep in C2V_FERMIONIC_NAMES
            return _C2VD_IRREPS[irrep]
        else
            throw(ArgumentError("未知 C_2v 不可约表示 :$(irrep)，可选 $(C2V_IRREP_NAMES)"))
        end
    elseif group == :C2v2
        return _C2VD_IRREPS[irrep]
    elseif group == :C3v
        if irrep in C3V_BOSONIC_NAMES
            return _C3V_IRREPS_SINGLE[irrep]
        elseif irrep in C3V_FERMIONIC_NAMES
            return _C3VD_IRREPS[irrep]
        else
            throw(ArgumentError("未知 C_3v 不可约表示 :$(irrep)，可选 $(C3V_IRREP_NAMES)"))
        end
    elseif group == :C3v2
        return _C3VD_IRREPS[irrep]
    else
        throw(ArgumentError("未知对称群 :$(group)，可选 :Oh, :Oh2, :C4v, :C4v2, :C2v, :C2v2, :C3v, :C3v2"))
    end
end

"""
    irrep_matrix(irrep::String, i::Int; group::Symbol=:Oh) -> Matrix

返回不可约表示 `irrep` 在群元 `g_i` 处的表示矩阵 D^Γ(g_i)。
`i` 是群元在 `group_elements(group)` 中的 1-indexed 序号。
"""
function irrep_matrix(irrep::String, i::Int; group::Symbol=:Oh)
    if group == :Oh
        return _OH_IRREPS[irrep][i]
    elseif group == :Oh2
        return _OH2_IRREPS[irrep][i]
    else
        mats = irrep_matrices(irrep; group=group)
        return mats[i]
    end
end

# ============ O_h 双覆盖群 2O_h（96 个群元，费米子体系）============

export OH2_IRREP_NAMES, OH2_BOSONIC_IRREP_NAMES, OH2_FERMIONIC_IRREP_NAMES

const OH2_BOSONIC_IRREP_NAMES  = OH_IRREP_NAMES  # 10 个玻色子不可约表示
const OH2_FERMIONIC_IRREP_NAMES = ["G1+", "G2+", "H+", "G1-", "G2-", "H-"]
const OH2_IRREP_NAMES = vcat(OH2_BOSONIC_IRREP_NAMES, OH2_FERMIONIC_IRREP_NAMES)

# Euler 角参数表（i=1..24 对应 proper 旋转）
# (α, β, γ) 来自 Bernard:2008ax Tab. 2
# ---------- O_h 纯旋转的轴-角参数（按 _OH_ALL[1:24] 顺序） ----------
# 数据源自 Bernard:2008ax Tab.2，按 _OH_ALL 顺序重排。
# D^(J)(n, ω) = exp(-i ω n·J)
const _OH_ROTATION_PARAMS = [
    ([0.0, 0.0, 1.0],                    0.0),        #  1: I
    ([1.0, 1.0, 1.0] ./ sqrt(3),         -2pi/3),      #  2: 8C3
    ([1.0, 1.0, 1.0] ./ sqrt(3),          2pi/3),      #  3
    ([-1.0, 1.0, 1.0] ./ sqrt(3),        -2pi/3),      #  4
    ([-1.0, 1.0, 1.0] ./ sqrt(3),         2pi/3),      #  5
    ([-1.0, -1.0, 1.0] ./ sqrt(3),       -2pi/3),      #  6
    ([-1.0, -1.0, 1.0] ./ sqrt(3),        2pi/3),      #  7
    ([1.0, -1.0, 1.0] ./ sqrt(3),        -2pi/3),      #  8
    ([1.0, -1.0, 1.0] ./ sqrt(3),         2pi/3),      #  9
    ([1.0, 0.0, 0.0],                    -pi/2),       # 10: 6C4
    ([1.0, 0.0, 0.0],                     pi/2),       # 11
    ([0.0, 1.0, 0.0],                    -pi/2),       # 12
    ([0.0, 1.0, 0.0],                     pi/2),       # 13
    ([0.0, 0.0, 1.0],                    -pi/2),       # 14
    ([0.0, 0.0, 1.0],                     pi/2),       # 15
    ([0.0, 1.0, 1.0] ./ sqrt(2),         -pi),         # 16: 6C2'
    ([0.0, -1.0, 1.0] ./ sqrt(2),        -pi),         # 17
    ([1.0, 1.0, 0.0] ./ sqrt(2),         -pi),         # 18
    ([1.0, -1.0, 0.0] ./ sqrt(2),        -pi),         # 19
    ([1.0, 0.0, 1.0] ./ sqrt(2),         -pi),         # 20
    ([-1.0, 0.0, 1.0] ./ sqrt(2),        -pi),         # 21
    ([1.0, 0.0, 0.0],                    -pi),         # 22: 3C2
    ([0.0, 1.0, 0.0],                    -pi),         # 23
    ([0.0, 0.0, 1.0],                    -pi),         # 24
]

# ---------- Wigner D 矩阵（轴-角参数化：D^(J)(n, ω) = exp(-i ω n·J)） ----------

# Pauli 矩阵
const _SX = ComplexF64[0 1; 1 0]
const _SY = ComplexF64[0 -im; im 0]
const _SZ = ComplexF64[1 0; 0 -1]

"""
    _wigner_D_half(n::Vector{Float64}, omega::Float64) -> Matrix{ComplexF64}

D^{1/2}(n, ω) = cos(ω/2) I - i sin(ω/2) (n·σ)
"""
function _wigner_D_half(n::Vector{Float64}, omega::Float64)
    c = cos(omega / 2)
    s = sin(omega / 2)
    return c * I - im * s * (n[1]*_SX + n[2]*_SY + n[3]*_SZ)
end

# Pre-computed SU(2) matrices for all 24 proper Oh rotations using parameter table.
# Used by _so3_to_su2 to ensure consistent SU(2) lifting with irrep matrices.
const _OH_PROPER_SU2 = Dict{SMatrix{3,3,Int}, Matrix{ComplexF64}}(
    _OH_ALL[i] => _wigner_D_half(_OH_ROTATION_PARAMS[i][1], _OH_ROTATION_PARAMS[i][2])
    for i in 1:24
)

# J=1 生成元（|1,m⟩ 基，m = +1, 0, -1，Condon-Shortley 相位约定）
const _JX_ONE = ComplexF64[0 1 0; 1 0 1; 0 1 0] / sqrt(2.0)
const _JY_ONE = ComplexF64[0 -im 0; im 0 -im; 0 im 0] / sqrt(2.0)
const _JZ_ONE = ComplexF64[1 0 0; 0 0 0; 0 0 -1]

"""
    _wigner_D_one(n::Vector{Float64}, omega::Float64) -> Matrix{ComplexF64}

D^{1}(n, ω) = exp(-i ω n·J)，通过矩阵指数计算。
"""
function _wigner_D_one(n::Vector{Float64}, omega::Float64)
    nJ = n[1]*_JX_ONE + n[2]*_JY_ONE + n[3]*_JZ_ONE
    return exp(-im * omega * nJ)
end

# J=3/2 生成元（同前，保留供 _wigner_D_threehalf 使用）
const _S32 = sqrt(3.0) / 2
const _JX_THREEHALF = ComplexF64[
    0  _S32  0      0
    _S32  0  1      0
    0      1  0  _S32
    0      0  _S32  0
]
const _JY_THREEHALF = ComplexF64[
    0  -im*_S32  0        0
    im*_S32  0  -im       0
    0        im  0  -im*_S32
    0        0  im*_S32  0
]
const _JZ_THREEHALF = ComplexF64[
    3/2  0    0     0
    0   1/2   0     0
    0   0   -1/2    0
    0   0    0   -3/2
]

"""
    _wigner_D_threehalf(n::Vector{Float64}, omega::Float64) -> Matrix{ComplexF64}

D^{3/2}(n, ω) = exp(-i ω n·J)，通过矩阵指数计算。
"""
function _wigner_D_threehalf(n::Vector{Float64}, omega::Float64)
    nJ = n[1]*_JX_THREEHALF + n[2]*_JY_THREEHALF + n[3]*_JZ_THREEHALF
    return exp(-im * omega * nJ)
end

# ---------- 构建 2O_h 不可约表示 ----------

function _build_oh2_irreps()
    irr = Dict{String, Vector{Matrix{ComplexF64}}}()

    # ---- 玻色子不可约表示：从 O_h 扩展到 96 元素 ----
    for name in OH_IRREP_NAMES
        mats_oh = _OH_IRREPS[name]  # 48 个 Float64 矩阵
        mats_96 = Vector{Matrix{ComplexF64}}(undef, 96)
        for i in 1:48
            mats_96[i]      = ComplexF64.(mats_oh[i])
            mats_96[i + 48] = ComplexF64.(mats_oh[i])  # D(ḡ) = D(g)
        end
        irr[name] = mats_96
    end

    # ---- 费米子不可约表示 ----
    for (name, J) in [("G1", :half), ("G2", :half), ("H", :threehalf)]
        proper_mats = Vector{Matrix{ComplexF64}}(undef, 24)
        for i in 1:24
            n, omega = _OH_ROTATION_PARAMS[i]
            if J == :half
                proper_mats[i] = _wigner_D_half(n, omega)
            else
                proper_mats[i] = _wigner_D_threehalf(n, omega)
            end
            # G2: 对 10≤i≤21 符号翻转（与 A2/T2 一致）
            if name == "G2" && 10 <= i <= 21
                proper_mats[i] = -proper_mats[i]
            end
        end

        for parity in ["+", "-"]
            mats = Vector{Matrix{ComplexF64}}(undef, 96)
            parity_sign = (parity == "+" ? 1.0 : -1.0)

            for i in 1:24
                mats[i]      = proper_mats[i]
                mats[i + 24] = parity_sign * proper_mats[i]
                mats[i + 48] = -proper_mats[i]               # D(ḡ) = -D(g) for fermions
                mats[i + 72] = -parity_sign * proper_mats[i]
            end

            irr[name * parity] = mats
        end
    end

    return irr
end

const _OH2_IRREPS = _build_oh2_irreps()

# ---------- 子群指标（2O_h 96 元素索引） ----------

# O_h 子群指标直接映射到 2O_h：g_k → {k, k+48}（k 已在 1..48 范围内，含 proper 和 improper）
function _oh2_subgroup_indices(oh_indices::Vector{Int})
    v = Int[]
    for k in oh_indices
        append!(v, [k, k + 48])
    end
    return sort(v)
end

const _OH2_C4V_INDICES  = _oh2_subgroup_indices(_C4V_INDICES)
const _OH2_C2V_INDICES  = _oh2_subgroup_indices(_C2V_INDICES)
const _OH2_C3V_INDICES  = _oh2_subgroup_indices(_C3V_INDICES)

# ============ 小群不可约表示（运动系，从 Morningstar:2013bda 生成元构建）============

export LG_IRREP_NAMES

# ---- C_4v (8 元素), C_4v^d 双覆盖 (16 元素) ----
# 生成元: C = g_15 (C_{4z}), R = g_47 (I_s C_{2y})
# 元素: [g_1, g_14, g_15, g_24, g_42, g_43, g_46, g_47]
# 乘法: g_1=g_15^4, g_14=g_15^3, g_24=g_15^2, g_42=g_47·g_15, g_43=g_47·g_15^3, g_46=g_47·g_15^2

const C4V_BOSONIC_NAMES = ["A1", "A2", "B1", "B2", "E"]
const C4V_FERMIONIC_NAMES = ["G1", "G2"]
const C4V_IRREP_NAMES = vcat(C4V_BOSONIC_NAMES, C4V_FERMIONIC_NAMES)

# C_4v 元素生成元配方 (rotation_power, use_reflection)
const _C4V_RECIPES = [
    (4, 0),  # g_1  = C^4
    (3, 0),  # g_14 = C^3
    (1, 0),  # g_15 = C^1
    (2, 0),  # g_24 = C^2
    (1, 1),  # g_42 = g_47·g_15 → D(R)·D(C)
    (3, 1),  # g_43 = g_47·g_15^3 → D(R)·D(C)^3
    (2, 1),  # g_46 = g_47·g_15^2 → D(R)·D(C)^2
    (0, 1),  # g_47 = R
]

# C_4v^d 双覆盖元素生成元配方
const _C4VD_RECIPES = [
    (8, 0),  # g_1  = C^8
    (7, 0),  # g_14 = C^7
    (1, 0),  # g_15 = C^1
    (6, 0),  # g_24 = C^6
    (1, 1),  # g_42 = g_47·g_15 → D(R)·D(C)
    (3, 1),  # g_43 = g_47·g_15^3 → D(R)·D(C)^3
    (2, 1),  # g_46 = g_47·g_15^2 → D(R)·D(C)^2
    (0, 1),  # g_47 = R
]

const _C4V_GENERATORS = Dict{String, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(
    # (Γ(C_{4z}), Γ(I_s C_{2y}))
    "A1" => ([1.0;;],       [1.0;;]),
    "A2" => ([1.0;;],       [-1.0;;]),
    "B1" => ([-1.0;;],      [1.0;;]),
    "B2" => ([-1.0;;],      [-1.0;;]),
    "E"  => (ComplexF64[0 -1; 1 0],   ComplexF64[1 0; 0 -1]),
    "G1" => (ComplexF64[1-im 0; 0 1+im]/sqrt(2.0),  ComplexF64[0 -1; 1 0]),
    "G2" => (ComplexF64[-(1-im) 0; 0 -(1+im)]/sqrt(2.0), ComplexF64[0 -1; 1 0]),
)

# ---- C_3v (6 元素), C_3v^d 双覆盖 (12 元素) ----
# 生成元: C = g_3 (C_{3δ}), R = g_43 (I_s C_{2b})
# 元素: [g_1, g_2, g_3, g_41, g_43, g_45]
# 乘法: g_1=g_3^3, g_2=g_3^2, g_41=g_43·g_3, g_45=g_43·g_3^2

const C3V_BOSONIC_NAMES = ["A1", "A2", "E"]
const C3V_FERMIONIC_NAMES = ["F1", "F2", "G"]
const C3V_IRREP_NAMES = vcat(C3V_BOSONIC_NAMES, C3V_FERMIONIC_NAMES)

const _C3V_RECIPES = [
    (3, 0),  # g_1  = C^3
    (2, 0),  # g_2  = C^2
    (1, 0),  # g_3  = C^1
    (1, 1),  # g_41 = g_43·g_3 → D(R)·D(C)
    (0, 1),  # g_43 = R
    (2, 1),  # g_45 = g_43·g_3^2 → D(R)·D(C)^2
]

const _C3VD_RECIPES = [
    (6, 0),  # g_1  = C^6
    (5, 0),  # g_2  = C^5
    (1, 0),  # g_3  = C^1
    (1, 1),  # g_41 = g_43·g_3 → D(R)·D(C)
    (0, 1),  # g_43 = R
    (2, 1),  # g_45 = g_43·g_3^2 → D(R)·D(C)^2
]

const _C3V_GENERATORS = Dict{String, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(
    # (Γ(C_{3δ}), Γ(I_s C_{2b}))
    "A1" => ([1.0;;],       [1.0;;]),
    "A2" => ([1.0;;],       [-1.0;;]),
    "E"  => (ComplexF64[-1 sqrt(3); -sqrt(3) -1]/2.0, ComplexF64[-1 0; 0 1]),
    "F1" => ([-1.0;;],      [im;;]),
    "F2" => ([-1.0;;],      [-im;;]),
    "G"  => (ComplexF64[1-im -1-im; 1-im 1+im]/2.0, ComplexF64[0 1-im; -1-im 0]/sqrt(2.0)),
)

# ---- C_2v (4 元素), C_2v^d 双覆盖 (8 元素) ----
# 生成元: C = g_16 (C_{2e}), R = g_41 (I_s C_{2f})
# 元素: [g_1, g_16, g_41, g_46]
# 乘法: g_1=g_16^2, g_46=g_41·g_16

const C2V_BOSONIC_NAMES = ["A1", "A2", "B1", "B2"]
const C2V_FERMIONIC_NAMES = ["G"]
const C2V_IRREP_NAMES = vcat(C2V_BOSONIC_NAMES, C2V_FERMIONIC_NAMES)

const _C2V_RECIPES = [
    (2, 0),  # g_1  = C^2
    (1, 0),  # g_16 = C^1
    (0, 1),  # g_41 = R
    (1, 1),  # g_46 = g_41·g_16 → D(R)·D(C)
]

const _C2VD_RECIPES = [
    (4, 0),  # g_1  = C^4
    (1, 0),  # g_16 = C^1
    (0, 1),  # g_41 = R
    (1, 1),  # g_46 = g_41·g_16 → D(R)·D(C)
]

const _C2V_GENERATORS = Dict{String, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(
    # (Γ(C_{2e}), Γ(I_s C_{2f}))
    "A1" => ([1.0;;],       [1.0;;]),
    "A2" => ([1.0;;],       [-1.0;;]),
    "B1" => ([-1.0;;],      [1.0;;]),
    "B2" => ([-1.0;;],      [-1.0;;]),
    "G"  => (ComplexF64[-im -1; 1 im]/sqrt(2.0), ComplexF64[im -1; 1 -im]/sqrt(2.0)),
)

const LG_IRREP_NAMES = Dict(
    :C4v => C4V_IRREP_NAMES,
    :C3v => C3V_IRREP_NAMES,
    :C2v => C2V_IRREP_NAMES,
)

# ---- 从生成元构建所有群元矩阵 ----

function _build_from_generators(recipes::Vector{Tuple{Int,Int}},
                                 gen_C::Matrix{ComplexF64},
                                 gen_R::Matrix{ComplexF64})
    mats = Vector{Matrix{ComplexF64}}(undef, length(recipes))
    for (k, (cpow, rpow)) in enumerate(recipes)
        mats[k] = gen_R^rpow * gen_C^cpow
    end
    return mats
end

function _build_little_group_irreps_single(recipes::Vector{Tuple{Int,Int}},
                                            generators::Dict{String, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}})
    irr = Dict{String, Vector{Matrix{ComplexF64}}}()
    for (name, (gen_C, gen_R)) in generators
        irr[name] = _build_from_generators(recipes, gen_C, gen_R)
    end
    return irr
end

function _build_little_group_irreps_double(recipes::Vector{Tuple{Int,Int}},
                                            generators::Dict{String, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}},
                                            bosonic_names::Vector{String},
                                            fermionic_names::Vector{String})
    n = length(recipes)
    irr = Dict{String, Vector{Matrix{ComplexF64}}}()

    for (name, (gen_C, gen_R)) in generators
        base = _build_from_generators(recipes, gen_C, gen_R)
        mats = Vector{Matrix{ComplexF64}}(undef, 2 * n)
        is_fermionic = name in fermionic_names
        for k in 1:n
            mats[k] = base[k]
            mats[k + n] = (is_fermionic ? -1.0 : 1.0) * base[k]
        end
        irr[name] = mats
    end
    return irr
end

# 预构建所有小群不可约表示
const _C4V_IRREPS_SINGLE  = _build_little_group_irreps_single(_C4V_RECIPES, _C4V_GENERATORS)
const _C4VD_IRREPS        = _build_little_group_irreps_double(_C4VD_RECIPES, _C4V_GENERATORS, C4V_BOSONIC_NAMES, C4V_FERMIONIC_NAMES)
const _C3V_IRREPS_SINGLE  = _build_little_group_irreps_single(_C3V_RECIPES, _C3V_GENERATORS)
const _C3VD_IRREPS        = _build_little_group_irreps_double(_C3VD_RECIPES, _C3V_GENERATORS, C3V_BOSONIC_NAMES, C3V_FERMIONIC_NAMES)
const _C2V_IRREPS_SINGLE  = _build_little_group_irreps_single(_C2V_RECIPES, _C2V_GENERATORS)
const _C2VD_IRREPS        = _build_little_group_irreps_double(_C2VD_RECIPES, _C2V_GENERATORS, C2V_BOSONIC_NAMES, C2V_FERMIONIC_NAMES)

# 查找函数：irrep 属于哪个小群
function _find_lg_group(irrep::String)
    irrep in C4V_IRREP_NAMES && return :C4v
    irrep in C3V_IRREP_NAMES && return :C3v
    irrep in C2V_IRREP_NAMES && return :C2v
    return nothing
end

end # module

