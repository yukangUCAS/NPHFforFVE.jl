# ============================================================
# SpinCG — M 粒子自旋耦合系数 C^{J,[κ]}
# ============================================================
#
# 将 M 个 spin-j 单粒子自旋态耦合到具有确定总角动量 J 和 S_N 置换对称性 [κ] 的态：
#
#   |J,M,[κ],a,r⟩ = Σ_{m_i} C^{J,[κ]}_{m_1,...,m_N}(M,a,r) |j,m_1⟩...|j,m_N⟩
#
# 数学上与同位旋 CG 完全相同（SU(2)^⊗M → SU(2) × S_N 约化），
# 因此直接复用 charge_to_isospin_cg 的实现。

# Type alias to avoid parse issues with deeply nested Dict{...} annotation
const _SpinCGCoeffs{N} = Dict{Rational{Int}, Dict{Int, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}}

"""
    SpinCG{N}

存储 N 个 spin-j 粒子耦合到总角动量 J 和 S_N 不可约表示 [κ] 的 CG 系数。

|J,M,[κ],a⟩ = Σ_{m₁,...,m_N} coeff[(m₁,...,m_N)] |j,m₁⟩⊗...⊗|j,m_N⟩

字段:
- j: 单粒子自旋
- J: 总角动量
- kappa: S_N 不可约表示名称 (如 "[2]", "[1,1]")
- multiplicity: [κ] 在总角动量 J 下的重数
- dim_kappa: dim([κ])
- coeffs: coeffs[M][a][r] = Dict{m_tuple => coefficient}
  其中 M = -J, -J+1, ..., J
       a = 1, ..., dim([κ])
       r = 1, ..., multiplicity
"""
struct SpinCG{N}
    j::Rational{Int}
    J::Rational{Int}
    kappa::String
    multiplicity::Int
    dim_kappa::Int
    coeffs::_SpinCGCoeffs{N}
end

function Base.show(io::IO, cg::SpinCG{N}) where {N}
    n_states = sum(length(r_dict) for M_dict in values(cg.coeffs)
                   for a_dict in values(M_dict)
                   for r_dict in values(a_dict); init=0)
    print(io, "SpinCG (N=$N, j=$(cg.j), J=$(cg.J), [κ]=$(cg.kappa), mult=$(cg.multiplicity)): $n_states states")
end

"""
    spin_cg_coefficients(N::Int, j, J, kappa::String; multiplicity::Int=1) -> SpinCG{N}

返回 N 个 spin-j 粒子耦合到总角动量 J 和 S_N 不可约表示 [κ] 的 CG 系数。

# 参数
- N: 粒子数 (2, 3, 或 4)
- j: 单粒子自旋 (支持 1/2 或 1)
- J: 总角动量
- kappa: S_N 不可约表示名称，如 "[2]", "[1,1]", "[2,1]" 等
- multiplicity: 重数标号 r (当 [κ] 在 J 下出现多次时使用，默认为 1)

# 返回
SpinCG{N} 对象，其中 coeffs[M][a][r] 给出以 m_tuple 为键的系数字典。

# 示例
```julia
cg = spin_cg_coefficients(2, 1//2, 1//1, "[2]")
coeffs = cg.coeffs[1//1][1][1]  # M=1, a=1, r=1
# coeffs[(-1//2, -1//2)] ≈ 1.0  (|↑↑⟩)
```
"""
function spin_cg_coefficients(N::Int, j::Union{Rational{Int},Int},
                               J::Union{Rational{Int},Int},
                               kappa::String; multiplicity::Int=1)
    iso_cg = charge_to_isospin_cg(N, j, J, kappa; multiplicity=multiplicity)

    new_coeffs = Dict{Rational{Int}, Dict{Int, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}}()
    for (M, a_dict) in iso_cg.coeffs
        new_coeffs[M] = Dict{Int, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}()
        for (a, coeff_dict) in a_dict
            new_coeffs[M][a] = Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}()
            new_coeffs[M][a][1] = coeff_dict
        end
    end

    return SpinCG{N}(iso_cg.j, iso_cg.J, iso_cg.irrep,
                     iso_cg.multiplicity, iso_cg.dim_irrep, new_coeffs)
end

"""
    get_coeffs(cg::SpinCG{N}, M, a=1, r=1) where N -> Dict{NTuple{N, Rational{Int}}, ComplexF64}

返回指定 M, a, r 的 CG 系数字典。
"""
function get_coeffs(cg::SpinCG{N}, M::Rational{Int}, a::Int=1, r::Int=1) where N
    return cg.coeffs[M][a][r]
end
