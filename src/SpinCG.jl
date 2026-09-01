# ============================================================
# SpinCG — M-particle spin-coupling coefficients C^{J,[κ]}
# ============================================================
#
# Couple M single-particle spin-j states to states with definite total angular momentum J and S_N permutation symmetry [κ]:
#
#   |J,M,[κ],a,r⟩ = Σ_{m_i} C^{J,[κ]}_{m_1,...,m_N}(M,a,r) |j,m_1⟩...|j,m_N⟩
#
# Mathematically identical to isospin CG(SU(2)^⊗M → SU(2) × S_N reduction),
# therefore reuse the implementation of charge_to_isospin_cg directly.

# Type alias to avoid parse issues with deeply nested Dict{...} annotation
const _SpinCGCoeffs{N} = Dict{Rational{Int}, Dict{Int, Dict{Int, Dict{NTuple{N, Rational{Int}}, ComplexF64}}}}

"""
    SpinCG{N}

Store CG coefficients that couple N spin-j particles to total angular momentum J and S_N irrep [κ].

|J,M,[κ],a⟩ = Σ_{m₁,...,m_N} coeff[(m₁,...,m_N)] |j,m₁⟩⊗...⊗|j,m_N⟩

Fields:
- j: single-particle spin
- J: total angular momentum
- kappa: S_N irrep name (for example "[2]", "[1,1]")
- multiplicity: [κ] multiplicity at total angular momentum J
- dim_kappa: dim([κ])
- coeffs: coeffs[M][a][r] = Dict{m_tuple => coefficient}
  where M = -J, -J+1, ..., J
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

Return CG coefficients coupling N spin-j particles to total angular momentum J and S_N irrep [κ].

# Arguments
- N: number of particles (2, 3, or 4)
- j: single-particle spin (supports 1/2 or 1)
- J: total angular momentum
- kappa: S_N irrep name, for example "[2]", "[1,1]", or "[2,1]"
- multiplicity: multiplicity index r (used when [κ] occurs multiple times at J; default: 1)

# Returns
SpinCG{N} object, where coeffs[M][a][r] gives a coefficient dictionary keyed by m_tuple.

# Example
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

Return the CG-coefficient dictionary for specified M, a, r.
"""
function get_coeffs(cg::SpinCG{N}, M::Rational{Int}, a::Int=1, r::Int=1) where N
    return cg.coeffs[M][a][r]
end
